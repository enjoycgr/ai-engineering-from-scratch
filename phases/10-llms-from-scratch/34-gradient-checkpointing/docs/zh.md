# Gradient Checkpointing and Activation Recomputation（梯度检查点与激活重计算）

> Backprop 保留每个中间激活。在 70B 参数和 128K 上下文时，这是每个 rank 3 TB 的激活。Checkpointing 用 FLOP 交换内存：重计算而非保存。问题是丢弃哪些段，答案不是"全部"。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## The Problem

训练 transformer 存储每层每个被微分操作的输入：attention 输入、Q/K/V 投影、softmax 输出、FFN 输入、norm 输出和残差流。对于隐藏大小 `d`、序列长度 `L`、batch `B` 的层，这大约是每层 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，BF16 下每层 800 MB。64 层模型是 51 GB 的激活——这还没乘以 microbatch 大小，还没加 attention-softmax 中间值（每个 head `L^2`），还没考虑 tensor-parallel 部分拷贝。

双面账单：BF16 权重加优化器状态可能容纳在 80GB 中，但激活将你推过。Gradient checkpointing（又称 activation recomputation）是标准修复。丢弃大多数激活；在 backward 期间重做前向以取回它们。成本：额外 FLOP。收益：内存按检查点段与总层数的比率下降。

朴素地做，checkpointing 每步大约多花费 33% 的前向传递 FLOP。做得好——根据 Korthikanti 等人的"smart selection"——你用不到 5% 的 FLOP 开销节省 5 倍内存。随着 FP8 matmul、FSDP offload 和 expert-parallel MoE，这真的很重要：你既负担不起内存也负担不起浪费的计算。

## The Concept

### Backward 实际需要什么

`output = layer(input)`。Backward 想要 `grad_input` 和 `grad_params`。计算它们需要：

- `input`（为线性层计算 `grad_params = input.T @ grad_output`）
- 一些激活导数中间值（ReLU/GELU/softmax 的导数依赖于激活值）

前向传递自动在 autograd 图中存储这些。每个 `tensor.retain_grad()` 和每个需要输入的操作都保留引用。

### 朴素 Full Checkpointing

将网络分成 `N` 段。前向期间，只存储每段的*输入*。当 backward 需要中间值时，重新运行该段的前向以物化它们，然后微分。

例子：32 层 transformer 分成 32 段，每段 1 层。

- 内存：32 个层输入（小）vs 32 *（每层激活体积）（巨大）。
- 额外计算：每段 1 个额外前向，即总计约 33% 更多前向 FLOP（因为 backward 是前向的 2 倍，完整步骤变为 1 + 1 + 2 = 4 单位而非 1 + 2 = 3）。

这是原始 Chen 等人 2016 配方：每 `sqrt(L)` 层一个检查点以平衡内存和计算。对于 L=64，即 8 个检查点。

### Selective Checkpointing (Korthikanti 2022)

并非所有激活成本相同。Attention softmax 输出是 `B*L*L*heads`，随序列长度*二次方*增长。FFN 隐藏激活是 `B*L*4d`，线性增长。对于长序列，softmax 主导。

Selective checkpointing 保留廉价存储的激活（线性投影、残差）并只重计算昂贵的（attention）。你支付最小 FLOP 来重计算但节省 O(L^2) 内存。

Megatron-Core 将其作为"selective" activation recomputation 实现。用于大多数 2024+ 前沿训练运行。

### Offload

重计算的替代：在前向和 backward 之间将激活运送到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过 rematerialization 成本时有益。混合策略很常见：检查点一些层，offload 其他层。

FSDP2 将 offload 作为一流选项提供。当 GPU 受内存瓶颈但 CPU-GPU 传输有 headroom 时，offload 发光。

### Recompute Cost Model

每 `k` 层 naive checkpointing 出 `L` 层的每步 FLOP：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 段中每层一个额外前向
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性 checkpointing 只重计算 attention kernel，非整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Memory Savings Model

每层激活体积：`A`。对于 `L` 层，总激活内存：`L * A`。

Full checkpoint（段大小 1）：只存储 `L * input_volume`（标准 transformer 约 `L * 1/10 A`）。节省约 `9 * L * A * 1/10`。

每 `k` 层 checkpoint：存储 `L/k * A` 加上活动段内 `k-1` 层。

在 `k = sqrt(L)` 时，内存和重计算成本都随 `sqrt(L)` 缩放——均匀成本层的最优权衡。

### 何时不 Checkpoint

- 流水线阶段最内层已在运行中的。它们反正必须完成。
- 如果它们主导阶段计算的首层和末层（transformer 中罕见）。
- 已经使用 FlashAttention 的 attention kernel——Flash 已经快速重计算 softmax，所以额外层级 checkpointing 在其上增加很少。

### 实现模式

1. **函数包装器：** 将段包装在 `torch.utils.checkpoint.checkpoint(fn, input)` 中。PyTorch 只存储 `input`，backward 时重计算其他一切。

2. **基于装饰器：** 将层标记为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重计算：** 自己写 backward 传递，调用自定义 `recompute_forward`，用存储的输入复制前向。

三者给出相同的功能结果。包装器是标准习语。

### 与 TP / PP / FP8 的交互

- **Tensor parallel：** checkpoint 输入必须在重计算时 gather 或 rescatter；处理通信成本。
- **Pipeline parallel：** 典型模式是检查点每个流水线阶段的前向，使反向序 microbatch 可以重用激活内存。
- **FP8 recompute：** 重计算期间更新的 amax 历史必须匹配原始前向的，否则 FP8 尺度漂移。大多数框架快照尺度。

## Build It

### Step 1: 带段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Step 2: 需要所有激活的朴素 Backward

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Step 3: Checkpoint-Every-k 内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Step 4: Cost Model

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Step 5: Memory Estimator

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Step 6: Optimal Segment Size

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Step 7: Selective Checkpoint Decision

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Use It

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — PyTorch 中的规范包装器。包装函数；只存储输入，backward 时重计算。
- **Megatron-Core activation recomputation**: 支持 `selective`、`full` 和 `block` 模式。2024+ 前沿训练的标准。
- **FSDP2 offload**: `module.to_empty(device="cpu")` 与 FSDP2 中的 `offload_policy` 将激活分片到 CPU 而非重计算。
- **DeepSpeed ZeRO-Offload**: CPU offload 用于优化器状态和激活，补充 checkpointing。

## Ship It

本课产出 `outputs/prompt-activation-recompute-policy.md` — 一个 prompt，接收你的模型配置（层、隐藏、序列、batch）和可用 GPU 内存，并发出每层重计算策略（none / selective / full / offload）。

## Exercises

1. 验证正确性。运行 `model_forward` + `model_backward`（完整激活）vs `model_forward_checkpointed` + `model_backward_checkpointed`（段）。参数梯度必须在机器精度上相同。

2. 扫描段大小 `k` 从 1 到 `L`。绘制 FLOP 开销和内存。找到曲线的拐点。

3. 实现 selective checkpointing：存储 attention 模块输入但不存储其中间值。测量 32 层模型在 seq=8192 时相对于 full-layer checkpointing 的 FLOP 开销。

4. 添加 offload。将段输入保存到模拟"CPU 缓冲区"（单独列表）。测量"PCIe 带宽"为字节/时间，找到 offload 和 recompute 之间的盈亏平衡点。

5. 用和不用 `torch.utils.checkpoint` 基准测试真实 PyTorch transformer。测量内存（通过 `torch.cuda.max_memory_allocated`）和步骤时间。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Gradient checkpointing | "Save memory by redoing forward" | 只存储段输入；backward 期间重计算中间值以获取梯度支持张量 |
| Activation recomputation | "Same as checkpointing" | 同一技术的 HPC 风格名称 |
| Segment size (k) | "How many layers per checkpoint" | 丢弃和一起 rematerialize 中间值的层数 |
| Selective checkpointing | "Korthikanti's trick" | 只重计算昂贵存储的激活（attention softmax）；保留廉价的 |
| Full checkpointing | "The naive version" | 在每个段中重计算每层的中间值 |
| Block checkpointing | "Coarse-grained" | 检查点整个 transformer block；最大粒度 |
| FLOP overhead | "The compute tax" | 每步额外 FLOP =（重计算 FLOP）/（前向 + 后向 FLOP）；朴素 33%，选择性 5% |
| Activation offload | "Ship to CPU" | 在前向->后向之间将激活移动到 CPU RAM；重计算的替代 |
| sqrt-L rule | "The classical optimum" | 对于均匀成本层，最优检查点间距是 sqrt(L) 层 |
| Attention-softmax volume | "The O(L^2) problem" | L^2 * heads * batch 浮点数；在长上下文时主导激活内存 |

## Further Reading

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 将 gradient checkpointing 形式化的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- selective activation recomputation 和形式成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过 reverse-mode rematerialization 的替代恒定内存方法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大规模激活 offload
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- selective、full 和 block 模式
