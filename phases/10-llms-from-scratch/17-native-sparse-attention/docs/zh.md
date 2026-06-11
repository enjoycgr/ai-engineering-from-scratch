# Native Sparse Attention (DeepSeek NSA)（原生稀疏注意力）

> 在 64k token 时，attention 消耗 70-80% 的解码延迟。每个开源模型实验室都有修复计划。DeepSeek 的 NSA（ACL 2025 最佳论文）是最终胜出的方案：三个并行 attention 分支——压缩的粗粒度 token、选择性保留的细粒度 token，以及用于局部上下文的滑动窗口——通过可学习的门控组合。它是硬件对齐的（对 kernel 友好）、原生可训练的（在预训练中工作，而非在推理时 bolt on），并且在 64k 解码时比 FlashAttention 更快，同时匹配或超越完整 attention 的质量。本课端到端构建这三个分支，并展示为什么稀疏性是端到端可微的。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## Learning Objectives

- 说明 NSA 的三个 attention 分支以及每个分支捕获的内容。
- 解释为什么 NSA 是"原生可训练的"，而先前的稀疏 attention 方法只能是推理时应用。
- 计算在 64k 上下文中 NSA 与完整 attention 的 attention 计算节省，作为压缩块大小和选择 top-k 的函数。
- 在标准库 Python 中用短合成序列实现三分支组合，并验证门控权重行为正常。

## The Problem

完整 attention 在序列长度 N 上花费 `O(N^2)` 时间和每层 `O(N)` 的 KV cache。在 64k token 时，计算和内存带宽数字是灾难性的。NSA 论文的测量理论估计：在 64k 时，attention 占总解码延迟的 70-80%。下游的一切——TTFT、tokens/sec、每百万 token 成本——都由 attention 成本主导。

稀疏 attention 是显而易见的答案。先前的尝试分为两类。固定模式稀疏性（滑动窗口、stride（步长）、块局部）会丢弃信息，在长距离召回任务上失败。推理时稀疏性（KV cache 剪枝、H2O、StreamingLLM）应用于在密集 attention 上预训练的模型，只能回收潜在加速的一小部分，因为模型从未被要求通过稀疏模式路由信息。

Native Sparse Attention（Yuan 等人，DeepSeek + PKU + UW，ACL 2025 最佳论文，arXiv:2502.11089）两者兼顾：模型在预训练期间学习的稀疏模式，以 kernel 对齐算法实现，在推理时实际交付计算节省。两年后，NSA 或其直接后代将是每个前沿长上下文模型的默认 attention。

## The Concept

### 三个并行分支

对于每个查询，NSA 运行三次 attention，针对 KV cache 的三个不同视图：

1. **Compressed branch（压缩分支）。** Token 被分组为大小 `l`（通常为 32 或 64）的块。每个块通过一个小型可学习的 MLP 压缩成单个摘要 token。查询在这些压缩 token 上进行 attention，获得整个序列的粗粒度视图。

2. **Selected branch（选择分支）。** 使用压缩分支的 attention 分数，识别与当前查询最相关的 top-k 块。读取这些块的细粒度（未压缩）token，查询对它们全部进行 attention。将压缩分支 attention 视为选择的路由信号。

3. **Sliding-window branch（滑动窗口分支）。** 查询对最近的 `W` 个 token（通常为 512）进行 attention 以获得局部上下文。该分支捕获其他两个可能遗漏的结构化短距离模式（语法、局部共指）。

三个分支输出通过可学习的逐位置门控组合：

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` 是来自查询上小型 MLP 的门控权重。它们不必总和为 1——可以独立加权分支。

### 为什么这是"原生可训练的"

选择步骤（top-k 块）是离散的。离散操作破坏梯度流。先前的稀疏 attention 工作要么跳过通过选择的反向传播（限制训练），要么使用在推理时无法提供真正稀疏性的连续松弛。

NSA 巧妙地避开了这一点：压缩分支 attention 是整个序列上的可微粗粒度 attention。Top-k 操作只是重用压缩分支的最高 attention 分数来选择加载哪些细粒度块。梯度通过压缩分支分数流动（这些分数同时影响压缩输出 AND 选择逻辑），并且选定块对最终输出的贡献也是可微的。不可微的 `top_k` 操作在前向计算图上是一个无操作——它只控制从内存加载哪些块。

这就是 NSA 可以端到端用于预训练的原因。模型联合学习通过三个分支路由信息，产生一个在推理时实际交付承诺加速的稀疏模式。

### 硬件对齐的 kernel

NSA 的 kernel 为现代 GPU 内存层次结构设计。Kernel 按 GQA 组加载查询（外循环），获取每组的相应稀疏 KV 块（内循环），并在 SRAM 上运行 attention。因为每个查询组看到相同的选定块（选择是逐查询组而非逐查询头），KV 加载在组内摊销。Arithmetic intensity（算术强度）保持高位。

论文报告 Triton kernel 在 64k 解码时比 FlashAttention 快 9 倍，加速比随序列长度增长。提供前向和后向 kernel。

### 计算预算

设 `N` 为序列长度，`l` 为压缩块大小，`k` 为 top-k 选择计数，`w` 为滑动窗口，`b` 为选定块大小（通常等于 `l`）。

- 压缩分支：每个查询 `O(N/l)` 个键，所以总计 `O(N * N / l)`。
- 选择分支：每个查询 `O(k * b)` 个键，所以总计 `O(N * k * b)`。
- 滑动分支：每个查询 `O(w)` 个键，所以总计 `O(N * w)`。

总计：`O(N * (N/l + k*b + w))`。

在 `N = 64k, l = 64, k = 16, b = 64, w = 512` 时：每个查询成本为 `1000 + 1024 + 512 = 2536` 个键。完整 attention 是 `64000` 个键。25 倍计算减少。

在 `N = 128k, l = 64, k = 16, b = 64, w = 512` 时：每个查询成本为 `2000 + 1024 + 512 = 3536` 个键。完整 attention 是 `128000` 个键。36 倍减少。收益随序列长度增长，这正是要点。

### 如何比较

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| Sliding window only | yes | yes | fails |
| Strided / block-sparse | yes | yes | partial |
| KV pruning (H2O, StreamingLLM) | N/A (inference-time) | yes | partial |
| MoBA (Moonshot) | partial | yes | good |
| NSA | yes (natively) | yes (9x at 64k) | matches full attention |

MoBA（Moonshot，arXiv:2502.13189）同时发表，采用类似的"三比一好"方法，将 MoE 原理应用于 attention 块。NSA 和 MoBA 是 2026 年长上下文预训练需要了解的两种架构。

## Build It

`code/main.py` 在短合成序列上实现三个分支并展示：

- 压缩 MLP（为教学清晰度使用简单的 mean-pool 基线；真正的 NSA 使用可学习的 MLP）。
- 由压缩分支分数驱动的 top-k 块选择。
- 最后 `w` 个 token 上的滑动窗口 attention。
- 门控组合。
- 与完整 attention 的计算计数打印输出。

### Step 1: 将 token 压缩成块

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2: 压缩分支 attention

对压缩键运行查询的 softmax attention。压缩分支分数同时充当 top-k 选择的信号。

### Step 3: top-k 块选择

选取 `k` 个最高分数压缩块的索引。从这些块加载原始未压缩 token 并对它们运行 attention。

### Step 4: 滑动窗口 attention

取最后 `w` 个 token 并对它们运行标准 attention。

### Step 5: 门控 + 组合

查询上的小型 MLP 产生三个门控权重。最终输出是三个分支输出的加权和。

### Step 6: 计算计数

打印每个分支每个查询的 attention 键数量以及总计。与 `N`（完整 attention）比较。在 1024 token 合成序列上，`l = 32, k = 4, w = 128`，NSA 每个查询看到 `32 + 128 + 128 = 288` 个键，而完整 attention 为 1024——少 3.5 倍。

## Use It

NSA 正在 DeepSeek 自己的长上下文预训练流水线中发货。截至 2026 年 4 月的公共推理栈集成状态：

- **DeepSeek 内部**：原生，发布的权重使用 NSA 或其继任者 DSA（Deepseek Sparse Attention）。
- **vLLM**：针对 DeepSeek-V3.x 权重的实验性 NSA 支持正在开发中。
- **SGLang**：已发布 NSA 基准测试；生产路径跟随 vLLM。
- **llama.cpp / CPU**：不支持；kernel 分解的开销在 CPU 吞吐量下不值得。

何时使用 NSA：

- 针对 64k 以上上下文的预训练或继续训练运行，具有严肃的计算预算。
- DeepSeek 自己长上下文检查点的推理。权重是 NSA 原生的。

何时不使用：

- 服务现有的密集 attention 预训练模型。没有继续训练就无法 retrofit NSA。
- 16k 以下的上下文。三分支开销主导节省。
- Batch-1 交互式聊天。延迟敏感解码有收益，但只在长上下文时。

## Ship It

本课产出 `outputs/skill-nsa-integrator.md`。给定长上下文预训练运行规范，它生成 NSA 集成计划：压缩块大小、top-k、滑动窗口、门控 MLP 宽度、kernel 选择，以及证明架构变更合理性的特定长上下文评估。

## Exercises

1. 在 1024 token 合成序列上运行 `code/main.py`。在三个预设上扫描 `(l, k, w)` 并打印计算计数。识别在 needle-in-haystack 测试上保持 95% 召回率的同时实现每个查询最低键数的预设。

2. 将 mean-pool 压缩器替换为微型可学习 MLP（2 层，隐藏层 32）。在一个信号为块平均值的合成任务上训练它。测量在留出数据上相对于 mean-pool 基线的困惑度差距。

3. 实现门控 MLP。它以查询为输入并输出三个标量。展示门控行为合理：随机查询上接近均匀加权，当查询命中远后块时在选择分支上权重较重。

4. 计算在 128k 上下文中启用 NSA 的 70B 模型的 KV cache 内存预算。KV head 为 8，head dim 128，BF16。与完整 attention 和 MLA（Phase 10 · 14 展示了 MLA 的数字）比较。识别 NSA 细粒度分支 KV cache 等于完整 attention 的序列长度。

5. 阅读 NSA 论文的 Section 4（arXiv:2502.11089），用三句话解释为什么压缩分支的 attention 分数被重用用于 top-k 选择，而不是计算单独的路由分数。将答案与梯度流联系起来。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Compressed branch | "Coarse view" | 对块平均键的 attention，以每个查询 O(N/l) 个键提供全局上下文 |
| Selected branch | "Top-k blocks" | 对 `k` 个最高压缩分支分数块的细粒度 attention |
| Sliding window | "Local context" | 对最后 `W` 个 token 的 attention，用于短距离模式 |
| Native trainability | "Pre-train with the sparsity on" | 稀疏模式在预训练期间学习，不是在推理时 bolt on |
| Compression block size l | "Group size for coarse view" | 多少个 token 合并成一个摘要；典型 32-64 |
| Top-k | "Blocks to keep" | 其未压缩 token 被读取的压缩块数量；典型 16 |
| Sliding window W | "Local attention radius" | 典型 512；更短损害局部连贯性，更长浪费计算 |
| Branch gate | "How to mix the three" | 逐位置 MLP 输出，加权三个分支的贡献 |
| Hardware alignment | "Kernel-friendly sparsity" | 选择稀疏模式以便实际 GPU kernel 达到理论加速 |
| DSA | "NSA's successor" | Deepseek Sparse Attention，DeepSeek  lineage 中 NSA 之后的架构 |

## Further Reading

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — 论文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA 针对的架构家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — 同期工作，MoE 风格的块 attention
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — 滑动窗口起源
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA 改进的推理时稀疏性基线
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA kernel 在 64k 时击败的完整 attention 基线
