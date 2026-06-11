# 混合专家模型（Mixture of Experts, MoE）

> 一个稠密（dense）的 70B Transformer 会为每个 token 激活所有参数。而一个 671B 的 MoE 每个 token 只激活 37B 参数，却在所有基准测试上击败了前者。稀疏性（sparsity）是过去十年最重要的扩展思想。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer），Phase 7 · 07（GPT）
**时间：** 约 45 分钟

## 问题背景

稠密 Transformer 的推理 FLOPs 等于其参数量（前向传播乘以 2）。扩展稠密模型时，每个 token 都要付出完整代价。到 2024 年，前沿模型撞上了算力墙：要想显著提升智能水平，每个 token 需要的 FLOPs 呈指数级增长。

混合专家模型（Mixture of Experts, MoE）打破了这一关联。将每个前馈网络（FFN）替换为 `E` 个独立专家（expert）加一个路由器（router），路由器为每个 token 挑选 `k` 个专家。总参数量 = `E × FFN_size`。每个 token 的激活参数量 = `k × FFN_size`。2026 年的典型配置：`E=256`，`k=8`。存储随 `E` 扩展，计算随 `k` 扩展。

2026 年的前沿模型几乎全是 MoE：DeepSeek-V3（671B 总参 / 37B 激活）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，前 10 名开源模型全是 MoE。

## 核心概念

![MoE 层：路由器为每个 token 从 E 个专家中选出 k 个](../assets/moe.svg)

### FFN 替换

稠密 Transformer 块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 每个 token 从 E 个中选 k 个
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每个专家都是一个独立的前馈网络（FFN，通常为 SwiGLU）。路由器是一个单独的线性层（linear layer）。每个 token 自主选择 `k` 个专家，并获得它们输出的门控加权和。

### 负载均衡问题

如果路由器把 90% 的 token 都送到专家 3，其他专家就会"饿死"（starve）。人们尝试过三种修复方案：

1. **辅助负载均衡损失（Auxiliary load-balancing loss）**（Switch Transformer、Mixtral）。添加一个与专家使用方差成比例的惩罚项。有效，但增加了一个超参数和第二条梯度信号。
2. **专家容量 + token 丢弃（Expert capacity + token dropping）**（早期 Switch）。每个专家最多处理 `C × N/E` 个 token；溢出的 token 跳过该层。损害质量。
3. **无辅助损失均衡（Auxiliary-loss-free balancing）**（DeepSeek-V3）。添加一个可学习的逐专家偏置（bias），仅影响路由器的 top-k 选择。偏置在训练损失之外更新。不对主目标施加惩罚。这是 2024 年的重大突破。

DeepSeek-V3 的做法：每个训练步骤后，对每个专家检查其使用量是否高于或低于目标值。将偏置按 `±γ` 进行微调。选择时使用 `scores + bias`。用于门控的专家概率保持原始 `scores` 不变。将路由与表达解耦。

### 共享专家（Shared experts）

DeepSeek-V2/V3 还将专家分为*共享专家（shared）*和*路由专家（routed）*。每个 token 都会经过所有共享专家。路由专家通过 top-k 选择。共享专家捕获通用知识；路由专家负责专业化。V3 使用 1 个共享专家 + 从 256 个路由专家中选 top-8。

### 细粒度专家（Fine-grained experts）

经典 MoE（GShard、Switch）：每个专家和一个完整 FFN 一样宽。`E` 较小（8–64），`k` 较小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家更窄（1/8 FFN 大小）。`E` 很大（256+），`k` 更大（8+）。总参数量相同，但组合空间急剧扩大。`C(256, 8) = 400 万亿` 种可能的"专家"组合 per token。质量上升，延迟保持不变。

### 成本分析

每个 token、每层：

| 配置 | 每个 token 的激活参数量 | 总参数量 |
|------|-----------------------|----------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B（稠密） | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2（MoE） | ~32B | 1T |

DeepSeek-V3 在几乎所有基准测试上击败了 Llama 3 70B（稠密），同时**每个 token 的激活 FLOPs 更少**。更多参数 = 更多知识。更多激活 FLOPs = 每个 token 更多计算。MoE 将两者解耦。

### 代价：内存

所有专家都驻留在 GPU 上，无论是否被激活。一个 671B 模型在 fp16 权重下需要约 1.3 TB 显存。前沿 MoE 部署需要专家并行（expert parallelism）—— 将专家分片到多个 GPU，通过网络路由 token。延迟主要由 all-to-all 通信主导，而非矩阵乘法。

## 动手构建

见 `code/main.py`。一个纯标准库的紧凑 MoE 层，包含：

- `n_experts=8` 个类 SwiGLU 专家（各一个线性层，用于演示）
- top-k=2 路由
- softmax 归一化的门控权重
- 通过逐专家偏置实现无辅助损失均衡

### 步骤 1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 对所选专家的原始分数做 softmax
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏置影响选择，不影响门控权重。这就是 DeepSeek-V3 的技巧 —— 偏置纠正负载不均衡，但不会引导模型的预测。

### 步骤 2：让 100 个 token 通过路由器

追踪每个专家的触发频率。没有偏置时，使用量偏斜。加入偏置更新循环（过度使用的专家 `-γ`，使用不足的 `+γ`），使用量在几次迭代后收敛到均匀分布。

### 步骤 3：参数量对比

打印 MoE 配置的"稠密等效值"。DeepSeek-V3 规格：256 路由 + 1 共享，8 激活，d_model=7168。总参数量令人瞠目。激活量只有稠密 Llama 3 70B 的七分之一。

## 实际使用

HuggingFace 加载：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年生产推理：vLLM 原生支持 MoE 路由。SGLang 拥有最快的专家并行路径。两者都自动处理 top-k 选择和专家并行。

**何时选择 MoE：**
- 你想以更低的每 token 推理成本获得前沿质量。
- 你有显存 / 专家并行基础设施。
- 你的工作负载是 token 密集型（聊天、代码），而非上下文密集型（长文档）。

**何时不选择 MoE：**
- 边缘部署 —— 你为任何激活 FLOP 付出完整存储代价。
- 延迟敏感的单用户服务 —— 专家路由增加开销。
- 小模型（<7B）—— MoE 的质量优势只在算力阈值以上显现（约 6B 激活参数）。

## 部署上线

见 `outputs/skill-moe-configurator.md`。该 skill 根据参数预算、训练 token 和部署目标，为新 MoE 选择 E、k 和共享专家布局。

## 练习题

1. **简单。** 运行 `code/main.py`。观察无辅助损失偏置更新如何在 50 次迭代内使专家使用量趋于均匀。
2. **中等。** 将学习路由器替换为基于哈希的路由器（确定性，无需学习）。对比质量和均衡性。为什么学习路由器更好？
3. **困难。** 实现 GRPO 风格的" rollout 匹配路由"（DeepSeek-V3.2 技巧）：记录推理时哪些专家被激活，在梯度计算时强制使用相同路由。在玩具策略梯度设置上测量其效果。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| Expert（专家） | "众多 FFN 之一" | 一个独立的前馈网络；参数专用于 FFN 计算的稀疏切片。 |
| Router（路由器） | "门控" | 一个微型线性层，为每个 token 对每个专家打分；top-k 选择。 |
| Top-k routing（Top-k 路由） | "每个 token 激活 k 个专家" | 每个 token 的 FFN 计算恰好经过 k 个专家，按门控加权。 |
| Auxiliary loss（辅助损失） | "负载均衡惩罚" | 额外的损失项，惩罚专家使用量的偏斜。 |
| Auxiliary-loss-free（无辅助损失） | "DeepSeek-V3 的技巧" | 仅通过路由器选择上的逐专家偏置实现均衡；无额外梯度。 |
| Shared expert（共享专家） | "始终开启" | 每个 token 都会经过的额外专家；捕获通用知识。 |
| Expert parallelism（专家并行） | "按专家分片" | 将不同专家分布到不同 GPU；通过网络路由 token。 |
| Sparsity（稀疏性） | "激活参数 < 总参数" | 比率 `k × expert_size / (E × expert_size)`；DeepSeek-V3 约为 37/671 ≈ 5.5%。 |

## 延伸阅读

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) —— 原始思想。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) —— Switch，经典 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) —— Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) —— MLA + 无辅助损失 MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) —— 基于偏置的均衡论文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) —— 本课路由器使用的细粒度 + 共享专家拆分。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) —— 原始共享专家论文。
