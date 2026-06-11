# Differential Attention (V2)（差分注意力 V2）

> softmax attention（软最大值注意力）会在每个不匹配的 token 上分散少量概率。在 100k token 的上下文中，这些噪声会累积并淹没信号。Differential Transformer（Ye 等人，ICLR 2025）通过计算两个 softmax 的差值来解决这个问题，减去共享的噪声基底。DIFF V2（Microsoft，2026 年 1 月）是面向生产栈的重写版本：解码延迟与基线 Transformer 持平，无需自定义 kernel，兼容 FlashAttention。本课从 V1 到 V2 完整讲解，并提供一个可在标准库 Python 中运行的差分操作玩具实现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention（自注意力）), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## Learning Objectives

- 精确说明为什么 softmax attention 存在噪声基底，以及为什么它随上下文长度增长。
- 推导差分注意力公式，并解释减法为什么能抵消共享的噪声分量而保留信号。
- 走读 V1 到 V2 的 diff：哪些变快了、哪些简化了、哪些更稳定了，以及每个变更对生产预训练为何必要。
- 用纯 Python 从头实现差分注意力，并在合成信号加噪声查询上实证验证噪声抵消特性。

## The Problem

标准 softmax attention 有一个数学特性，在大规模下会变成操作上的麻烦。对于查询 `q`，注意力权重为 `softmax(qK^T / sqrt(d))`。Softmax 永远无法产生精确的零值——每个不匹配的 token 都会获得一些正质量。这部分残余质量就是噪声，并且随上下文长度缩放。在 128k token 时，即使每个不匹配的 token 只获得 0.001% 的概率，127,999 个 token 加起来也贡献了约 12% 的总量。模型必须学会绕过这个随上下文增长的噪声基底。

经验上，这表现为 attention head（注意力头）之间的干扰：长上下文 RAG 中的幻觉引用、100k token 检索任务上的 lost-in-the-middle（中间丢失）失败，以及 32k 以上 needle-in-haystack（大海捞针）基准的微妙精度下降。Differential Transformer 论文（arXiv:2410.05258，ICLR 2025）测量了这一差距：DIFF Transformer 在相同规模下比基线达到了更低的困惑度、更高的长上下文精度和更少的幻觉。

DIFF V1 有三个问题使其无法进入前沿预训练流水线。它的 value cache 在每个解码步骤需要加载两次，需要打破 FlashAttention 兼容性的自定义 CUDA kernel，以及其 per-head RMSNorm（逐头 RMS 归一化）在 70B 以上规模的长期训练中不稳定。DIFF V2（Microsoft unilm blog，2026 年 1 月 20 日）修复了所有三个问题。本课将走读两个版本，构建差分算子，并在玩具查询上基准测试噪声抵消。

## The Concept

### softmax 的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 精确为零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 不是 0——它以方差 `||q||^2 / d` 在零附近波动。经过 softmax 归一化后，每个无关 token 仍然贡献 `O(1/N)` 的加权和。无关 token 的总贡献是 `O((N-1)/N) = O(1)`——不是一个小量。

模型想要的是类似硬 top-k 的效果：匹配 token 上权重高，其他地方接近零。Softmax 本身太平滑，无法直接做到这一点。

### 差分思想

将每个 head 的 Q 和 K 投影分成两个：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法抵消了两个图共享的任何噪声分布。如果两个图在 127k 个无关 token 上大致均匀分布（在随机初始化时确实如此），这些会相互抵消。信号——在少数真正相关 token 上的尖峰权重——只有当它以相同幅度出现在两个图中时才会抵消，而模型训练后这不会发生。

`lambda` 是每个 head 的可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负数。`lambda_init` 默认为一个小的正数，如 0.8。

### 为什么这类似于降噪耳机

想象两个有噪声的麦克风录制同一个声音。两者都拾取了说话者加上相关的背景噪声。将其中一个从另一个中减去，共享的噪声就会消失。声音得以保留，因为两个信号在相位或幅度上差异足够大，不会完全抵消。每个 head 的 `lambda` 学习的就是这种平衡。

### V1 vs V2：差异

V1 保持参数数量与基线 Transformer 相等。为了每个 head 获得两个查询，它将 head dimension（头维度）减半。这损失了 head 表达能力——更痛苦的是——将每个 head 的 value cache 减半。解码时每个步骤需要加载两次 value cache（每个 softmax 分支一次）。结果：尽管参数数量匹配，解码速度比基线慢。

V2 将 query head（查询头）数量翻倍，保持 KV head 不变（从 up-projection 借用参数）。Head dimension 与基线相同。减法后，额外维度被投影回以匹配基线 Transformer 的 O_W 投影。三件事同时发生：

1. 解码速度匹配基线（KV cache 只加载一次）。
2. FlashAttention 无需改动即可运行（无需自定义 kernel）。
3. 解码时的 arithmetic intensity（算术强度）提高（每字节 HBM 加载有更多计算）。

V2 还移除了 V1 用于稳定减法的 per-head RMSNorm。在 70B 级预训练规模下，该 RMSNorm 在训练后期不稳定。V2 用一个更简单的初始化方案替代它，无需额外模块即可保持训练稳定。

### 何时使用

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | 更干净的注意力图，更少的幻觉引用 |
| Needle-in-haystack benchmarks | 32k 以上有显著的精度提升 |
| Multi-document QA | 更少的跨文档干扰 |
| Code completion at 8k | 边际效果，不值得架构变更 |
| Short chat (< 4k) | 与基线几乎无法区分 |

价值随上下文长度增长。在 4k token 时噪声基底足够小，标准 attention 即可。在 128k 时它会伤害你。

### 如何与其他 2026 年的技术叠加

| Feature | Compatible with DIFF V2? |
|---------|------------------------|
| GQA | Yes（V2 增加 Q head，不是 KV head） |
| MLA (DeepSeek) | 原则上可行，无已发表论文结合两者 |
| MoE | Yes（attention 独立于 MLP block） |
| RoPE | Yes（不变） |
| YaRN / long-context scaling | Yes（DIFF 最帮助的地方） |
| FlashAttention | V2 中 Yes（V1 中是 No） |
| Speculative decoding | Yes（attention 变更对 spec-decode 循环不可见） |

## Build It

`code/main.py` 用纯 Python 实现差分注意力。一个具有已知信号加噪声结构的玩具查询让你直接测量噪声抵消比率。

### Step 1: 标准 softmax attention

标准库矩阵操作：列表的列表，手动 matmul，数值稳定性 softmax（减去最大值）。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Step 2: 将 Q, K 分成两半

V1 风格：将 head dimension 减半。V2 风格：保持 head dimension 并翻倍 head 数量。玩具实现使用 V1 以获得教学清晰度——数学完全相同，只有簿记不同。

### Step 3: 两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以为负。这没问题——value cache 仍然处理有符号贡献。后续的 V 投影吸收符号。

### Step 4: 噪声抵消测量

构建一个长度为 1024 的合成序列。将信号 token 放在已知位置，其余填充噪声。计算 (a) 标准 softmax attention 在信号位置上的权重和 (b) 差分注意力权重。测量两者的信噪比。DIFF attention 可靠地产生比标准 attention 高 3x-10x 的信噪比，具体取决于两个分支被训练到多大程度不同。

### Step 5: V1 vs V2 参数核算

给定配置 (hidden=4096, heads=32, d_head=128)，打印：

- 基线 Transformer：Q, K, V 每个大小 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q, K 每个大小 `hidden * hidden`，V 大小 `hidden * hidden`（不变），内部 head dim 减半。增加 per-head `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小 `2 * hidden * hidden`，K 大小 `hidden * hidden`，V 大小 `hidden * hidden`。额外维度在 O_W 前投影回降。增加相同的 `lambda` 参数。

玩具测量 V2 的额外参数成本（每个 attention block 大约 `hidden * hidden` 额外）并打印它。

## Use It

截至 2026 年 4 月，DIFF V2 尚未在每个生产推理服务器中发货，但集成正在进行中，包括 vLLM 和 SGLang。同时，该模式出现在：

- Microsoft 内部长上下文生产模型。
- 多个针对 256k 以上上下文的开源模型训练运行的研究复现。
- 将 DIFF attention 与滑动窗口 attention 在交替层结合的混合架构。

2026 年何时使用：

- 从头训练一个针对 64k 以上有效上下文的新模型。从一开始就添加 differential attention；稍后重新训练代价高昂。
- 对一个长上下文模型进行 fine-tuning（微调），其中 lost-in-the-middle 失败主导你的评估。对 Q 投影的 LoRA 可以近似 DIFF 结构。

何时不使用：

- 你正在服务一个具有稳定长上下文性能的预训练密集模型。对现有权重重新训练很少能回本。
- 你的上下文始终在 16k 以下。噪声基底可忽略不计。

## Ship It

本课产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉配置文件和训练预算，它为在新预训练运行或 LoRA fine-tune 中添加 differential attention 生成集成计划。

## Exercises

1. 运行 `code/main.py`。验证差分注意力报告的信噪比高于合成查询上的标准 softmax attention。改变噪声幅度，展示标准 attention 变得不可用的交叉点。

2. 计算从基线到 DIFF V1 和从基线到 DIFF V2 的参数数量差异，针对一个 7B 级模型（hidden=4096, heads=32, d_head=128, 32 layers）。展示哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文的 Section 3（arXiv:2410.05258）和 DIFF V2 Hugging Face blog 的 Section 2。用两句话解释为什么 V1 的 per-head RMSNorm 是必要的，以及为什么 V2 可以在不导致训练发散的情况下去除它。

4. 实现一个消融：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全减法）计算差分注意力。在合成查询上，测量信噪比如何随扫描变化。确定最大化信噪比的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择 8 个 KV head 和 32 个 Q head。展示 KV cache 大小与具有相同 (8, 32) 配置的基线 GQA 模型匹配。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Differential attention | "Two softmaxes minus each other" | 将 Q, K 分成两半，计算两个 softmax 图，将第二个（按 lambda 缩放）从第一个中减去，然后乘以 V |
| Noise floor | "The non-zero tail of softmax" | softmax 在每个无关 token 上放置的 O(1/N) 权重，在长上下文中总和为 O(1) |
| lambda | "The subtraction scale" | 每个 head 的可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以为负 |
| DIFF V1 | "The ICLR 2025 version" | 原始 Differential Transformer；将 head dim 减半以保持参数数量，需要自定义 kernel，解码更慢 |
| DIFF V2 | "The January 2026 fix" | 翻倍 Q head 保持 KV head；匹配基线解码速度并与 FlashAttention 兼容 |
| Per-head RMSNorm | "The V1 stabilizer" | V1 在差值后应用的额外归一化；V2 移除它以防止训练后期不稳定 |
| Signal-to-noise ratio | "How much attention is wasted" | 真实信号位置上的权重与无关位置平均权重的比率 |
| Lost in the middle | "Long-context failure mode" | 检索精度在长上下文中间文档处下降的经验现象——DIFF attention 减少此现象 |
| Arithmetic intensity | "FLOPs per byte loaded" | V2 通过每个 KV 加载翻倍查询来提高解码时的比率；对内存受限解码很重要 |

## Further Reading

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — 原始论文，含噪声抵消理论和长上下文消融
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — 生产栈重写，匹配基线解码，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — 理论分析为什么减法恢复了预训练 attention 结构
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — 参数共享变体
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF 所减去的基线 Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF attention 针对的长上下文基准
