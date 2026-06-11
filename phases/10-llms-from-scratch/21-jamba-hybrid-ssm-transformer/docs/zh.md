# Jamba — Hybrid SSM-Transformer（混合 SSM-Transformer）

> State space models (SSM)（状态空间模型）和 transformer 想要不同的东西。Transformer 以二次方成本购买质量。SSM 以线性时间推理和恒定内存购买质量，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）将它们放在同一个模型中：每 7 个 Mamba 层配 1 个 Transformer 层，每隔一个 block 使用 MoE，以及一个 256k 上下文窗口，可放入单个 80GB GPU。Mamba-3（ICLR 2026）通过复数值状态空间和 MIMO 投影收紧了 SSM 侧。本课从头到尾阅读两种架构，并解释为什么混合配方在纯 SSM 和纯 Transformer 长上下文尝试都失败时，已经存活了三年扩展。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## Learning Objectives

- 解释 Jamba block 中的三个原语——Transformer 层、Mamba 层、MoE——以及 1:7:even 交错配方。
- 高层面说明 SSM 的递推形式以及为什么它实现恒定内存推理。
- 计算 Jamba 模型在 256k 上下文时的 KV cache 占用，并与纯 Transformer 模型需要的比较。
- 命名三个 Mamba-3 创新（exponential-trapezoidal discretization、complex-valued state update、MIMO）以及每个针对的问题。

## The Problem

Attention 在序列长度上是二次方的。State space model 是线性的。这种差异会复合：在 256k token 时，Transformer attention 图是每个 head 65B 条目；SSM 的递推状态无论序列长度如何都是固定大小。

纯 SSM 模型（Mamba、Mamba-2）在小规模时匹配 Transformer 的困惑度，但在状态跟踪任务上落后，并在某些类别的上下文内检索上失败。直觉：SSM 将历史压缩成固定状态，当历史很长时，信息会泄漏。Attention 精确记住一切但支付二次方成本。

显而易见的修复：两者都用。在精确召回重要的地方放 Transformer 层。在其他地方使用 SSM 层。调整比率。Jamba 是第一个以规模发布这种混合配方的生产级模型（52B 总计，12B 活跃，256k 上下文，单个 80GB GPU）。Jamba 1.5 将家族扩展到 398B 总计 / 94B 活跃。Mamba-3（ICLR 2026）是当前最佳纯 SSM 基线，混合模型可以围绕它重建。

本课阅读所有三篇论文并产生"选择正确比率"的心智模型。

## The Concept

### 一页 SSM

状态空间模型通过固定大小状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步状态通过线性动力学 `A` 演化，接收输入 `B x_t`，并发射输出 `C h_t`。`A, B, C` 可以学习。注意关键特性：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。内存恒定。推理为每个 token O(1)。

建模质量的技巧在于 `A` 的结构。S4（Gu 2021）使用高度结构化的矩阵，可以在训练期间作为长卷积高效求值。Mamba（Gu, Dao 2023）用数据相关的 `A, B, C` 替代固定的（"选择性"部分）。Mamba-2（2024）进一步简化结构。Mamba-3（2026）在特定位置重新添加复杂性。

关键特性：对于 decoder LLM，SSM 层是 attention 层的即插即用替代，用固定大小的每层状态替代增长的 KV cache。

### Jamba block

Jamba block 根据两个数字交错层：

- `l`：attention 到 Mamba 的比率。Jamba 使用 `l = 8`，意味着每 7 个 Mamba 层配 1 个 Transformer 层（7 个 Mamba + 1 个 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，意味着每隔一层应用 MoE。

Block 内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba block 是 8 层。在 4 个 block 深度（32 层总计），你得到 28 个 Mamba 和 4 个 Attention 层。其中 16 个使用 MoE。

### 为什么 1:7 比率

AI21 运行消融：什么 attention-to-Mamba 比率在 perplexity-per-parameter 和它们的长上下文评估上的上下文内召回方面最好？

- 太多 attention（1:1）：质量上升但内存和速度下降。
- 太少 attention（1:15）：内存很好但上下文内检索失败。
- 甜点：1:7 或 1:8。

直觉：Transformer 层处理精确召回和状态跟踪。Mamba 层处理廉价的批量处理。

### 位置编码

Mamba 层本身是位置感知的（通过递推）。原始基于 Mamba 的混合模型中的 attention 层不使用 RoPE——SSM 层提供位置信息。Jamba 1.5 为 attention 层添加 RoPE 以实现更长上下文泛化，这是基于经验长上下文评估的事后改进。

### 内存预算

对于 Jamba-1 形状（32 层：28 个 Mamba + 4 个 Attention，隐藏 4096，32 个 attention head）：

- KV cache（仅 attention 层）：`2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`，256k BF16。只有 4 个 attention 层贡献。
- SSM state：`28 * hidden * state_size` 每个 token 前缀，但这是每层固定大小，不随序列长度缩放。典型 Mamba state 是每个特征 16，隐藏 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

与 32 层纯 Transformer 比较，相同隐藏，32 个 head 的完整 MHA：`2 * 32 * 32 * 128 * 256k * 2 = 128 GB`，256k BF16。KV cache 减少 8 倍。即使对抗大多数 2024 模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），Jamba 的 1:7 混合在 16 GB 时仍然小 2 倍。

这就是 AI21 所说的"256k 上下文在单个 80GB GPU 上"。完整 MHA 纯 Transformer 的 KV cache 无法容纳；即使 GQA 基线也没有空间容纳权重和激活；Jamba 的可以。

### Mamba-3：2026 年纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 侧引入三个创新：

1. **Exponential-trapezoidal discretization。** 将 Mamba-2 中的 Euler-method discretization 替换为更具表达性的递推。在核心递推内对状态-输入应用类卷积操作，而不是作为 `x_t` 的外卷积。

2. **Complex-valued state update。** 先前的 Mamba 将状态矩阵从复数（S4）减少到实对角线（Mamba）再到缩放单位矩阵（Mamba-2）。Mamba-3 重新添加复数值——等效于状态上的数据相关 rotary embedding。这恢复了先前实数值简化所损失的状态跟踪能力。

3. **Multi-input multi-output (MIMO) projections。** 替代每个特征的标量投影，使用矩阵值投影。在不增加解码延迟的情况下提高建模能力和推理时硬件利用率。

在 1.5B 参数下，Mamba-3 在平均下游精度上比 Gated DeltaNet 提高 0.6 点；MIMO 变体再增加 1.2 点，总共 1.8 点增益。在相同状态大小下，Mamba-3 用一半状态匹配 Mamba-2。

Mamba-3 尚未以规模发布在生产混合模型中——但它是下一个 Jamba 类模型 SSM 侧的明显候选者。

### 何时使用混合

混合模型在以下情况获胜：

- 上下文足够长，纯 Transformer KV cache 变得痛苦（64k+）。
- 任务混合短距离结构（适合 SSM）与长距离召回（需要 Transformer）。
- 你想在单 GPU 内存预算上部署，Transformer KV cache 单独无法容纳。

混合模型在以下情况失败：

- 上下文短（16k 以下）。SSM 开销浪费；纯 Transformer 即可。
- 任务需要 everywhere-to-everywhere attention（深度推理、多文档交叉引用）。混合中 attention 层的稀疏性伤害性能。
- 你正在扩展到万亿参数前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前正在赢得能力竞赛。

### 竞争格局

| Model | Family | Scale | Unique claim |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

2026 年格局：纯 Transformer MoE 主导前沿，但混合模型拥有 256k 以上上下文利基。Mamba-3 的状态跟踪胜利可能推动下一代混合比率更低（更多 SSM，更少 attention）。

## Use It

`code/main.py` 是混合架构的内存计算器。给定 SSM-Transformer 比率和隐藏大小 / 层数配置，它计算：

- 目标上下文时的 KV cache。
- SSM state 内存。
- 上下文 N 时一系列模型形状的总内存。

计算器支持：

- 纯 Transformer 基线（KV cache 随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（无 KV cache）。

数字直接来自 Jamba-1 和 Jamba-1.5 论文的发布形状，并为假设变体外推。

真实部署的集成考虑：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。检查特定版本。
- 在 256k 上下文时，Jamba 的内存优势体现在并发请求吞吐量上。在相同 VRAM 上，你比 Transformer 序列容纳更多 Jamba 序列。
- Mamba-3 作为独立模型尚未在生产中发布——1.5B 的研究预览。

## Ship It

本课产出 `outputs/skill-hybrid-picker.md`。给定工作负载规范（上下文长度分布、任务混合、内存预算），它推荐纯 Transformer、Jamba 风格混合和纯 SSM 之间的选择，并明确说明内存和质量权衡的推理。

## Exercises

1. 运行 `code/main.py` 计算 32 层纯 Transformer（隐藏 4096，32 个 head）和相同形状的 Jamba-1 混合在 256k 上下文时的 KV cache。验证 AI21 论文声称的约 8 倍内存减少。

2. 修改计算器以建模 1:3 混合（4 个 Mamba : 1 个 Attention）和 1:15 混合（14 个 Mamba : 1 个 Attention）。绘制 KV cache vs 比率。在什么比率下 KV cache 等于 SSM state 内存？

3. 阅读 Jamba 论文（arXiv:2403.19887）的 Section 3。解释为什么 AI21 使用 Mamba-1 而非 Mamba-2，尽管 Mamba-2 更快。提示：混合消融部分记录了这一点。

4. 计算 Jamba 1.5 Large（398B 总计，94B 活跃）中每隔一层 MoE 的参数开销。将活跃比率与 DeepSeek-V3（37B/671B）比较，并解释为什么 Jamba 的架构推动活跃比率更高。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）的 Section 3。用三句话解释为什么复数值状态更新等效于数据相关的 rotary embedding。将答案与 Phase 7 · Lesson 04 的 RoPE 推导联系起来。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| State space model (SSM) | "Recurrence with a fixed state" | 具有学习递推 `h_t = A h_{t-1} + B x_t` 的层；每个 token 恒定内存 |
| Selective SSM | "Mamba's trick" | 数据相关的 A, B, C 参数，给模型类似门控的选择性，线性时间 |
| Attention-to-Mamba ratio | "How many attention layers" | 在 Jamba 中，`l = 8` 意味着每 7 个 Mamba 层配 1 个 attention 层 |
| Jamba block | "The 8-layer group" | 一个 attention + 七个 Mamba + 交替位置上的 MoE |
| SSM state | "The hidden buffer" | 替代 Mamba 层 KV cache 的固定大小每层状态 |
| 256k context | "Jamba's flagship number" | Jamba-1 在单个 80GB GPU 上容纳的序列长度；该大小下纯 Transformer 无法容纳 |
| Mamba-3 | "2026 pure SSM" | 当前最佳纯 SSM 架构，具有复数状态 + MIMO；混合模型重建的基线 |
| MIMO | "Multi-input multi-output" | Mamba-3 创新，使用矩阵值投影替代标量每特征 |
| Exponential-trapezoidal discretization | "Mamba-3's recurrence" | 更具表达性的递推，包含 Mamba-2 的 Euler-method discretization |
| Hybrid architecture | "Mix attention and SSM" | 任何交错 Transformer 和 SSM 层的模型；Jamba 是生产原型 |

## Further Reading

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — 原始 Jamba 论文，比率消融，256k 上下文声明
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — 扩展家族，398B/94B 和 12B/52B 公开发布
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba 构建的选择性 SSM 论文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — 简化的结构化状态空间继任者
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — 复数值状态，MIMO，2026 纯 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 论文，LLM 的 SSM 谱系起点
