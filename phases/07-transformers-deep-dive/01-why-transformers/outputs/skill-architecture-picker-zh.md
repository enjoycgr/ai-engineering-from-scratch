---
name: sequence-architecture-picker
description: 根据序列长度、吞吐量和训练预算选择序列架构（RNN、transformer、SSM、hybrid）。
version: 1.0.0
phase: 7
lesson: 1
tags: [transformers, architecture, rnn, ssm]
---

给定一个序列问题（最大长度、batch shape、训练 token 预算、推理延迟目标、设备类别），输出：

1. 主要架构。以下之一：transformer、state-space model（SSM，Mamba/RWKV）、hybrid SSM+attention、RNN。一句理由，与主导约束相关。
2. 上下文长度策略。如果是 transformer：full attention（全注意力）截止长度、sliding window（滑动窗口）大小、RoPE（Rotary Positional Embedding，旋转位置编码）缩放因子。如果是 SSM：scan chunk size（扫描块大小）。如果是 RNN：hidden width（隐藏层宽度）。
3. 训练 FLOP 概况。来自架构 + 上下文的每 token 近似 FLOPs；注意规格是否符合计算预算。
4. 推理内存概况。Transformer 的 KV cache（KV 缓存）、SSM 的 state size（状态大小）、RNN 的 per-token memory（每 token 内存）。标记目标设备是否能容纳 batch size 为 1 的单个 batch。
5. 风险说明。该选择在规格规模上已知的一个具体失败模式（例如，在没有 Flash Attention 的 24GB GPU 上，transformer 在 64K 上下文时 OOM）。

对于超过 1B token 的训练运行，拒绝推荐纯 RNN，除非明确说明梯度流和并行性惩罚。对于 >64K 上下文，拒绝推荐 full-attention transformer，除非说明 `O(N^2)` 内存成本。对于生产环境，拒绝推荐全新架构（发表 <12 个月），除非有命名的 fallback（回退方案）。
