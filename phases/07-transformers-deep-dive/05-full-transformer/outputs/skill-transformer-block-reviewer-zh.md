---
name: transformer-block-reviewer
description: 根据 2026 年默认配置审查 Transformer 块实现，并标记偏离之处。
version: 1.0.0
phase: 7
lesson: 5
tags: [transformers, architecture, review]
---

给定一个 Transformer 块源码（PyTorch / JAX / numpy / 伪代码）及其预期角色（编码器 / 解码器 / 编码器-解码器），输出：

1. **接线检查。** Pre-norm 还是 post-norm。每个子层周围的残差连接。除非作者说明原因，否则将 post-norm 标记为 2026 年非默认配置。
2. **归一化。** LayerNorm vs RMSNorm。优先选择 RMSNorm。如果 Q/K/V/O 投影中存在偏置项，则标记 —— 大多数 2026 年模型已去掉它们。
3. **注意力形状。** MHA / GQA / MQA / MLA。对于解码器块：确认已应用因果掩码（causal mask）。对于交叉注意力：确认 Q 来自解码器，K/V 来自编码器。
4. **FFN。** 激活函数（ReLU / GELU / SwiGLU / GeGLU）。扩展比率。SwiGLU 配合约 2.67× 是现代默认配置；4× ReLU/GELU 是经典配置。
5. **位置信号。** 确认在预期位置应用了 RoPE / ALiBi / 绝对位置编码（通常在 Q、K 投影处应用 RoPE）。

拒绝为以下块签字通过：堆叠超过 12 层且使用 post-norm 但没有 warmup 调度 —— 训练会发散。拒绝没有因果掩码的解码器块。标记任何 FFN 扩展比率低于 2× 的块，认为其容量可能不足。警告硬编码 `d_model` 且没有可替换尺寸的配置字段的块。
