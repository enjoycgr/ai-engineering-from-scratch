---
name: attention-shapes
description: 调试 attention 实现中的形状错误。
phase: 5
lesson: 10
---

给定一个有缺陷的 attention 实现，你识别出形状不匹配。输出：

1. 哪个矩阵形状错误。命名该张量。
2. 它的形状应该是什么，从 `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)` 推导。
3. 一行修复。转置、reshape 或投影。
4. 捕获回归的测试。通常断言 `output.shape == (batch, T_dec, d_h)` 和 `weights.shape == (batch, T_dec, T_enc)` 以及 `weights.sum(dim=-1)` 接近 1。

拒绝推荐静默广播的修复。广播隐藏的 bug 稍后表现为静默的准确率下降。

对于 Bahdanau 混淆，坚持 decoder 输入是 `s_{t-1}`（pre-step state）。对于 Luong，是 `s_t`（post-step state）。Dot-product attention 中最常见的首次错误是 query/key 维度不匹配 —— 明确标记它。
