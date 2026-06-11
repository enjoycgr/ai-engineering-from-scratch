---
name: prompt-tensor-debugger
description: 深度学习代码中 tensor shape error (张量形状错误) 的分步调试 prompt
phase: 1
lesson: 12
---

我的深度学习代码中有一个 tensor shape error (张量形状错误)。帮我修复它。

**Error message (错误信息):** [在此处粘贴错误]

**我的 tensor shapes (张量形状):**
- [name]: [shape]
- [name]: [shape]

**我尝试进行的运算:** [描述它]

---

调试时，请遵循以下精确流程：

**Step 1: 识别运算类型。**
什么运算产生了错误？映射到以下之一：
- Matrix multiply / Linear layer (矩阵乘法 / 线性层)（内部维度必须匹配）
- Broadcasting (广播)（从右侧对齐，每个维度必须相等或为 1）
- Concatenation (拼接)（所有维度匹配，除了 cat dimension）
- Convolution (卷积)（需要特定 rank 和 channel position）
- Reshape (重塑)（总元素数必须保留）

**Step 2: 写出 shape contract (形状契约)。**
对识别的运算，显式写出期望的 shape：
```
matmul(A, B): A 是 (..., m, k), B 是 (..., k, n) -> (..., m, n)
broadcast(A, B): 从右侧对齐，每对必须是 (相等) 或 (其中一个为 1)
cat([A, B], dim=d): 所有维度匹配，除了 dim d
Linear(in_f, out_f): 输入最后一个维度必须等于 in_f
Conv2d(in_c, out_c, k): 输入必须是 (B, in_c, H, W)
```

**Step 3: 找到 mismatch (不匹配)。**
将实际 shape 与契约对比。识别违反规则的精确维度。

**Step 4: 选择最小修复。**
从这个表中选择：

| Symptom (症状) | Fix (修复) |
|---|---|
| Missing batch dimension (缺失 batch 维度) | `.unsqueeze(0)` |
| Missing channel dimension (缺失 channel 维度) | `.unsqueeze(1)` |
| Extra size-1 dimension (多余的大小为 1 的维度) | `.squeeze(dim)` |
| Inner dims wrong for matmul (matmul 内部维度错误) | `.transpose(-1, -2)` 或检查 weight shape |
| Need NCHW from NHWC (需要 NCHW) | `.permute(0, 3, 1, 2)` |
| Need NHWC from NCHW (需要 NHWC) | `.permute(0, 2, 3, 1)` |
| Flatten spatial dims for linear (为线性层展平空间维度) | `.flatten(1)` 或 `.reshape(B, -1)` |
| Split heads: (B,T,D) to (B,H,T,D/H) (拆分 head) | `.reshape(B, T, H, D//H).transpose(1, 2)` |
| Merge heads: (B,H,T,D/H) to (B,T,D) (合并 head) | `.transpose(1, 2).reshape(B, T, H*(D//H))` |
| Non-contiguous tensor with .view() (非连续 tensor 调用 .view) | `.contiguous().view(...)` 或改用 `.reshape(...)` |

**Step 5: 验证修复。**
展示每一步后的 shape。确认总元素数在 reshape 中保留。确认运算的 shape contract 现在已满足。

**Step 6: 检查静默 bug (silent bugs)。**
即使 shape 匹配，也要验证：
- Broadcasting 沿预期的 axis 发生（而非意外发生）
- Reduction 在正确的 dimension 上求和
- Batch dimension (dim 0) 在整个 forward pass 中保留
- 当 dimension ordering (维度顺序) 重要时使用 Transpose + reshape（而非仅 reshape）

将响应格式化为：
```
OPERATION (运算): [失败的运算]
EXPECTED (期望): [shape contract]
ACTUAL (实际): [提供的 shape]
MISMATCH (不匹配): [哪个维度，原因]
FIX (修复): [精确代码]
RESULT (结果): [修复后的 shape]
```
