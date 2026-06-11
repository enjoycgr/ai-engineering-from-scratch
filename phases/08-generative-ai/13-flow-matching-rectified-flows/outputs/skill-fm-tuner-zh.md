---
name: fm-tuner
description: 将 diffusion 训练计划转换为 flow-matching / rectified-flow 配置。
version: 1.0.0
phase: 8
lesson: 13
tags: [flow-matching, rectified-flow, diffusion]
---

给定一份 diffusion-style 训练计划（数据、算力、调度、目标步数、质量门槛），输出 flow-matching 等价方案：

1. Schedule + interpolant. 线性（rectified flow）、optimal transport (最优传输, Lipman OT-CFM)、variance-preserving (方差保持) 或 cosine (余弦)。一句话说明理由。
2. Time sampling. Uniform (均匀)、logit-normal (SD3) 或 mode-weighted (模态加权)。当 1000 Hz 的均匀采样浪费端点容量时发出警告。
3. Target. Velocity $v = x_1 - x_0$（rectified flow）或 $\alpha'(t)x_1 + \sigma'(t)x_0$（CFM）。明确说明是哪一种。
4. Optimizer + lr warmup. 包含 AdamW，其中 beta2 = 0.95 以保证 transformer 规模下的稳定性。
5. Reflow (重流) plan. 是否运行 0、1 或 2 次 reflow 迭代；每次迭代预算约等于对精选子集的完整重新推理。
6. Step counts. 训练步数目标、预期 inference (推理) 步数（20、4、2、1）、guidance scale 范围。
7. Eval. FID / CLIP-score 与 diffusion 基线对比，绘制 quality vs step count 曲线。

在 $v_1$ 收敛之前拒绝进行 reflow（在劣质模型上做 reflow 只会固化错误方向）。未经过 consistency distillation (一致性蒸馏) 的情况下拒绝推荐 1-step inference。标记任何目标 inference > 20 步的 flow-matching 模型——如果你需要那么多步，说明你浪费了这个重新建模。
