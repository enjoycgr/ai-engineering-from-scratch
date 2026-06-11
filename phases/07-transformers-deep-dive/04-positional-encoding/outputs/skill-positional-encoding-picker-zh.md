---
name: positional-encoding-picker
description: 根据上下文长度和训练预算选择位置编码（RoPE、ALiBi、sinusoidal）+ 缩放策略。
version: 1.0.0
phase: 7
lesson: 4
tags: [transformers, positional-encoding, rope, alibi]
---

给定一个 transformer 规格（推理时目标上下文长度、训练时上下文长度、外推需求、微调预算 token 数），输出：

1. 基础编码 (Base encoding)。选项：RoPE、ALiBi、sinusoidal、learned-absolute。附一句理由。
2. 超参数 (Hyperparameters)。如果是 RoPE：`base` 值、`d_head` 需要为偶数以便拆分。如果是 ALiBi：斜率公式。如果是 sinusoidal：`max_len`。
3. 扩展策略 (Extension strategy)。如果目标长度 > 训练长度：NTK-aware 缩放因子、YaRN 配置、LongRoPE 规格，或位置插值 (position-interpolation) 比例。说明微调 token 预算。
4. 测试计划 (Test plan)。在最大上下文长度下的 NIAH (needle-in-a-haystack, 大海捞针) 通过率目标，以及困惑度 (perplexity) 与训练长度基线的差距上限 X。
5. 回退方案 (Fallback)。如果长上下文评估失败：使用更大的 `base` 重新训练、切换到 ALiBi，或限制部署上下文长度。

拒绝为 2026 年的新模型推荐 sinusoidal 或 learned-absolute —— 它们无法外推，且每个现代技术栈都默认使用 RoPE 或 ALiBi。拒绝在没有微调阶段的情况下将 RoPE 缩放超过训练长度的 8 倍。拒绝在未对完整部署长度运行 NIAH 的情况下发布长上下文配置。
