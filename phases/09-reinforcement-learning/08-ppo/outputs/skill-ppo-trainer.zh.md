---
name: ppo-trainer
description: 为给定环境生成 PPO 训练配置和诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定一个环境和训练预算，输出：

1. 展开大小 (Rollout size)。`N` 环境 × `T` 步。
2. 更新调度 (Update schedule)。`K` 轮次、小批量大小、LR 调度。
3. 替代目标参数 (Surrogate params)。`ε`（裁剪）、`c_v`、`c_e`、开启优势归一化。
4. 优势 (Advantage)。显式 `γ` 和 `λ` 的 GAE(`λ`)。
5. 诊断计划 (Diagnostics plan)。KL、裁剪比例、解释方差阈值及警报。

拒绝 `K > 30` 或 `ε > 0.3`（不安全的信任域）。没有优势归一化或 KL/裁剪监控则拒绝任何 PPO 运行。将持续高于 0.4 的裁剪比例标记为漂移。
