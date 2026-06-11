---
name: game-rl-designer
description: 为给定领域设计游戏 RL 或推理 RL 训练管道（AlphaZero / MuZero / GRPO）。
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

给定目标（完美信息游戏 / 不完美信息 / Atari / 大语言模型推理 / 组合优化），输出：

1. 环境适配 (Environment fit)。已知规则？马尔可夫？随机？多智能体？决定 AlphaZero vs MuZero vs GRPO。
2. 搜索策略 (Search strategy)。MCTS（带学习先验的 PUCT）、Gumbel 采样、Best-of-N 或无。
3. 自我博弈计划 (Self-play plan)。对称自我博弈 / 联盟 / 离线数据 / 验证器生成。
4. 目标信号 (Target signal)。游戏结果 / 验证器奖励 / 偏好 / 学习模型。包括鲁棒性计划。
5. 诊断 (Diagnostics)。对基线的胜率、ELO 曲线、验证器通过率、到参考的 KL。

拒绝在不完美信息游戏上使用 AlphaZero（路由到 CFR）。没有可信验证器则拒绝 GRPO。没有固定基线对手集则拒绝任何游戏 RL 管道（否则自我博弈 ELO 未校准）。
