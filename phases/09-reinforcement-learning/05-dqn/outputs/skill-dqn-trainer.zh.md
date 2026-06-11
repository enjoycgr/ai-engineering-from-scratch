---
name: dqn-trainer
description: 为离散动作强化学习任务生成 DQN 训练配置（缓冲区、目标同步、ε 调度、奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定一个离散动作环境（观察形状、动作数量、视界、奖励尺度），输出：

1. 网络 (Network)。架构（MLP / CNN / Transformer）、特征维度、深度。
2. 回放缓冲区 (Replay buffer)。容量、小批量大小、预热大小。
3. 目标网络 (Target network)。同步策略（每 C 步硬同步或软 τ）。
4. 探索 (Exploration)。ε 起始 / 结束 / 调度长度。
5. 损失 (Loss)。Huber vs MSE、梯度裁剪值、奖励裁剪规则。
6. 双 DQN (Double DQN)。默认开启，除非有明确理由禁用。

拒绝交付没有目标网络、没有回放缓冲区或 ε 保持为 1 的 DQN。拒绝连续动作任务（路由到 SAC / TD3）。将任何奖励范围 > 10 倍每步均值标记为需要裁剪或尺度归一化。
