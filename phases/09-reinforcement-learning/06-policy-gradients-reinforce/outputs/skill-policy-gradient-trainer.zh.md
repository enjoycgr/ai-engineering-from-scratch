---
name: policy-gradient-trainer
description: 为给定任务生成 REINFORCE / actor-critic / PPO 训练配置并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定一个环境（离散 / 连续动作、视界、奖励统计），输出：

1. 策略头 (Policy head)。Softmax（离散）或高斯（连续）及参数数量。
2. 基线 (Baseline)。无（普通）、运行均值、学习到的 `V̂(s)` 或 A2C 评论器。
3. 方差控制 (Variance controls)。默认开启回报到终点、回报归一化、梯度裁剪值。
4. 熵奖励 (Entropy bonus)。系数 β 和衰减调度。
5. 批量大小 (Batch size)。每次更新的片段数；同策略数据新鲜度约定。

拒绝在视界 > 500 步的问题上使用无基线 REINFORCE。拒绝用 softmax 头做连续动作控制。将任何 `β = 0` 且观察到的策略熵 < 0.1 的运行标记为熵崩溃。
