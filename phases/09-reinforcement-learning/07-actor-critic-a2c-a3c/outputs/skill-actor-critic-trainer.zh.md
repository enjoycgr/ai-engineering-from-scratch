---
name: actor-critic-trainer
description: 为给定环境生成 A2C / A3C / GAE 配置，指定优势估计和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定一个环境和计算预算，输出：

1. 并行度 (Parallelism)。A2C（GPU 批量）vs A3C（CPU 异步）及工作者数量。
2. 展开长度 T (Rollout length T)。每次更新每个环境的步数。
3. 优势估计器 (Advantage estimator)。n-step 或 GAE(λ)；指定 λ。
4. 损失权重 (Loss weights)。`c_v`（价值）、`c_e`（熵）、梯度裁剪。
5. 学习率 (Learning rates)。演员和评论器（如使用则分开）。

拒绝在视界 > 1000 的环境上使用单工作者 A2C（太同策略、太慢）。没有优势归一化则拒绝交付。将任何 `c_e = 0` 且观察到的熵 < 0.1 的运行标记为熵崩溃。
