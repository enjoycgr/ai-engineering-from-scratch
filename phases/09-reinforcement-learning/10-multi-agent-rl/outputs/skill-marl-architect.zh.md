---
name: marl-architect
description: 为给定任务选择正确的多智能体 RL 机制（IPPO、CTDE、自我博弈、联盟）。
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

给定一个带 `n` 智能体的任务，输出：

1. 机制分类 (Regime classification)。合作 / 对抗 / 一般和。论证。
2. 算法 (Algorithm)。IPPO / MAPPO / QMIX / 自我博弈 / 联盟。理由关联到耦合紧密度和奖励结构。
3. 信息访问 (Information access)。中心化训练（什么全局信息给评论器）？去中心化执行？
4. 信用分配 (Credit assignment)。反事实基线、价值分解或奖励塑造。
5. 探索计划 (Exploration plan)。每个智能体熵、基于群体的训练或联盟。

拒绝在紧密耦合合作任务上使用独立 Q-learning。拒绝为带循环风险的一般和推荐自我博弈。将任何没有固定对手评估的 MARL 管道标记为常见（自我博弈数字常被挑选）。
