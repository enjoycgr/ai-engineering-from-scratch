---
name: mc-evaluator
description: 通过蒙特卡洛 (Monte Carlo) 展开评估策略，并在可用时生成与 DP 对比的收敛报告。
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

给定一个环境（分段的，带有 reset+step API）和一个策略，输出：

1. 方法 (Method)。首次访问 vs 每次访问 MC。理由。
2. 片段预算 (Episode budget)。目标数量、方差诊断、预期标准误差。
3. 探索计划 (Exploration plan)。ε 调度（如需要）或探索性启动。
4. 黄金标准对比 (Gold-standard comparison)。如果是表格型则用 DP 最优 V*；否则用 Q-learning / PPO 基线的界限。
5. 终止检查 (Termination check)。最大步数上限、超时、非终止轨迹的处理。

拒绝在没有有限视界上限的非分段任务上运行 MC。拒绝报告表格型任务每个状态少于 100 个片段的 V^π 估计。将任何零方差动作的策略标记为探索风险。
