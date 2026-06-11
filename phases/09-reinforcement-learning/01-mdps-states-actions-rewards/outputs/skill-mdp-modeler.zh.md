---
name: mdp-modeler
description: 根据任务描述生成马尔可夫决策过程 (Markov Decision Process, MDP) 规范，并在训练前标记形式化风险。
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

给定一个任务（控制 / 游戏 / 推荐 / 大语言模型微调），输出：

1. 状态 (State)。精确的特征向量或张量规范。论证马尔可夫性质。
2. 动作 (Action)。离散集合或连续范围。维度。
3. 转移 (Transition)。确定性的、已知模型的随机性，或仅可采样。
4. 奖励 (Reward)。函数和来源。稀疏 (sparse) vs 塑造 (shaped)。终止 vs 每步。
5. 折扣因子 (Discount)。数值和视界论证。

如果状态是非马尔可夫的，且未明确提及帧堆叠 (frame-stacking) 或循环状态，则拒绝交付。如果奖励不是根据目标结果定义的，则拒绝。在无限视界任务上标记任何 `γ ≥ 1.0`。将任何奖励范围 >100 倍典型步进奖励的标记为可能的梯度爆炸源。
