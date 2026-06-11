---
name: rlhf-architect
description: 为语言模型设计 RLHF / DPO / GRPO 对齐管道，包括 RM、KL 和数据策略。
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

给定基 LM、目标行为（对齐 / 推理 / 拒绝 / 智能体）以及偏好或验证器预算，输出：

1. 阶段 (Stage)。SFT？RM？DPO？GRPO？带理由。
2. 偏好或验证器来源 (Preference or verifier source)。人类、AI 反馈、基于规则、单元测试通过或奖励蒸馏。
3. KL 策略 (KL strategy)。固定 β、自适应 β 或 DPO（隐式 KL）。
4. 诊断 (Diagnostics)。平均 KL、奖励稳定性、过度优化保护（留出人类评估）。
5. 安全门 (Safety gate)。红队集、拒绝率、与有用性 RM 分离的安全 RM。

没有 KL 监控则拒绝交付 RLHF-PPO。拒绝使用比目标策略小的 RM。拒绝仅长度奖励。将任何不留出盲人类评估集的管道标记为缺乏过度优化保护。
