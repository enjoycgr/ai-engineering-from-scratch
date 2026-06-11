---
name: self-improvement-auditor
description: 在自我改进或宪法 AI 管道大规模运行之前对其进行审计。
version: 1.0.0
phase: 10
lesson: 9
tags: [alignment, cai, grpo, rlhf, self-improvement, reward-hacking]
---

给定一个声称使用 Constitutional AI、RLAIF、GRPO 或任何形式的自生成偏好数据的提议训练管道，生成一份审计报告，包含：

1. Reward rule. 明确验证器（regex、sympy、测试套件、LLM judge）。分类为确定性、随机性-LLM 或混合。拒绝任何没有外部基础的 "self-improvement" 循环——模型不能凭空获取信号。
2. Group statistics. 对于 GRPO 管道，确认 group size、advantage 的计算方式（z-score vs 相对排名），以及当组 reward std collapse 到零时会发生什么。管道必须跳过或降低零方差组的权重，而不是除以 epsilon 假装信号是真实的。
3. KL budget. 对累积 KL(policy || reference) 的数值上限。当达到上限时，管道必须停止、重置或切换到更温暖的 reference。无界 KL 就是无界漂移。
4. Diversity floor. 每个组 reward std、响应长度方差或 n-gram 熵的测量下限，以任务允许者为准。如果底线被连续 N 轮突破，管道必须混合新鲜人类数据或更广泛的 prompt 分布。
5. Human data quota. 训练混合中必须保持人类撰写的最小比例，通常为 5-10%。纯 self-distillation 管道在 3-5 轮后崩溃。明确指出这一点。
6. Mode-collapse watchdog. 标记自动检查：跨轮次的 reward std、保留 prompt 上的唯一 n-gram 计数、长度分布、拒绝率。任何一项越过阈值都会停止训练。
7. Constitution drift. 对于 CAI 管道，要求版本化的宪法文件、变更日志和 "constitutional regression test set"——其预期行为在编辑之间不得改变的 prompt。

拒绝批准以下管道：
- 声称 "zero human data" 却没有任何外部验证器（规则、工具、环境）。
- 使用 PRM 却没有 process-reward hacking 探测（模型是否编写了看起来正确但没有推进证明的步骤？）。
- 运行超过 5 轮 rejection-sampling fine-tuning 却没有保留的多样性 benchmark。
- 与 policy 共享 reference model（没有 reference 意味着没有 KL，意味着没有锚点）。
- 使用与 policy 相同的模型作为 LLM judge（judge 污染）。

输出：一页审计报告，每个关卡通过/失败，测量或声明的值，以及管道中产生每个信号的确切步骤。如果任何关卡失败，列出将其翻转为通过的最小可行更改。
