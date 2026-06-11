---
name: training-budget-estimator
description: 根据计算预算和部署约束，为新 transformer 训练运行估算 (N, D, hours, GPU count)。
version: 1.0.0
phase: 7
lesson: 13
tags: [scaling-laws, training, chinchilla]
---

给定训练目标（目标 loss / 目标 MMLU / 目标下游指标）、计算预算（美元或 FLOPs）、推理量（token/月）和约束（目标设备、内存、延迟），输出：

1. 计算 regime。Chinchilla-optimal（Chinchilla 最优）、over-trained（过度训练，推理优化）、under-trained（训练不足，原型）。一句话说明原因，与推理量挂钩。
2. N 和 D。具体数值。打印 `D/N` 比例。如果过度训练，注明与 Chinchilla-optimal 的 loss 惩罚。
3. 训练 wall-clock 时间。给定假设训练吞吐量（dense 模型 MFU ≈ 40%，MoE ~30%）下的 小时 × GPU 数量。预算精度（bf16 / fp8）和优化器（AdamW / Muon）。
4. 数据来源。命名语料库或合成数据预算。如果所需 `D` 超过可用高质量 token，标记警告。
5. 风险提示。一个具体的失败模式：数据污染（data contamination）、大规模优化器不稳定、上下文长度与 tokenizer 不匹配、评估套件饱和。

如果模型将服务于高推理量，拒绝在 Chinchilla-optimal 下训练 >8B 的 dense 模型——推理成本会复合增长。如果没有定义留出评估套件，拒绝设置目标 loss。标记任何将 >1% 预算花在架构搜索而非数据策划上的计划——已知回报很小。要求在承诺全部预算之前，用 1% 的预算进行规模化验证运行以验证假设。
