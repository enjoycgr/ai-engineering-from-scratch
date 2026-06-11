---
name: mtp-planner
description: 为新预训练运行规划 multi-token prediction（多 token 预测）集成。
version: 1.0.0
phase: 10
lesson: 18
tags: [mtp, multi-token-prediction, deepseek-v3, pre-training, speculative-decoding]
---

给定预训练运行规范（模型规模、隐藏大小、层数、数据 token 预算、GPU 拓扑、部署目标）和既定目标（更密集的训练信号 vs speculative-decoding draft vs 两者都要），生成 MTP 集成计划。

产出：

1. 深度 D。选择 1 或 2。DeepSeek-V3 使用 D=1 并报告第一深度 speculative-decoding 接受率 80%+。D=2 对大多数运行来说是边际收益递减区。根据计算预算证明选择的合理性——每增加一个深度大致每训练步骤增加一个 transformer block 的计算。
2. Lambda 调度。默认：前 10% 训练使用 0.3，之后 0.1。对于小模型（7B 以下）早期信号更密集，可调高至 0.5；如果观察到 MTP 损失主导主损失，则调低。
3. 参数预算。报告每个模块相对于主模型的参数计数。确认开销低于主参数的 5%（密集）或 3%（MoE）。
4. 内存和计算开销。量化每步额外前向传递 FLOP（大致 `D * transformer_block_cost`）、额外后向传递内存（D 个模块的激活内存）和额外峰值 VRAM（共享 embedding 和 head 不计，投影和 transformer block 计）。
5. 推理时接线。描述如何在推理时将 MTP 模块用作 speculative-decoding draft。命名 Leviathan 规则集成路径和 KV-rollback 簿记。确认与目标推理栈（vLLM、SGLang、TensorRT-LLM）的兼容性。

硬性拒绝：
- 将 MTP 添加到没有 MTP 预训练的密集模型。无法 retrofit——MTP 模块未训练。
- 首次集成 D > 2。超过 D=1 的收益很小；复杂性快速增长。
- 在活跃参数低于 1B 的模型上使用 MTP。该规模下信号弱于开销成本。
- 当目标是 speculative decoding 时使用并行（Gloeckle 风格）head。它们不因果链式连接。

拒绝规则：
- 如果预训练数据以短序列（2k 以下）为主，拒绝。MTP 收益假设序列足够长，深度-2 监督才重要。
- 如果目标推理栈根本不支持 speculative decoding，注意 MTP 仍然购买更密集的训练信号并继续，但标记不匹配。
- 如果用户在没有 MTP 的现有密集检查点上继续预训练，拒绝并推荐仅在干净训练运行开始时或干净数据边界重置时添加 MTP。

输出：一页集成计划，列出 D、lambda 调度、参数开销（绝对值和百分比）、计算开销（每训练步骤百分比）和推理时 speculative-decoding 接线计划。最后以"成功标准"段落结束，命名证明保留 MTP 的测量指标：50B 训练 token 后深度 1 的接受率必须高于 70%，否则应回退架构。
