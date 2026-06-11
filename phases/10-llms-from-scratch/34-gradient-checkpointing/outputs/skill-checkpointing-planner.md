---
name: checkpointing-planner
description: 给定训练配置和 HBM 预算，为每层选择激活重计算策略（none / selective / full / offload）。
version: 1.0.0
phase: 10
lesson: 34
tags: [gradient-checkpointing, activation-recomputation, selective-checkpoint, fsdp-offload, training-memory]
---

给定训练配置（层数 L、隐藏大小 d、序列长度 S、microbatch B、dtype 每值字节数、attention kernel、tensor-parallel 度 TP、pipeline-parallel 度 PP、如果是 MoE 则 expert-parallel 度 EP）以及权重和优化器状态后的每 rank HBM 预算，输出：

1. 每层策略。为栈中的每个层家族（embedding、attention、FFN、MoE expert、norm、output head）选择 none、selective、full 或 offload。S 超过 4_096 时 attention 默认 selective；残差流和 norm 默认 none；FFN 上仅当该层激活的测量 PCIe 传输时间小于其测量重计算时间时默认 offload。
2. 段大小 k。如果开启 full checkpointing，均匀层成本时 k 选 round(sqrt(L))，激活内存主导预算时选更小的 k。报告额外 FLOP 百分比为前向 FLOP 的 (1/k)。
3. FlashAttention 交互。确认 attention kernel 是否已经重计算 softmax。如果是，选择性 attention checkpointing 收益很小；降级为 none。按名称说明 kernel（FlashAttention-2/3、xFormers memory-efficient、vanilla）。
4. TP / PP 计划。对于 TP，命名重计算时需要 gather 或 rescatter 的激活以及每步增加的通信字节。对于 PP，确认哪些流水线阶段端到端检查点，使反向 microbatch 在流回之前释放激活内存。
5. 预算数学。预测策略前后的激活内存（每 rank MB）。预测 FLOP 开销为 fwd+bwd 的百分比。拒绝任何在 10% headroom 内无法容纳 HBM 预算的计划。

当仅 attention 上的 selective 就能关闭预算时，拒绝每层的 full checkpointing；分析显示 FLOP 开销比 selective 高很多倍却获得相同的内存节省，且精确比率是工作负载特定的。当目标 PCIe 链路上该层的测量激活传输时间超过其测量重计算时间时，拒绝 offload；重计算获胜。当所选框架不快照 amax 历史时，拒绝 FP8 训练的"到处检查点"；重计算会使尺度漂移并静默损坏梯度。

示例输入："L=64, d=8192, S=8192, B=1, bf16, FlashAttention-3, TP=8, PP=4, HBM budget per rank 32 GB after weights, MoE with 8 experts and EP=8."

示例输出：
- 每层策略：attention selective、FFN none、MoE expert full、embedding none、output head offload。
- 段大小：full 仅在 MoE 上应用，k=8；expert 路径 FLOP 开销 12%，其他地方 0。
- FlashAttention 交互：FA-3 已经重计算 softmax；selective 在层包装器上，不在 kernel 内部。
- TP / PP 计划：TP gather 重计算时的 attention 输入，每步额外通信 0.3 GB；PP 阶段各自检查点其完整前向；PP stage 3 保留其激活用于最终后向。
- 预算数学：无策略时激活 38 GB，有策略时 11 GB。总 FLOP 开销 7.5% fwd+bwd。
