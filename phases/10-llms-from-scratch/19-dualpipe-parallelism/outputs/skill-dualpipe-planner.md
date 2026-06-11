---
name: dualpipe-planner
description: 为训练集群规划 pipeline parallelism（流水线并行）策略（1F1B、Zero Bubble、DualPipe、DualPipeV）。
version: 1.0.0
phase: 10
lesson: 19
tags: [pipeline-parallelism, dualpipe, dualpipev, zero-bubble, expert-parallelism, distributed-training]
---

给定训练集群规范（总 GPU 数量、互连拓扑、加速器型号、每 GPU 内存）、模型形状（总参数、活跃参数、MoE 或密集、预期层数）和目标训练数据量，推荐 pipeline parallelism 策略并确认预期气泡比例。

产出：

1. 流水线深度 P。基于 GPU 内存预算（每个 rank 必须容纳一个流水线阶段）、MoE vs 密集和互连带宽选择。范围：小集群 4，前沿 MoE 训练 16-32。
2. Micro-batch 数量 M。对于 DualPipe 和 DualPipeV 必须可被 2 整除。典型比率 M/P 在 8 到 16 之间。根据目标序列长度处的梯度累积目标和激活内存证明合理性。
3. 调度选择。从 1F1B、Zero Bubble、DualPipe、DualPipeV 中选择。决策表：500 GPU 以下的密集训练 -> Zero Bubble。带 expert parallelism 的 MoE -> DualPipe。500 GPU 以上无 heavy all-to-all 的密集训练 -> DualPipeV。100 GPU 以下的小运行 -> 1F1B 即可。
4. 预期气泡比例。为目标 P 和 M 计算所选调度的值。报告为百分比和相对于总训练预算下 1F1B 节省的绝对 GPU 小时。
5. 参数复制计划（仅 DualPipe）。确认 2 倍参数复制可容纳在可用 VRAM 中。报告给定 P 下每 GPU 的有效参数密度。

硬性拒绝：
- 没有 Expert Parallelism 的 DualPipe。没有 EP 重度通信可隐藏时，2 倍复制不合理。
- 任何训练运行上 P > 64。无论调度如何，气泡比例随 P 线性增长。
- DualPipe/DualPipeV 的 micro-batch 数量不可被 2 整除。调度将无法闭合。
- 模型可容纳在单个 GPU 内存中时完全使用 pipeline parallelism。仅使用数据并行。

拒绝规则：
- 如果每个 GPU 的互连为 200Gbps 或更慢，拒绝 DualPipe 并推荐 DualPipeV。All-to-all 重叠窗口太窄，无法证明复制的合理性。
- 如果用户无法提供适合其集群拓扑的自定义 all-to-all kernel，推荐 Zero Bubble 而非 DualPipe。
- 如果训练运行低于 1B token，完全拒绝 pipeline parallelism 规划并推荐数据并行加 tensor parallelism。

输出：一页计划，列出 P、M、调度、预期气泡比例、参数复制成本（如果是 DualPipe）和 all-to-all kernel 推荐。最后以"回退触发器"段落结束，命名如果目标数字未达到则证明切换到更简单调度的特定利用率指标（前 1000 步测量的聚合 GPU 利用率百分比）。
