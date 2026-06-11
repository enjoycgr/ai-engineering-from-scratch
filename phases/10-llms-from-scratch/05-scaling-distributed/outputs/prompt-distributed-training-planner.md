---
name: prompt-distributed-training-planner
description: 给定模型大小和可用硬件，规划分布式训练运行
version: 1.0.0
phase: 10
lesson: 5
tags: [distributed-training, fsdp, deepspeed, tensor-parallelism, pipeline-parallelism, scaling]
---

# Distributed Training Planner（分布式训练规划器）

为大型语言模型规划分布式训练运行时，使用此框架确定并行策略、内存预算、通信开销和预期吞吐量。

## Input Requirements（输入要求）

提供：
- **Model size**（参数数量，以十亿计）
- **Target training tokens**（以万亿计）
- **Available GPUs**（类型：A100/H100/H200，数量，互联：NVLink/InfiniBand）
- **GPU memory**（A100/H100 为 80GB，H200 为 141GB）
- **Nodes**（每节点 GPU 数，节点数）
- **Budget constraints**（最大成本（美元），最大 wall-clock 时间）

## Step 1: Memory Budget（内存预算）

计算每个组件的每卡内存：

| Component | Formula | FP16 | FP32 |
|-----------|---------|------|------|
| Weights | params x bytes_per_param | params x 2 | params x 4 |
| Adam optimizer (m + v) | params x 4 x 2 | 8 bytes/param always | 8 bytes/param |
| Gradients | params x bytes_per_param | params x 2 | params x 4 |
| Activations (estimate) | seq_len x batch x hidden x layers x 2 | varies | varies |

如果总量超过 GPU 内存，需要分片。按顺序尝试：
1. ZeRO-1（仅分片 optimizer）——通信最便宜
2. ZeRO-2（+ gradients）——中等通信
3. FSDP/ZeRO-3（+ weights）——通信最高但内存节省最大
4. 如果 activations 仍然太大，添加 activation checkpointing
5. 如果单层放不进一块 GPU，添加 tensor parallelism

## Step 2: Parallelism Strategy（并行策略）

### Decision Tree（决策树）

1. **一层能放进一块 GPU 吗？**
   - 不能：需要 tensor parallelism。设置 TP = 2, 4 或 8（在节点内）。
   - 能：跳过 tensor parallelism。

2. **完整模型（带分片）能放进一个节点内的 GPU 吗？**
   - 不能：需要 pipeline parallelism。设置 PP = 节点数 / 组数。
   - 能：跳过 pipeline parallelism。

3. **剩余多少 GPU 用于 data parallelism？**
   - DP = total_gpus / (TP x PP)

4. **Data parallel 组内的分片级别？**
   - 从 FSDP (ZeRO-3) 开始。如果通信是瓶颈，降到 ZeRO-2 或 ZeRO-1。

### Typical Configurations（典型配置）

| Model Size | Total GPUs | TP | PP | DP | Sharding |
|-----------|-----------|----|----|-----|----------|
| 7B | 8 | 1 | 1 | 8 | FSDP |
| 13B | 16 | 2 | 1 | 8 | FSDP |
| 70B | 64 | 8 | 1 | 8 | FSDP |
| 70B | 128 | 8 | 2 | 8 | FSDP |
| 405B | 16,384 | 8 | 16 | 128 | FSDP |

## Step 3: Communication Analysis（通信分析）

估算每训练步的通信量：

- **Data parallel (all-reduce)**：每步 2 x gradient_size x (N-1)/N
- **FSDP (all-gather + reduce-scatter)**：每步约 3 x weight_size x (N-1)/N（高于 DP）
- **Tensor parallel (all-reduce per layer)**：每步 2 x activation_size x num_layers（需要 NVLink）
- **Pipeline parallel (point-to-point)**：每 stage 边界 activation_size（最小）

如果通信时间超过计算时间的 20%，策略受通信限制。解决方案：
- Gradient accumulation（降低 all-reduce 频率）
- 通信与计算重叠（FSDP 默认这样做）
- 增加 micro-batch size（更好的计算-通信比）
- 切换到通信量更少的分片阶段

## Step 4: Throughput and Cost Estimate（吞吐量和成本估算）

**每训练步 FLOPS：**
- Forward：~2 x params x tokens_per_batch
- Backward：~4 x params x tokens_per_batch（forward 的 2 倍）
- 总计：~6 x params x tokens_per_batch

**训练时间：**
- total_flops = 6 x params x total_tokens
- time_seconds = total_flops / (num_gpus x gpu_tflops x 1e12 x utilization)
- 典型利用率：35-45%（考虑通信、pipeline bubbles、内存开销）

**成本：**
- total_gpu_hours = num_gpus x time_seconds / 3600
- cost = total_gpu_hours x cost_per_gpu_hour

## Step 5: Validation Checklist（验证检查清单）

启动前：

1. 每卡内存适合硬件限制（留 10% 余量）
2. Effective batch size 匹配目标（per_gpu_batch x DP x gradient_accumulation_steps）
3. 通信-计算比低于 20%
4. Pipeline bubble fraction 低于 15%（足够的 micro-batches）
5. 学习率按 effective batch size 缩放
6. Checkpoint 频率考虑故障概率（大型运行每 1-2 小时保存一次）
7. 设置了 gradient clipping（大型模型通常为 1.0）
8. Warmup steps 与总 steps 成比例（通常为总 steps 的 0.1-1%）

## Red Flags（危险信号）

- **TP > 8**：跨节点的 tensor parallelism（通过 InfiniBand）几乎总是比 pipeline parallelism 慢
- **Pipeline stages > 32**：即使有大量 micro-batches，bubble 开销也变得显著
- **Effective batch size > 10M tokens**：收益递减；可能损害收敛
- **Utilization 低于 30%**：受通信限制——重新评估并行策略
- **13B 以上没有 activation checkpointing**：Backward pass 期间会内存不足
- **没有 gradient accumulation 的每卡小 batch**：Gradient noise 增加；累积到 effective batch 256+ 样本
