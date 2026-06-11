# DualPipe Parallelism（双流水线并行）

> DeepSeek-V3 在 2,048 张 H800 GPU 上训练，MoE expert 分散在节点间。跨节点 expert all-to-all 通信成本为每 1 GPU 小时计算对应 1 GPU 小时通信。GPU 一半时间空闲。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，将前向和后向计算与它们触发的 all-to-all 通信重叠。气泡减少，吞吐量攀升，而保留两份模型参数拷贝（名称中"dual"的来源）一旦 Expert Parallelism 已经将 expert 分散到各 rank，成本就很低。本课是 Learn 类型的走读，解释 DualPipe 实际做什么，以及为什么 Sea AI Lab 的 DualPipeV 改进以边际更紧的气泡为代价去掉了 2 倍参数成本。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## Learning Objectives

- 命名 DualPipe 前向-后向块的四个组件，以及为什么每个组件获得自己的重叠窗口。
- 解释大规模下的 pipeline bubble（流水线气泡）问题，以及"无气泡"在实践与营销中的含义。
- 手工追踪 8 个 PP rank 和 16 个 micro-batch 的 DualPipe 调度，确认前向和反向流填充彼此的闲置时隙。
- 说明 DualPipeV（Sea AI Lab，2025）的权衡：在 Expert Parallelism 不活跃时，以略大的气泡为代价去掉了 2 倍参数复制。

## The Problem

在 2k 张 H800 GPU 上训练 671B MoE 模型遇到三个复合瓶颈：

1. **内存压力。** 每张 GPU 持有模型的一个切片。序列 8k 跨 61 层 128 个 head 的 activation 内存是巨大的。
2. **Pipeline bubbles。** 传统 pipeline parallelism（GPipe, 1F1B）在 GPU 等待阶段输入或梯度时让它们空闲。在 8 个阶段，即使使用 1F1B 调度，大约 12% 的 GPU 时间可能是气泡。
3. **跨节点 all-to-all。** 带 expert parallelism 的 MoE 将 expert 分散到节点间。每个前向传递触发一个 all-to-all 将 token 分派到它们的 expert，另一个 all-to-all 合并。在 2k GPU 上这很容易变成 1:1 的计算-通信比率。

每个都有单独的解决方案：gradient checkpointing（梯度检查点）用于内存，Zero Bubble（Sea AI Lab，2023）用于 pipeline bubble，expert-parallel comm kernel 用于 all-to-all。DualPipe 做的是让它们协同工作。调度在单个前向-后向块内重叠计算和通信，从流水线两端同时注入 micro-batch，并使用生成的调度将 all-to-all 隐藏在计算窗口内。

报告结果：pipeline bubble 近乎消除，DeepSeek-V3 14.8T token 训练运行中平均 GPU 利用率超过 95%。

## The Concept

### Pipeline parallelism 复习

将 N 层模型跨 P 个设备拆分。设备 `i` 持有层 `i * N/P .. (i+1) * N/P - 1`。一个 micro-batch 前向流过设备 0 到 P-1，然后从 P-1 到 0 反向。每个设备只有在前一个设备发送其输出后才能开始其前向阶段，只有在下游设备发送上游梯度后才能开始反向。

GPipe（Huang 等人，2019）一次调度一个 micro-batch，浪费大部分 GPU 时间。1F1B（Narayanan 等人，2021）交错多个 micro-batch 的前向和反向传递。Zero Bubble（Qi 等人，2023）将反向传递分成两部分——backward-for-input (B) 和 backward-for-weights (W)——并调度它们填充气泡。Zero Bubble 之后，流水线几乎紧凑。

DualPipe 是下一步。它在之上添加两个想法：

### 想法 1：块分解

每个前向块分成四个组件：

- **Attention。** Q/K/V 投影、attention、输出投影。
- **All-to-all dispatch。** 跨节点通信，将 token 发送到它们的 expert。
- **MLP。** MoE expert 计算。
- **All-to-all combine。** 跨节点通信，将 expert 输出带回。

反向块添加每个的梯度版本。DualPipe 调度它们，使 all-to-all dispatch 与下一个块的 attention 计算并行发生，all-to-all combine 与后续块的 MLP 计算并行发生。

### 想法 2：双向调度

大多数流水线调度从阶段 0 注入 micro-batch 并流向阶段 P-1。DualPipe 从两端注入 micro-batch。阶段 0 看到起源于那里的前向 micro-batch；阶段 P-1 也看到起源于那里的前向 micro-batch。两个流在中间相遇。

为此，设备 `i` 必须同时持有早期流水线层 `i` AND 晚期流水线层 `P - 1 - i`。这就是 DualPipe 的"dual"部分：每个设备保留两份它需要服务的模型层拷贝（每个方向一份）。在 DeepSeek-V3 的规模下，这是 2 倍参数复制成本。它是可承受的，因为 Expert Parallelism 已经将 MoE expert 分散得很薄，复制非 expert 层两次只是小意思。

关键的是，一个方向的前向流和另一个方向的反向流恰好在单向调度中气泡所在的位置精确重叠。气泡消失。

### 手工追踪的调度

考虑 P = 4 个 rank，8 个 micro-batch，分为 4 前向 / 4 反向。时间从左到右移动；行是设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

阅读 "F4/F5R" 符号：rank 1 在同一时隙运行 micro-batch 4 的前向（在流水线中从左到右）AND micro-batch 5 的前向（从右到左）。这就是"双向"在操作上的含义。

在 rank 2 交叉流更早重叠，在 rank 0 和 P-1 重叠最晚。在调度的稳定中间阶段，每个 rank 运行 X 方向的前向与 Y 方向的反向重叠。计算繁忙。前向传递的 all-to-all dispatch 隐藏在反向计算内。All-to-all combine 隐藏在前向计算内。气泡被挤出。

### Bubble 核算

标准 1F1B pipeline bubble（每个 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进降低了它但未降到零。DualPipe 在稳定阶段，如果 micro-batch 数量可被 2 倍流水线深度整除，则气泡为零。稳定阶段之外（warmup 和 cooldown），有一些气泡但它不随 micro-batch 数量增长——论文强调的关键特性。

营销术语："无气泡"。技术术语：气泡不随 micro-batch 数量增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）显示，只有当 Expert Parallelism 不是瓶颈时才有完全零气泡；在 EP 驱动的 all-to-all 情况下，一些调度妥协总是存在。

### DualPipeV — 改进版

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2 倍参数复制是浪费的。他们的 DualPipeV 调度将双向注入折叠成在单份参数拷贝上运行的"V 形"调度。气泡略大于 DualPipe 的，但内存节省是巨大的。DeepSeek 在其开源 DualPipe 实现中采用 DualPipeV 作为 EP-off 模式。

权衡：

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Param copies per device | 2 | 1 | 1 | 1 |
| Bubble vs micro-batches | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| Use when | EP-heavy MoE | dense or EP-light | baseline | any pipeline |

### 对 14.8T token 运行的意义

DeepSeek-V3 的预训练在 2,048 张 H800 GPU 上消耗 14.8T token，约 2.8M GPU 小时。使用朴素 1F1B，他们会因 pipeline bubble 损失 12-15%——340-420K GPU 小时，足够训练一个完整的 70B 模型。DualPipe 回收了大部分。没有内部日志很难直接量化贡献，但论文中的声明是训练期间平均 GPU 利用率超过 95%。

对于较小的运行（1k GPU 以下），DualPipe 是过度设计——pipeline bubble 相对于总成本较小，密集模型训练很少遇到 all-to-all 瓶颈。对于数千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 在栈中的位置

- 与 **FSDP**（Phase 10 · 05）互补。FSDP 跨 rank 分片模型参数；DualPipe 跨 rank 调度计算。它们结合使用。
- 兼容 **ZeRO-3** 梯度分片。两份拷贝复制的簿记需要与 ZeRO 的分片梯度协作。
- 需要为特定集群拓扑调优的**自定义 all-to-all kernel**。DeepSeek 的开源 kernel 是参考实现。

## Use It

`code/main.py` 是一个流水线调度模拟器。它接受 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 每个的稳定阶段利用率。它是教学工具——数字匹配论文的定性声明，不是关于生产测量加速的声明。

模拟器的价值：用不同的 P 和 micro-batch 数量运行它，观察 1F1B 的气泡分数如何增长而 DualPipe 的不增长。

真实训练运行的集成考虑：

- 选择一个能干净整除 micro-batch 数量的 pipeline-parallel 深度。
- 确保 expert-parallel mesh 支持双向 all-to-all。DeepSeek 的 kernel 是参考。
- 预期第一次要在调度本身上烧掉一周的调试时间。簿记很繁琐。
- 监控每个 rank 的 GPU 利用率，不只是聚合。DualPipe 的收益来自收紧落后者。

## Ship It

本课产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规范（GPU 数量、拓扑、互连、模型形状），它推荐 pipeline parallelism 策略、要使用的调度算法，以及目标规模下的预期气泡分数。

## Exercises

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异并将其表示为每百万训练 token 回收的 GPU 小时。

2. 手工绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用 micro-batch ID 和方向标记每个时隙。识别气泡消失的第一个时隙。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的 Figure 5。识别 DualPipe 前向块内 all-to-all dispatch 的重叠窗口。解释计算调度如何隐藏它。

4. 计算 DualPipe 对 70B 密集模型（P=8 个流水线阶段）和 671B MoE 模型（P=16 个流水线阶段）的 2 倍参数开销。展示为什么 MoE 情况的开销比例更小（大多数参数是 expert，分散在大型 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年的竞争双向调度器）比较。识别 DualPipe 添加而 Chimera 没有的两个具体特性，使用论文的 Section 3.4 作为参考。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Pipeline bubble | "Idle time per rank" | GPU 周期浪费，因为流水线阶段正在等待其输入或梯度 |
| 1F1B | "Default pipeline schedule" | 一个前向 / 一个反向交错调度；DualPipe 击败的基线 |
| Zero Bubble | "Sea AI Lab 2023" | 将反向分成 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | "DeepSeek-V3 schedule" | 双向流水线 + 计算-通信重叠；气泡不随 micro-batch 数量增长 |
| DualPipeV | "Cut-in-half" | V 形改进，以略大气泡为代价去掉 2 倍参数复制 |
| Chunk | "Unit of pipeline work" | 一个 micro-batch 通过一个流水线阶段的前向或反向传递 |
| All-to-all dispatch | "Send tokens to experts" | 将 token 路由到其分配的 MoE expert 的跨节点通信 |
| All-to-all combine | "Bring expert outputs back" | 在 MLP 后收集 expert 输出的跨节点通信 |
| Expert Parallelism (EP) | "Experts across GPUs" | 跨 rank 分片 MoE expert，使不同 GPU 持有不同 expert |
| Pipeline Parallelism (PP) | "Layers across GPUs" | 跨 rank 分片模型层；DualPipe 调度的维度 |
| Bubble fraction | "Wasted GPU time" | (bubble_time / total_time)；DualPipe 驱向零的分数 |

## Further Reading

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — 主要 DualPipe 参考
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — 开源参考实现，包括 DualPipeV (Cut-in-half) 模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble 前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — 影响 DeepSeek EP-off 模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe 比较的 1F1B 调度
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — 原始 pipeline parallelism 论文和 bubble 问题
