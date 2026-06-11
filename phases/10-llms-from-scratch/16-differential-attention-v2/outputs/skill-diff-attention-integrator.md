---
name: diff-attention-integrator
description: 将 Differential Attention V2 添加到新预训练运行或 LoRA 微调中的集成计划。
version: 1.0.0
phase: 10
lesson: 16
tags: [differential-attention, diff-transformer, long-context, flash-attention, pre-training, lora]
---

给定模型架构（hidden、heads、KV heads、layers、d_head）、目标上下文长度、幻觉或长上下文配置文件（现有评估上的失败模式）以及训练预算（可用 token、GPU 小时），生成 DIFF V2 的集成计划。

产出：

1. 集成模式。从头开始的预训练、训练中架构交换，或对 Q 投影的 LoRA 微调。根据训练预算和现有可用权重证明选择的合理性。
2. 架构差异。具体的逐字段变更列表：哪些投影增长、哪些保持不变、添加了哪些参数计数，以及减法在 attention 块中的放置位置。包括按层深度的 `lambda_init` 调度（论文默认：`0.8 - 0.6 * exp(-0.3 * (depth - 1))`；如果逐层遥测显示不稳定，则按深度调整）。
3. Kernel 选择。鉴于 V2 的 head 数量翻倍，确认 FlashAttention 2 或 3 支持。除非用户明确需要它用于可复现性，否则拒绝 V1 的自定义 kernel 路径。
4. 内存预算。KV cache 保持基线（KV head 不变）。计算每 token 激活内存增量（额外的 Q head、额外计算）。报告目标上下文下的绝对数字。
5. 训练稳定性计划。描述要监控的内容：每层的 `lambda` 漂移、每个 head 的 attention 熵、Q 投影上的梯度方差。命名如果遥测表明发散应触发回退到基线 attention 的特定指标。

硬性拒绝：
- 在没有持续预训练的情况下将 DIFF attention 添加到预训练模型。输出分布会漂移——不是即插即用的修复。
- 2026 年 4 月之后的任何新运行使用 DIFF V1。V2 在所有测量维度上都严格更好。
- 在未同时启用长上下文训练数据的情况下集成 DIFF。收益只在 32k 以上显现。
- 在没有对照实验的情况下将 `lambda_init` 更改为负值。负初始化会减去超过噪声基底的部分并导致训练崩溃。

拒绝规则：
- 如果目标上下文低于 16k，拒绝集成并推荐标准 attention。噪声基底论证无法证明添加的参数成本是合理的。
- 如果用户无法提供长上下文评估数据（RULER、needle-in-haystack、MultiNeedle），拒绝并首先请求校准数据。
- 如果用户使用的是 FlashAttention-2 之前的栈，拒绝并建议在尝试集成之前升级栈。

输出：一页集成计划，列出模式、参数计数增量、KV cache 影响、FlashAttention 确认、`lambda` 调度和 3 指标监控面板。最后以"成功标准"段落结束，命名证明在架构中保留 DIFF V2 而非回退的特定长上下文评估数字（RULER 64k 上的百分点增量或等效指标）。
