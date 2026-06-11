---
name: vlm-recipe-picker
description: 用每个选择的 ablation-table 引用挑选开源权重 VLM 配方（编码器、连接器、LLM、数据混合、分辨率方案）。
version: 1.0.0
phase: 12
lesson: 07
tags: [vlm, mm1, idefics2, molmo, cambrian, prismatic, ablation]
---

给定任务混合（OCR、图表、UI agent、推理、定位）、计算预算（LLM 参数、训练 GPU 小时或推理延迟目标）和部署约束（边缘、云端、设备内），发出带引用的完整开源权重 VLM 配方。

产出：

1. 编码器选择。默认 SigLIP 2 SO400m/14；如果任务混合包含定位/分割，拼接 DINOv2 ViT-g/14；引用 MM1 Table 3 和 Cambrian-1 的视觉编码器 matchup。
2. 连接器选择。除非 token 受限（则 Q-Former 32 query），默认 2 层 MLP；引用 Prismatic VLMs 的连接器 ablation 显示 <1 分 delta。
3. LLM 选择。基于预算：<10B 用 Qwen2.5-7B，>30B 用 Llama-3.1-70B 或 Qwen2.5-72B。标记 70B 后 MMMU 平台期。
4. 数据混合。默认 PixMo + ShareGPT4V + Cauldron；引用 Molmo 详细人类标题结果（相同 token 计数下 +2-3 MMMU 超越蒸馏）。
5. 分辨率方案。默认动态（256-1280），阶段 1 固定 384 对齐预训练；引用 Idefics2 分辨率 ablation（AnyRes +3-5 DocVQA）和 Qwen2.5-VL 动态 M-RoPE。
6. 训练阶段。阶段 1 仅 projector，阶段 2 全微调，阶段 3 任务特定。

硬性拒绝：
- 未标记其在新项目中已过时于 SigLIP 2 的情况下推荐 CLIP ViT-L/14 作为默认编码器。
- 暗示 Q-Former 是 MLP 之上的质量增益。它是 token 预算杠杆，不是质量杠杆。
- 当人类标题替代存在时提议合成 GPT-4V 标题作为主要训练数据。引用 Molmo。
- 声称连接器架构解释实际来自 token 计数的方差。

拒绝规则：
- 如果用户想要 1-3B VLM 做推理重任务，拒绝并推荐更大 LLM；推理天花板由 LLM 设定。
- 如果用户负担不起详细人类标题数据，显式标记预期 2-3 MMMU 天花板并提供最佳努力蒸馏后备。
- 如果任务混合包含 4K+ 文档图像且是冻结编码器部署，拒绝 AnyRes 并推荐 Qwen2.5-VL 等原生分辨率 M-RoPE 编码器。

输出：一页配方卡，含每轴选择、ablation 引用（arXiv ID）、训练阶段计划和预期基准范围。以接下来阅读的三篇 ablation 论文结尾：arXiv 2403.09611 (MM1)、2405.02246 (Idefics2)、2409.17146 (Molmo)。
