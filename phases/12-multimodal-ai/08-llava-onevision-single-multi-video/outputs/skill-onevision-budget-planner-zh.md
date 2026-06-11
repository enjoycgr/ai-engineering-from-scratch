---
name: onevision-budget-planner
description: 为目标产品混合在单图像、多图像和视频场景间分配 LLaVA-OneVision 风格统一视觉 token 预算。
version: 1.0.0
phase: 12
lesson: 08
tags: [llava-onevision, token-budget, curriculum, multi-image, video]
---

给定产品预期任务分布——单图像、多图像和视频请求的百分比——以及每样本视觉 token 预算，发出每场景分配计划和训练课程。

产出：

1. 每场景配置。单图像：AnyRes 瓦片数 + 缩略图 + pooling 因子；多图像：每样本图像数 + 每图像 pooling；视频：帧数 + 每帧 pooling。
2. Token 预算平衡。每场景总 token 应落在目标预算 ±30% 内；标记任何低于目标 70%（token 不足）或高于 130%（上下文风险）的场景。
3. 课程计划。三阶段（SI → OV → TT）含数据权重。TT 阶段使用用户产品混合。
4. 预期涌现技能。给定用户产品混合，预测哪些 LLaVA-OneVision 风格涌现能力可能出现（多摄像头、set-of-mark、截屏-agent 或产品特定变体）。
5. 训练数据估算。给定 7B 基础 LLM，引用 OneVision-1.5 数据规模，每阶段所需近似 token / 图像 / 帧计数。

硬性拒绝：
- 提出把视频或多图像放在单图像之前的阶段顺序。OneVision 表明这丢失 2-4 MMMU。
- 当产品 80% 单图像时把所有预算分配给视频。浪费，不是平衡。
- 假设 AnyRes-16 (4x4 网格) 适配 4k token 预算而不做激进 pooling。它不适配。

拒绝规则：
- 如果每样本 token 预算低于 1024，拒绝多图像或视频用例——低于该底线，场景崩溃。
- 如果用户想要 5+ 帧视频在全 729-token 分辨率，拒绝；推荐 3x pooling 或更少帧。
- 如果产品分布完全省略单图像，拒绝并推荐 Qwen2.5-VL 风格 M-RoPE——OneVision 课程假设单图像作为感知基础。

输出：一页计划，含每场景 token 配置、课程阶段权重、涌现技能预测和数据规模估计。以 arXiv 2408.03326 (OneVision) 和 arXiv 2509.23661 (OneVision-1.5 fully open) 的指针结尾。
