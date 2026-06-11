---
name: vit-configurator
description: 为新视觉任务选择 ViT 变体、patch size 和预训练来源。
version: 1.0.0
phase: 7
lesson: 9
tags: [transformers, vit, vision]
---

给定一个视觉任务（分类 / 分割 / 检测 / 检索）、图像分辨率、数据集大小（有标签 + 无标签）和部署目标，输出：

1. **Backbone（骨干网络）**。以下之一：DINOv2 ViT-L/14（检索/分类的默认选择）、SAM 3 encoder（分割）、SigLIP（视觉-语言）、ConvNeXt（延迟敏感场景）。一句话说明理由。
2. **Patch size**。224 分辨率标准分类用 16，DINOv2 用 14，高分辨率密集预测用 8。标注序列长度 `(H/P)^2 + 1` 和注意力成本 `O(N^2)`。
3. **Pretraining source（预训练来源）**。检查点名称。有标签数据少（<10k）：DINOv2 特征冻结 + linear probe（线性探测）。数据多（>100k）：微调最后几个块。说明理由。
4. **Training recipe（训练配方）**。优化器（AdamW）、学习率、数据增强（RandAug、MixUp、Random Erasing）、label smoothing（标签平滑，典型值 0.1）、EMA（指数移动平均）。
5. **Risk note（风险说明）**。数据量级风险（数据太少不适合完全微调）、分辨率不匹配（预训练 224 → 部署 1024 未做位置插值）、缺少 register token（可能影响 DINOv2 特征）。

拒绝在少于 1M 张图像上推荐从头训练 ViT —— CNN 基线会赢。拒绝推荐序列长度 > 4096 的 patch size，除非明确讨论 Flash Attention + 分层变体（Swin）。标记任何改变输入分辨率但未插值位置嵌入的部署。
