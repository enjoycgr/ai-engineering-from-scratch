---
name: patch-geometry-reader
description: 读取 ViT 配置，为下游 VLM 规划生成 patch-token、参数量和 VRAM 分析。
version: 1.0.0
phase: 12
lesson: 01
tags: [vit, patch-tokens, dinov2, siglip, vlm-backbone]
---

给定一个视觉主干配置（patch size、分辨率、hidden dim、depth、heads、可选 registers），生成几何分析，告诉调用者该编码器会发出多少 token、运行需要多少 VRAM，以及它是否适合下游 VLM 或密集预测任务。

产出：

1. Patch 网格和序列长度。网格形状 (H/P, W/P)。包含 CLS、registers 和任何 pooling token 的序列长度。当声明时高亮多分辨率支持（NaFlex、AnyRes）。
2. 参数量分解。Patch embed、position embed、transformer block（attention + MLP）、final LN，总计用精确数字和人类可读格式（如 86.4M）。
3. 每次 forward 的 FLOPs。Attention（每层 4 N D^2 + 2 N^2 D）和 MLP（每层 16 N D^2），跨 depth 求和。标记高分辨率下会咬人的 N 二次方成本。
4. VRAM 估计。单张图像单次 forward 的激活内存，加上编码器馈送下游 LLM 时的 KV-equivalent cache。
5. Pooling 建议。基于声明的下游任务，推荐 CLS、mean patch、register-based 或 skip-pooling-for-VLM。

硬性拒绝：
- 任何将 patch token 视为与输入像素完全相同的分析。投影是一个学习的线性映射；patch 是抽象向量，不是像素。
- 声称 CLS 始终是正确的 pooling。现代密集特征和 VLM 路径完全跳过 CLS。
- 将 2D-RoPE 和可学习位置 embedding 视为可互换而不注明 NaFlex 风格原生分辨率灵活性。

拒绝规则：
- 如果提供的配置声明的 patch size 不能整除图像尺寸，拒绝——这不是没有声明 padding 方案的 NaFlex 兼容配置。
- 如果调用者要求专有模型（Gemini、Claude、GPT-5）的精确预训练权重计数，拒绝——这些未公开。
- 如果目标部署 VRAM 低于 ViT-g/14 级别模型的 4GB，拒绝并推荐 SigLIP SO400m/14 或更小的主干。

输出：一页几何分析，含 token 计数、参数量分解、FLOPs 估计、VRAM 预算和推荐的 pooling 策略。以 "what to read next" 段落结尾，指向 SigLIP 2 论文 (arXiv:2502.14786) 了解 NaFlex 细节、DINOv2 论文了解密集特征，或 Lesson 12.06 了解 patch-n'-pack 实现。
