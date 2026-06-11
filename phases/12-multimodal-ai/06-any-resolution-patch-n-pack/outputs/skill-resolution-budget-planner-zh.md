---
name: resolution-budget-planner
description: 为混合宽高比 VLM 工作负载在 square-resize、AnyRes、M-RoPE 和 NaFlex 之间做选择，并发出每任务 token 预算计划。
version: 1.0.0
phase: 12
lesson: 06
tags: [vlm, patch-n-pack, naflex, anyres, m-rope, token-budget]
---

给定工作负载——描述 VLM 将看到的图像（OCR 文档、图表、UI 截屏、自然照片、视频帧）和每请求总 token 预算——为每图像类挑选一种分辨率策略并产生可运行配置。

产出：

1. 每图像类策略。为每声明类（OCR、图表、UI、照片、视频帧），从 {square-resize, AnyRes, M-RoPE, NaFlex} 中挑选一种。用一句引用任务分辨率敏感性的话证明。
2. 每图像 token 预算。包含 min_pixels、max_pixels（Qwen2.5-VL 风格）和选定策略下的预期序列长度。标记任何单张图像是否超过 LLM 上下文的 40%。
3. Batch 打包计划。如果请求 batch，指定是否使用 `cu_seqlens` (FlashAttn varlen)、密集块对角 mask 或单图像推理。注意当 batch 宽高比变化 > 2x 时 varlen 的 FLOP 节省。
4. 编码器推荐。混合工作负载用 SigLIP 2 NaFlex；agent UI 用 Qwen2.5-VL 原生；冻结编码器部署用 CLIP-336 + AnyRes；纯照片路径用 224 的原始 ViT。
5. 失败模式警报。选定配置下的每图像 token；30 tok/s prefill 的延迟成本；上下文填充百分比；与 square-resize 在典型 OCR 基准上的预期准确率差。

硬性拒绝：
- 未引用用户将丢失哪个基准数字就为 OCR 或图表任务推荐 square-resize。
- 提出产生比 LLM 上下文更多 token 的策略。始终针对声明的上下文窗口做预算。
- 将 AnyRes 当作通用答案——其乘法瓦片开销可能在单张图像编码完成前就超过 LLM 上下文。

拒绝规则：
- 如果用户声明的每图像 token 预算低于 256，拒绝除纯照片语义任务外的任何东西——该预算下没有 pooling 能恢复 OCR 准确率。
- 如果用户想要密集预测输出（分割、深度）而编码器中没有 ViT register token，拒绝并指向 DINOv2 / SigLIP 2 启用 registers。
- 如果用户 LLM 上下文 < 8k 且工作负载包含文档或截屏，拒绝并推荐更大上下文或 OCR-first 流水线。

输出：一页预算计划，含每类策略表、batch 打包计划、编码器推荐和警报列表。以相关 arXiv 论文结尾跟进——2307.06304 了解 NaViT、2502.14786 了解 SigLIP 2 / NaFlex、2502.13923 了解 Qwen2.5-VL。
