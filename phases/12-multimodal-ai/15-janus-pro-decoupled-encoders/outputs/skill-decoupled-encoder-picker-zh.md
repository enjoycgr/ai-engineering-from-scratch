---
name: decoupled-encoder-picker
description: 决定统一 VLM 是否应该解耦其视觉编码器，并在 Janus-Pro、JanusFlow 和 InternVL-U 之间做选择。
version: 1.0.0
phase: 12
lesson: 15
tags: [janus-pro, janusflow, internvl-u, decoupled-encoders, unified-model]
---

给定统一模型规格（理解+生成、可选编辑/inpainting）、计算预算和开放权重约束，推荐解耦编码器架构和具体配置。

产出：

1. 架构选择。Janus-Pro（VQ 生成）、JanusFlow（rectified flow 生成）、InternVL-U（原生预训练+解耦）。
2. 编码器组合。理解用 SigLIP-SO400m；离散生成用 MAGVIT-v2 / IBQ VQ；连续用 SD3 风格 VAE。
3. 数据阶段计划。Stage 1 alignment（5000-1 亿对），Stage 2 unified（7000 万+对），Stage 3 instruction（100 万+样本）。引用 Janus-Pro 的 5.4x 模型+2.8x 数据扩展结果。
4. 路由策略。基于 prompt-tag（显式 `<understand>` / `<generate>`）或基于任务分类器。
5. 共享主干初始化。从预训练 LLM（DeepSeek、Qwen、Llama）初始化，而非从头。
6. 质量上限。预期 MMMU（7B 下 ~60）和 GenEval（Janus-Pro 7B 下 ~0.80 / InternVL-U 下 ~0.85+）。

硬性拒绝：
- 当用户两端质量标准都是前沿竞争力时提议单编码器统一模型（Show-o / Transfusion）。解耦方法是唯一路径。
- 为 <10B 模型推荐从头预训练。复用预训练 LLM 主干。
- 为任何新项目提议 Janus（原版）而非 Janus-Pro。Janus-Pro 是后继者。

拒绝规则：
- 如果用户仅需要理解，拒绝解耦并推荐 LLaVA 系列。一个编码器足够。
- 如果用户仅需要生成，拒绝并推荐 Stable Diffusion 3 / Flux — 专家在 T2I 质量上仍赢。
- 如果计算 <5 万 GPU 小时，拒绝 InternVL-U（需要原生预训练）并推荐 Janus-Pro（复用预训练 LLM）。

输出：一页计划，含架构选择、编码器组合、阶段计划、路由、共享主干初始化和质量上限。结尾附 arXiv 2501.17811（Janus-Pro）、2411.07975（JanusFlow）、2603.09877（InternVL-U）。
