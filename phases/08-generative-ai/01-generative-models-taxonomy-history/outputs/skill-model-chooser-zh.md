---
name: generative-model-chooser
description: 为给定任务和预算选择一个生成式模型家族、主干模型和托管替代方案。
version: 1.0.0
phase: 8
lesson: 01
tags: [generative, taxonomy]
---

给定一个任务描述（模态、领域、延迟预算、计算预算、条件信号），输出：

1. Family（家族）。显式-可计算（Explicit-tractable）、显式-近似（VAE / diffusion）、隐式（GAN）、score / flow matching，或 token-AR。一句理由，关联模态 + 延迟。
2. Backbone + open reference（主干 + 开源参考）。一个用户可以立即微调的开源预训练模型（如 Stable Diffusion 3、Flux.1-dev、AudioCraft 2、StyleGAN3、3D Gaussian Splatting）。
3. Hosted alternatives（托管替代方案）。三个生产级 API，按质量 / 成本 / 延迟权衡排序（fal.ai、Replicate、Stability、Runway、Veo、Kling、ElevenLabs 等）。
4. Failure mode（失效模式）。所选家族的已知病理（mode collapse、exposure bias、sampler drift、tokenizer artifacts、CLIP-score gaming）。
5. Budget（预算）。单张 A100 上的大致训练小时数、每次采样的推理成本、VRAM 下限。

当任务需要似然评分时，拒绝推荐 GAN。当高分辨率实时使用时，拒绝推荐基于像素的自回归。如果列出的开源主干已经覆盖该领域，标记任何"从头训练"的建议。
