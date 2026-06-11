---
name: vae-trainer
description: 为给定的数据集和下游用途指定 VAE 架构、隐变量 (latent) 大小、beta 调度方案和评估计划。
version: 1.0.0
phase: 8
lesson: 02
tags: [vae, latent, generative]
---

给定数据集画像（模态、分辨率、数据集大小）和下游用途（仅重建、采样，或作为 latent-diffusion /token-AR 模型的输入编码器），输出：

1. 变体。Plain VAE、beta-VAE、VQ-VAE、RVQ（残差）或 NVAE。一句话理由，关联模态和下游用途。
2. 架构。编码器/解码器拓扑（卷积下采样因子、通道宽度、隐藏维度、注意力块）。适用时提及公开参考权重（`sd-vae-ft-ema`、Encodec、DAC、WAN-VAE）。
3. 隐变量维度 (Latent dim)。空间和通道维度。每样本总比特数。相对于原始数据的压缩比。
4. Beta 调度。Warmup 斜坡、最终值，以及如使用的 free-bits 阈值。
5. 评估计划。重建 MSE / SSIM / PSNR、每维 KL、活跃维度数、后验坍塌 (posterior collapse) 告警阈值、`q(z|x)` 与先验 (prior) 之间的 Fréchet 距离。

拒绝在训练开始时发送 beta > 0.5 的 VAE（会导致后验坍塌 (posterior collapse)）。拒绝将普通高斯 VAE 作为图像的最终生成器——它会模糊；应将其作为扩散 (diffusion) 或流匹配 (flow matching) 模型的隐变量编码器。标记任何码本使用率低于 20% 的 VQ-VAE 为码本重置策略配置错误。
