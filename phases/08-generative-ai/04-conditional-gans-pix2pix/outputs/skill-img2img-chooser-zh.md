---
name: img2img-chooser
description: 根据配对/未配对数据、域特异性和延迟预算选择图像到图像方法。
version: 1.0.0
phase: 8
lesson: 04
tags: [pix2pix, img2img, conditional]
---

给定任务描述（源域、目标域、数据可用性 - 配对/未配对/N 个样本、延迟预算、质量门槛），输出：

1. 方法。Pix2Pix（配对、狭窄）、Pix2PixHD（配对、高分辨率）、CycleGAN（未配对）、SPADE（分割到图像），或基于 SD3 / Flux.1 的 ControlNet 变体（通用、开放域）。
2. 训练数据规格。最小配对数、分辨率、数据增强、许可证注意事项。
3. 架构。G（U-Net 深度、通道宽度）、D（PatchGAN 感受野、谱归一化）、损失权重（adv、L1、VGG-perceptual）。
4. 推理延迟。单张消费级 GPU（RTX 4090、M3 Max）上的目标 ms/图像，分辨率权衡。
5. 评估。对留出配对数据的 LPIPS、5k 样本的 FID、任务特定指标（分割任务的 mIoU、超分辨率的 PSNR）、人类偏好。

当数据未配对时拒绝推荐 Pix2Pix——改开 CycleGAN 或 ControlNet。配对数据少于 500 对时拒绝训练配对模型，除非给出增强/预训练建议。标记任何包含"任意文本提示"的请求——那些需要扩散 + ControlNet，而不是配对 GAN。
