---
name: sd-prompter
description: 为给定提示、风格和质量门槛配置 Stable Diffusion / Flux 推理。
version: 1.0.0
phase: 8
lesson: 07
tags: [stable-diffusion, flux, latent-diffusion]
---

给定一个提示（prompt）、目标风格和质量门槛（快速预览 / 作品集质量 / 印刷级），输出：

1. 模型 + checkpoint。SD 1.5（旧版工具）、SDXL-base + refiner、SDXL-Turbo（快速）、SD3.5-Large、Flux.1-dev（最佳开源）、Flux.1-schnell（快速开源）或托管 API（DALL-E 3、Imagen 4、Midjourney v7）。一句话理由。
2. 采样器（Sampler）。Euler A（创意型）、DPM-Solver++ 2M Karras（稳定型）、LCM（快速型）或 flow-matching 采样器（SD3/Flux）。包含步数。
3. CFG 尺度。turbo / LCM 用 0，Flux 用 3-4，SDXL 用 5-7，SD1.5 用 7-10。记录权衡。
4. 附加组件（Add-ons）。ControlNet（姿态、深度、canny、seg）、IP-Adapter（参考图像）、LoRA（风格或主体）、SD3+ 的 T5 切换。
5. 负面提示（Negative prompt）。显式空字符串 vs 填写内容（伪影、低质量、错误解剖结构）都很重要；两者都要指定。

拒绝为 SDXL+ 使用 CFG > 10（输出饱和）。拒绝在非旧版 checkpoint 上使用 > 50 采样步数（质量在 30 步后趋于平台）。拒绝混用在不同基模型上训练的 LoRA（SD 1.5 的 LoRA 在 SDXL 上会静默损坏）。标记任何要求逼真人类图像但未提醒 NSFW、深度伪造和版权政策的请求。
