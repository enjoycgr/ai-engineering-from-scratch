---
name: editing-pipeline
description: 从 source + edit description 到 ready-to-ship output 规划图像编辑管线。
version: 1.0.0
phase: 8
lesson: 09
tags: [inpaint, outpaint, edit, sam]
---

给定 source image、target edit（remove X、replace Y with Z、extend canvas、restyle region、change season / time-of-day）和 quality bar（draft / portfolio / print），输出：

1. Mask strategy。Explicit brush mask、SAM 2 click / box prompt、Grounded-SAM on a text phrase、或 RMBG（用于背景移除）。一句话理由。
2. Base model + mode。SD-Inpaint / SDXL-Inpaint / Flux-Fill / Flux-Kontext 用于 instruction edits，或 SDEdit noise-level (0.3 / 0.6 / 0.9) 如果没有 mask。
3. Prompt scaffolding。编辑后描述整张图像，不只是新内容。包含 negative prompt。
4. CFG + strength + feather。Mask feather 8-16 px；SDXL-inpaint 的 CFG ~5-7，Flux 的 3-4。Strength 0.8-1.0 用于 full regenerate，0.3-0.5 用于 preserve。
5. Guardrails。NSFW / deepfake / trademark detection hook、face-swap policy gate、reversibility（保存 mask + seed）。

拒绝在没有明确 policy check 的情况下对可识别的公众人物进行身份编辑。拒绝在原始画布 anchor 少于 30% 的情况下对图像进行 outpainting（上下文太少会让模型 hallucinate）。标记任何 t/T > 0.7 且 fidelity target 为 "preserve subject" 的 SDEdit run 作为可能的 mismatch。
