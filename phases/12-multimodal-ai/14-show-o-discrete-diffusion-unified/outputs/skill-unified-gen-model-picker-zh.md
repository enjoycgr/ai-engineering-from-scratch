---
name: unified-gen-model-picker
description: 为需要统一多模态理解和生成且带开放权重的 Show-o / Transfusion / Emu3 / Janus-Pro 系列之间做选择。
version: 1.0.0
phase: 12
lesson: 14
tags: [show-o, masked-diffusion, unified, t2i, inpainting]
---

给定需要统一理解+生成（VQA、captioning、T2I、可选 inpainting）且带开放权重约束和延迟预算的产品，选择模型系列并输出参考配置。

产出：

1. 系列裁决。Show-o（masked 离散 diffusion）、Transfusion / MMDiT（连续 diffusion）、Emu3 / Chameleon（自回归离散）、或 Janus-Pro（解耦编码器）。
2. 推理步预算。Show-o 16 步，Transfusion 20 步，Emu3 1024+ 步。用用户延迟预算论证选择。
3. Inpainting 支持。Show-o 免费；Transfusion 添加 mask channel；Emu3 需要单独微调。向用户标记这点。
4. Tokenizer 选择。离散系列推荐 IBQ / MAGVIT-v2 / SBER；连续系列推荐 SD3 的 VAE。
5. 训练稳定性。双 loss（Transfusion）需要 weight 调优；Show-o 单 loss 更干净。
6. 用户成长时的迁移路径。从 Show-o 到 Transfusion，当质量成为限制。

硬性拒绝：
- 当推理延迟 <10s 每图像时提议 Emu3 / Chameleon。在 ~1024 token 上的自回归太慢。
- 声称 Show-o 在前沿图像质量上匹配 Transfusion。并非如此。Tokenizer 是上限。
- 为需要 VQA 的产品推荐 Stable Diffusion。SD 无法推理图像。

拒绝规则：
- 如果用户想要 <2s 每图像生成，拒绝 Show-o 并推荐 Stable Diffusion + 单独 VLM 用于理解。接受多模型复杂度。
- 如果用户想要开放权重下的"最佳质量"，拒绝 Show-o / Emu3 并推荐 Transfusion 系列（MMDiT）或 JanusFlow。
- 如果用户无法承诺 tokenizer（担心许可、质量上限），拒绝仅离散系列并推荐 Transfusion。

输出：一页选择，含系列裁决、步预算、inpainting 支持、tokenizer 推荐、稳定性计划和迁移路径。结尾附 arXiv 2408.12528（Show-o）、2408.11039（Transfusion）、2501.17811（Janus-Pro）。
