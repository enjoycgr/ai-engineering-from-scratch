---
name: token-gen-cost-analyzer
description: 为 Emu3 风格的 next-token 生成计算 token 计数、推理延迟和质量上限，并在 Emu3 系列和 diffusion 之间做选择。
version: 1.0.0
phase: 12
lesson: 12
tags: [emu3, next-token-prediction, video-gen, diffusion, cfg]
---

给定生成产品规格（图像或视频、目标分辨率、质量等级、吞吐量要求），计算 Emu3 风格 next-token 生成的 token 数，估算推理成本，并在 Emu3 系列和 diffusion 之间做选择。

产出：

1. Token 计数。所选 tokenizer 缩减下的每图像 token（图像通常每维 8x）。3D VQ 下的每视频 token（通常 4x4x4 时空）。
2. 推理延迟。Emu3 系列的 token / 吞吐量（每秒 token 数）；diffusion 的去噪步数 * 每步时间。引用具体 A100 / H100 范围。
3. 质量上限。Tokenizer 重建 PSNR（IBQ 类 30-32 dB）、MJHQ-30K 上 FID 预期、视频 FVD。
4. CFG 配置。每任务推荐 guidance weight（gamma）；标准生成典型 3.0，强提示遵循 5-7。
5. 选择。如果产品需要统一理解+生成或任意模态灵活性则选 Emu3 系列；如果产品是仅图像生成且延迟严格则选 diffusion（SDXL / SD3 / Flux）。

硬性拒绝：
- 声称 Emu3 推理比 diffusion 快。并非如此；在数千图像 token 上的自回归解码是固定成本。
- 未指定 CFG weight 就推荐 Emu3 系列。没有它质量会崩塌。
- 为严格 4K 图像生成提议 Emu3。2048+ 分辨率下的 token 数会炸掉 KV cache 并耗时数分钟。

拒绝规则：
- 如果延迟预算 <5s 每图像，拒绝 Emu3 并推荐 SDXL 或 SD3。
- 如果产品必须生成图像 AND 描述它们 AND 推理第三方图像，推荐 Emu3 系列（统一 loss 是关键）；diffusion 无法做到这点，除非另加 VLM。
- 如果用户想要开放权重且商业使用许可宽松，拒绝 Emu3 — 先检查其许可证；某些版本仅限研究。

输出：一页分析，含 token 计数、延迟估算、质量上限、CFG 配置和选择及理由。结尾附 arXiv 2409.18869（Emu3）和 2408.11039（Transfusion）供替代方案参考。
