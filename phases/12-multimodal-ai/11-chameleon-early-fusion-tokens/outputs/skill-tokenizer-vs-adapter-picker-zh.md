---
name: tokenizer-vs-adapter-picker
description: 为 VLM 项目在 Chameleon 风格 early fusion（共享词汇 tokenizer）和 LLaVA 风格 late fusion（冻结 LLM 上的 adapter）之间做选择。
version: 1.0.0
phase: 12
lesson: 11
tags: [chameleon, early-fusion, vq-vae, late-fusion, adapter]
---

给定产品规格（仅理解或理解+生成）、目标图像质量（社交帖子/杂志/印刷/广播）和成本预算（训练 + 推理），推荐 Chameleon 系列或 LLaVA 系列并给出具体架构大纲。

产出：

1. 裁决。Early-fusion（Chameleon / Emu3 / AnyGPT）或 late-fusion（LLaVA / BLIP-2 / Qwen-VL）系列。
2. Tokenizer 选择（early-fusion 裁决）。VQ-VAE（Chameleon）、MAGVIT-v2、IBQ 或 SBER-MoVQGAN；引用预期重建上限 PSNR。
3. 训练稳定性计划。大规模 early-fusion 的 QK-Norm、dropout 放置、LayerNorm 排序。
4. 成本估算。训练 GPU 小时数和每图像推理延迟 vs late-fusion 替代方案。
5. 生成质量上限。用户可以预期的 PSNR / FID 范围；产品的质量标准是否可用离散 token 达到，还是需要连续（Transfusion 风格）生成。
6. 迁移路径。如果用户成长且 late-fusion 成为限制（需要图像输出），迁移是什么样子。

硬性拒绝：
- 为仅理解产品推荐 Chameleon 风格。Late-fusion 更简单、更便宜，且对纯理解的上限更高。
- 为生产图像生成提议 K<4096 的 VQ-VAE。码本太小，伪影可见。
- 声称 early-fusion 推理免费。VQ 解码器每生成图像增加 50-200ms，通常超过 LLM 输出时间。

拒绝规则：
- 如果用户想要前沿质量图像生成（FID < 15，印刷就绪），拒绝离散 token 并指向 Transfusion / Stable Diffusion 3 / MMDiT（课程 12.13）。
- 如果产品从不需要图像输出，拒绝 early-fusion — 复杂度不必要。
- 如果用户想插入现有 Llama / Qwen LLM 权重，拒绝 early-fusion — 它需要从头预训练新模型。

输出：一页计划，含裁决、tokenizer 选择、稳定性检查清单、成本估算、质量上限、迁移路径。结尾附 arXiv 2405.09818（Chameleon）和 2408.11039（Transfusion）供对比阅读。
