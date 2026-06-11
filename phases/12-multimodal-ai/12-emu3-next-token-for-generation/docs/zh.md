# Emu3：图像和视频生成的 Next-Token Prediction

> BAAI 的 Emu3 (Wang 等人, 2024 年 9 月) 是本该结束 diffusion 与 autoregressive 辩论的 2024 年结果。单个 Llama 风格 decoder-only transformer，仅在 next-token-prediction 目标上训练，跨越文本 + VQ 图像 token + 3D VQ 视频 token 的统一词汇，在图像生成上击败 SDXL，在感知上击败 LLaVA-1.6。无 CLIP loss。无 diffusion schedule。Classifier-free guidance 在推理时用于质量，但核心训练目标是 teacher forcing 的 next-token prediction。发表于 Nature。本课阅读 Emu3 论点——为什么更好的 tokenizer 加规模就是全部所需——并与 diffusion 方法对比。

**类型：** Learn
**语言：** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**前置知识：** Phase 12 · 11 (Chameleon)
**时间：** ~120 分钟

## 学习目标

- 解释为什么 Emu3 的单一 loss next-token 目标有效，尽管长期假设 diffusion 对图像质量是必需的。
- 描述 3D video tokenizer：时空 VQ codebook 长什么样，为什么 patch 跨越时间。
- 对比 Emu3 与 Stable Diffusion XL（训练计算、推理成本、质量天花板）。
- 说出同一 Emu3 模型扮演的三个角色：Emu3-Gen（图像生成）、Emu3-Chat（感知）、Emu3-Stage2（视频生成）。

## 问题

到 2024 年的传统智慧：图像生成需要 diffusion。论据：离散图像 token 丢失太多信息以重建细节，且 autoregressive 采样在数千 token 上累积误差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 都使用某种 diffusion。Chameleon（Lesson 12.11）在小规模上部分反驳这一点，但未在质量上匹配 SDXL。

Emu3 正面攻击这个论据。声明：更好的视觉 tokenizer + 足够规模 + next-token loss = 在同样做感知的模型中击败 diffusion 的图像生成。

发表时有争议。两年后，开源统一生成家族（Emu3、Show-o、Janus-Pro、Transfusion）是研究的默认路径；生产前沿模型似乎使用某种变体。

## 概念

### Emu3 tokenizer

关键成分是视觉 tokenizer。Emu3 训练自定义 IBQ-class tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN 家族），每 token 8x8 分辨率缩减。512x512 图像变成 64x64 = 4096 token，codebook 大小 32768。

这比 Chameleon 每 512x512 的 1024 token 在 K=8192 时更大，但每 token 更便宜（更小的 codebook 查找、更简单的 codec）。关键指标：重建 PSNR 在 30.5 dB，与 Stable Diffusion 连续 latent space 的 32 dB 竞争。

对视频：3D VQ tokenizer 编码时空 patch（4x4x4 像素）为一个整数。8 FPS 的 4 秒片段有 32 帧；256x256 带 4x 空间和 4x 时间缩减，token 数是 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 token。

Tokenizer 质量是天花板。Emu3 的贡献部分是"我们训练了一个非常好的 tokenizer。

### 单一 loss 训练

Emu3 使用一个目标：跨文本 token、2D 图像 token 和 3D 视频 token 的共享词汇上的 next-token prediction。权重在训练期间按模态特定因子相乘以平衡贡献，但 loss 函数相同。

在以下混合上训练：
- 图像生成：`<text caption> <image> image_tokens </image>`
- 图像感知：`<image> image_tokens </image> <question> text_tokens`
- 视频生成：`<text caption> <video> video_tokens </video>`
- 视频感知：类似。
- 纯文本：标准 NTP。

模型从数据分布学习何时发出图像 token vs 文本 token。生成来自模型在 `<image>` 标签后预测图像 token。

### Classifier-free guidance 和 temperature

Autoregressive 图像生成用 classifier-free guidance (CFG) 在推理时大幅提升。Emu3 使用它：生成两次，一次带完整 caption，一次带空 caption，用 guidance weight（典型 3.0-7.0）混合 logits。这是 diffusion 使用的相同 CFG 技巧，借到 autoregressive 设置。

Temperature 重要：太高，伪影；太低，mode collapse。Emu3 推荐 perception 用 temperature 1.0，图像生成用 0.8。

### 三个角色，一个模型

Emu3 作为三个功能不同的 API 发货，但底层是一个权重集：

- Emu3-Gen。图像生成。输入文本，输出图像 token。
- Emu3-Chat。VQA 和标题生成。输入图像（token），输出文本。
- Emu3-Stage2。视频生成和视频 VQA。输入文本或视频，输出文本或视频。

无任务特定 head。只是不同 prompt 模板。相同 checkpoint。

### 基准

来自 Emu3 论文（2024 年 9 月）：

- 图像生成：在 MJHQ-30K FID 上击败 SDXL（5.4 vs 5.6），GenEval 总体（0.54 vs 0.55——统计平局），Deep-Eval 综合大致持平。
- 图像感知：在 VQAv2 上击败 LLaVA-1.6（75.1 vs 72.4），MMMU 大致匹配。
- 视频生成：4 秒片段质量与 Sora 时代公开基准模型的 FVD 竞争。

数字并非总是获胜——Emu3 在这里交易一分那里交易一分——但"next-token prediction 就是全部所需"的声明跨模态是可辩护的。

### 计算成本

Emu3 用 7B 参数模型在约 3000 亿多模态 token 上训练。GPU 小时大致与 Llama-2-7B 预训练相当（A100 级硅上 2k-4k GPU 年）。Stable Diffusion 3 等 diffusion 模型在类似预算下训练，但需要单独的文本编码器和更复杂的流水线。

推理时，Emu3 比 SDXL 每图像更慢：4096 图像 token 在 30 tok/s 下约 2 分钟每 512x512 图像，vs SDXL 的 2-5 秒。推测解码和 KV cache 优化缩小差距但不关闭它。Autoregressive 图像生成计算重；这是持续的交易。

### 为什么重要

Emu3 的深层贡献是概念性的。如果 next-token prediction 在图像生成上能扩展到匹配 diffusion，统一模型路径（一个 loss、一个主干、任何模态）是可行的。未来模型不需要单独的文本编码器、单独的 diffusion scheduler、单独的 VAE。一个 transformer、每模态一个 tokenizer、规模。

Show-o、Janus-Pro 和 InternVL-U 都建立在这个论点上或挑战它。中国实验室（BAAI、DeepSeek）比美国实验室在 2025 年更激进地朝这个方向发表。

## 使用它

`code/main.py` 构建两个玩具片段：

- 2D vs 3D VQ tokenizer 计数计算器：给定（分辨率、patch、clip_length、FPS），计算图像 vs 视频的 token 数。
- 带 classifier-free guidance 在 temperature 下的 autoregressive 图像 token 采样器。

CFG 实现匹配 Emu3 的配方——混合条件和无条件 logits 带 guidance weight。

## 交付它

本课产生 `outputs/skill-token-gen-cost-analyzer.md`。给定生成产品规格（图像或视频、目标分辨率、质量级别、延迟预算），它计算 token 数、推理成本，并在 Emu3 家族 vs diffusion 之间挑选。

## 练习

1. Emu3 在 8x8 缩减下每 512x512 图像产生 4096 token。计算 1024x1024 和 2048x2048 的等效值。推理延迟发生什么？

2. 阅读 Emu3 Section 3.3 关于 video tokenizer。描述 3D VQ patch 形状及为什么它是 4x4x4 而非 8x8x1。

3. Classifier-free guidance weight 5.0 vs 3.0：什么视觉效果？在 `code/main.py` 中追踪数学。

4. 计算 Emu3-7B 在 300B token 上的训练 FLOPs 并与 Stable Diffusion 3 对比。哪个训练更贵？

5. Emu3 在 FID 上击败 SDXL 但在 VQAv2 vs 专用 VLM 上未击败。解释为什么统一 loss 方法在不同基准上 vs 专家显示不同优势。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Next-token prediction | "NTP" | 标准自回归 loss：给定 token[0..i] 预测 token[i+1]；tokenized 时每模态都工作 |
| IBQ tokenizer | "Inverse bottleneck quantizer" | 一类 VQ-VAE，更大 codebook（32768+）且比 Chameleon 重建更好 |
| 3D VQ | "Spatiotemporal quantizer" | 由 (时间, 行, 列) 索引的 codebook；一个 token 覆盖 4x4x4 像素立方 |
| Classifier-free guidance | "CFG" | 用 gamma 混合条件和无条件 logits；推理时提升图像质量 |
| Unified vocabulary | "Shared tokens" | 文本 + 图像 + 视频都从相同整数空间抽取；模型预测接下来哪个模态 |
| MJHQ-30K | "Image gen benchmark" | 30k prompt 的 Midjourney 质量基准；Emu3 在此报告 FID |

## 延伸阅读

- [Wang 等人 — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun 等人 — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu 等人 — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu 等人 — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian 等人 — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
