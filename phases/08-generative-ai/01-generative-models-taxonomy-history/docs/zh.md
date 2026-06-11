# 生成式模型 —— 分类法与历史

> 每一个图像模型、文本模型、视频模型和 3D 模型都可以归入五个桶之一。选错桶，你会跟数学缠斗数周；选对桶，过去十二年的领域进展就会整齐地堆叠在你脑海中。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 分钟

## The Problem（问题）

生成式模型（generative model）只做一件事：给定从某个未知分布 `p_data(x)` 中抽取的训练样本，输出看起来像是从同一分布中采样的新样本。人脸、句子、MIDI 文件、蛋白质结构——如果你眯起眼睛看，它们都是同一个问题。

麻烦在于 `p_data` 生活在一个拥有数百万维度的空间里（一张 512×512 的 RGB 图像约 786k 维），样本坐落在这个空间内的一条薄流形（manifold）上，而你可能只有大约 1000 万个样本。暴力估计密度毫无希望。每个生成式模型都是一种妥协，把一个难题换成一个稍微不那么难的难题。

过去十二年有五大家族存活了下来。知道每个家族做出了哪种妥协，你就能明白为什么它在某些任务上获胜，而在另一些任务上崩溃。

## The Concept（概念）

![生成式模型的五大家族——按建模对象分类](../assets/taxonomy.svg)

**1. 显式密度，可计算（Explicit density, tractable）。** 把 `log p(x)` 写成一个你确实可以求值的和。自回归模型（autoregressive models，如 PixelCNN、WaveNet、GPT）将 `p(x) = ∏ p(x_i | x_<i)` 分解。标准化流（normalizing flows，如 RealNVP、Glow）把 `p(x)` 构建为一个简单基分布的可逆变换。优点：精确似然（exact likelihood），训练损失简洁。缺点：自回归 inference（推理）是顺序的（长序列慢），流需要可逆架构（架构上受限）。

**2. 显式密度，近似（Explicit density, approximate）。** 从下界（ELBO）约束 `log p(x)` 并优化该界。VAE（Kingma 2013）使用带变分后验（variational posterior）的编码器-解码器。扩散模型（diffusion models，DDPM，Ho 2020）训练一个去噪器（denoiser），它隐式地优化一个加权 ELBO。扩散（diffusion）是 2026 年图像、视频和 3D 的主干架构。

**3. 隐式密度（Implicit density）。** 完全跳过密度；学习一个生成器 `G(z)` 来产出样本，以及一个判别器 `D(x)` 来区分真假。GAN（Goodfellow 2014）。推理速度快（一次前向传播），但训练时 notoriously unstable（ notoriously 不稳定）。StyleGAN 1/2/3 在固定域真实感（人脸、卧室）上仍是 SOTA，即使在 2026 年。

**4. 基于分数 / 连续时间（Score-based / continuous-time）。** 直接学习对数密度的梯度 `∇_x log p(x)`（即 score）。Song & Ermon（2019）证明 score matching 将扩散推广到了 SDE。流匹配（flow matching，Lipman 2023）是 2024–2026 年的热门：无模拟训练（simulate-free training），更直的路径，采样速度比 DDPM 快 4–10 倍。Stable Diffusion 3、Flux、AudioCraft 2 都使用流匹配。

**5. 基于离散码元的自回归（Token-based autoregressive over discrete codes）。** 用 VQ-VAE 或残差量化器（residual quantizer）将高维数据压缩成一段短的离散 token 序列，然后用 Transformer 建模该 token 序列。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都用这个。这是桶 1 加上一个学习到的 tokenizer。

## A brief history（简史）

| Year | Model | Why it mattered |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | First deep generative model with a usable training loss. |
| 2014 | GAN (Goodfellow) | Implicit density, no likelihood — shockingly sharp samples. |
| 2015 | DRAW, PixelCNN | Sequential image generation. |
| 2017 | Glow, RealNVP | Invertible flows; exact likelihood with depth. |
| 2017 | Progressive GAN | First megapixel faces. |
| 2019 | StyleGAN / StyleGAN2 | Photorealistic faces still hard to beat for that one domain. |
| 2020 | DDPM (Ho) | Diffusion becomes practical. |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image goes mainstream. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = commodity. |
| 2022 | ControlNet, LoRA | Fine control over pretrained diffusion. |
| 2023 | SDXL, Midjourney v5, Flow matching | Scale + better training dynamics. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion; flow matching wins. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Production-grade video. |
| 2026 | Consistency + Rectified Flow | One-step sampling from diffusion backbones. |

## The five-question triage（五个问题的分诊）

当一篇新的生成式模型论文出现时，在阅读方法部分之前先回答这五个问题。

1. **What is being modeled?** 像素、隐变量、离散 token、3D 高斯、网格、波形？
2. **Is the density explicit or implicit?** 他们是否写下了 `log p(x)`？
3. **Sampling: one-shot or iterative?** 迭代意味着推理更慢；one-shot 通常意味着对抗或蒸馏。
4. **Conditioning: unconditional, class, text, image, pose?** 这决定了损失和架构脚手架。
5. **Evaluation: FID, CLIP score, IS, human preference, task accuracy?** 每个都有已知的失效模式（见 Lesson 14）。

你会在这个阶段的每一课中重新回答这五个问题。到最后，它们会成为你的条件反射。

## Build It（动手实现）

本课的代码是一个轻量级可视化：用三种玩具方法（核密度估计、离散直方图，以及一个最近样本的"GAN 风格"生成器）从样本中拟合一个 1-D 高斯混合模型，让你能在一个屏幕就能打印的问题上看到显式密度与隐式密度的区别。

运行 `code/main.py`。它从一个双峰高斯混合中抽取 2000 个样本，然后打印：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前两种方法让你可以问"这个点有多大概率？"第三种不行。这就是显式与隐式的区别，它将在未来的每一课中都很重要。

## Use It（如何使用）

2026 年，哪个家族适合哪个任务？

| Task | Best family | Why |
|------|-------------|-----|
| Photoreal faces, narrow domain | StyleGAN 2/3 | Still sharpest, fastest inference. |
| General text-to-image | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| Fast text-to-image | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| Speech + music | Token-based AR (AudioLM, VALL-E, MusicGen) or flow matching (AudioCraft 2) | Discrete tokens scale cheaply. |
| 3D scenes | Gaussian Splatting fit, diffusion prior | 3D-GS for reconstruction, diffusion for novel-view. |
| Density estimation (no sampling) | Flows | Only family with exact `log p(x)`. |
| Simulation / physics | Flow matching, score SDE | Straight-line paths, smooth vector fields. |

## Ship It（交付技能）

保存为 `outputs/skill-model-chooser.md`。

该技能接收一个任务描述，输出：(1) 使用哪个家族，(2) 三个开源和三个托管方案的排序列表，(3) 你应该注意的潜在失效模式，以及 (4) 计算/时间预算。

## Exercises（练习）

1. **Easy.** 对于以下五个产品，识别其家族和主干：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。证据应来自公开技术报告。
2. **Medium.** 你明天要读的那篇论文声称比扩散快 100 倍。写下三个问题，检验这个加速在条件化和高分辨率下是否仍然成立。
3. **Hard.** 选一个你关心的领域（如蛋白质结构、CAD、分子、轨迹）。为该领域当前的 SOTA 模型回答五个分诊问题，并草拟一个更好的模型会改变什么。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Generative model | "It makes new stuff" | Learns a sampler for `p_data(x)`, optionally exposes `log p(x)`. |
| Explicit density | "You can evaluate it" | Model provides a closed-form or tractable `log p(x)`. |
| Implicit density | "GAN-style" | Only a sampler — no way to evaluate `p(x)` of a given point. |
| ELBO | "Evidence lower bound" | A tractable lower bound on `log p(x)`; VAEs and diffusion optimize it. |
| Score | "Gradient of log-density" | `∇_x log p(x)`; diffusion and SDE models learn this field. |
| Manifold hypothesis | "Data lives on a surface" | High-dim data concentrates on a low-dim manifold; why dimensionality reduction works. |
| Autoregressive | "Predict the next piece" | Factorize joint as product of conditionals. |
| Latent | "Compressed code" | Low-dim representation from which a decoder can reconstruct the input. |

## Production note: five families, five inference shapes（生产备注：五大家族，五种推理形态）

每个家族对应不同的 inference-server 成本曲线。production-inference 文献将 LLM inference 框架化为 prefill + decode；同样的分解也适用于这里：

- **Autoregressive（桶 1 和 5）。** 顺序 decode 主导延迟；KV-cache、continuous batching 和 speculative decoding 都直接适用。
- **VAE / diffusion / flow-matching（桶 2 和 4）。** 没有 LLM 意义上的 decode。成本 = `num_steps × step_cost`，而 `step_cost` 是在完整隐变量分辨率上的 transformer 或 U-Net 前向。生产旋钮是步数（DDIM / DPM-Solver / distillation）、batch size 和精度（bf16 / fp8 / int4）。
- **GAN（桶 3）。** 一次前向传播。没有调度，没有 KV-cache。TTFT ≈ 总延迟。这就是为什么 StyleGAN 在窄域 UX 上仍然获胜。

当你在一篇论文摘要中看到"比扩散更快"时，把它翻译为"更少的步数 × 相同的步成本"或"相同的步数 × 更便宜的步成本"。其他都是营销。

## Further Reading（延伸阅读）

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — the GAN paper.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — the VAE paper.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — the DDPM paper.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — diffusion as an SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — the flow matching paper.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
