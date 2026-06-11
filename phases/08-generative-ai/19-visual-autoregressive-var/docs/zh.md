# 视觉自回归建模 (VAR)：下一尺度预测

> 扩散模型在时间中迭代采样（去噪步骤）。VAR 在尺度中迭代采样 —— 它先预测一个 1×1 token，然后是 2×2，然后是 4×4，直到最终分辨率，每个尺度都以之前的尺度为条件。2024 年的论文表明，VAR 在图像生成上匹配 GPT 风格的缩放定律，并在相同计算预算下击败 DiT。本节课构建核心机制。

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 分钟

## 问题

自回归生成主导了语言建模，因为它可预测地缩放：更多计算、更多参数、更低的 perplexity (困惑度)、更好的输出。图像生成在 2024 年之前有两种主要的 AR 尝试：PixelRNN/PixelCNN（逐像素）和 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 码上逐 token）。

两者都 suffers from a generation-order problem (生成顺序问题)。像素和 token 排列在二维网格中，但 AR 模型必须以一维光栅顺序访问它们。一个早期角落像素对图像最终会变成什么一无所知。生成质量的缩放比 GPT-on-text 更差，在匹配计算下从未达到扩散模型的质量。

VAR 通过改变生成对象来修复生成顺序问题。它不是逐个预测空间中的图像 token，而是预测分辨率递增的整张图像。步骤 1：预测 1×1 token（整体图像"摘要"）。步骤 2：预测 2×2 token 网格（粗略特征）。步骤 3：预测 4×4 网格。步骤 K：预测最终的 (H/8)×(W/8) 网格。

每个尺度 attend to (关注) 所有之前的尺度（在"尺度顺序"中因果地）并在其自身尺度内并行。顺序问题消失了：尺度 k 上的整张图像在一次 transformer pass (transformer 前向传播) 中产生。

## 概念

### VQ-VAE 多尺度分词器

VAR 需要一个 **multi-scale discrete tokenizer (多尺度离散分词器)**。对于图像 x，它产生一系列分辨率递增的 token 网格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每个 z_k 使用相同的 codebook (码本)（典型大小 4096–16384）。每个尺度的 tokenization 不是独立的 —— 它的训练目标是让在每个尺度上累加残差来重构 f：

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

这是一种 **residual VQ (残差 VQ)** 变体。尺度 k 捕捉了尺度 1..k-1 遗漏的内容。解码器取所有尺度 embedding 的和并生成图像。

多尺度 VQ 分词器只训练一次（像 VQGAN）然后冻结。所有生成工作都由其上方的自回归模型完成。

### 下一尺度预测

生成模型是一个 transformer，它看到所有之前尺度的 token 并预测下一尺度的 token。

输入序列结构：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embeddings (位置嵌入) 同时编码 scale index (尺度索引) 和尺度内的 spatial position (空间位置)。Attention (注意力) 在尺度顺序中是 causal (因果的)：尺度 k 中位置 (i, j) 的 token 可以关注尺度 1..k 中的所有 token，以及尺度 k 中在任意 intra-scale order (尺度内顺序) 中更早出现的 token（VAR 使用固定的位置注意力，没有 intra-scale causality (尺度内因果性) —— 一个尺度内的所有位置都是并行预测的）。

Training loss (训练损失)：在每个尺度 k，给定所有之前尺度的 token 预测 token z_k。在离散 VQ 码上的 cross-entropy loss (交叉熵损失)。结构与 GPT 相同，只是"序列"现在是尺度结构的。

### 生成

在推理时：
```
generate z_1 = sample from p(z_1)                    # 1 个 token
generate z_2 = sample from p(z_2 | z_1)              # 4 个 token 并行
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 个 token 并行
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

对于 K = 10 个尺度，生成是 10 次 transformer 前向传播。每次前向传播在其整个尺度内并行产生 —— 尺度内没有逐 token 自回归。对于 256×256 图像，这大约是 10 次传播 vs DiT 的 28–50 次。

### 为什么下一尺度胜过下一 Token

三个结构性优势：
1. **Coarse-to-fine (由粗到细) 与自然图像统计一致。** 人类视觉感知和图像数据集都表现出尺度相关的规律性：低频结构稳定且可预测；高频细节以低频内容为条件。下一尺度预测利用了这一点。
2. **尺度内并行生成。** 与 GPT 风格 token AR 不同，VAR 一步产生尺度内的所有 token。有效生成长度是对数级的而不是线性的。
3. **无生成顺序偏差。** 尺度 k 的 token 看到整个尺度 k-1；没有"左侧"或"上方"偏差迫使早期 token 在晚期上下文可用之前提交。

### 缩放定律

Tian 等人证明 VAR 在 ImageNet 上遵循 FID 的幂律缩放曲线 —— 就像 GPT 对 perplexity 所做的那样。翻倍参数或计算可靠地减半误差。这是第一种图像生成模型，能像语言模型一样干净地展示这种缩放行为。结果是 VAR 尺度的预测可以从计算量推断，而不是每个架构的实证猜测。

### 与扩散的关系

VAR 和扩散共享相同的数据压缩故事：两者都将生成问题分解为一系列更简单的子问题。

- 扩散：逐步添加噪声，学习撤销一步。
- VAR：逐步添加分辨率，学习预测下一尺度。

它们是问题的不同轴向。两者都产生可处理的条件分布。实证上 VAR 在推理时更快（更少传播，尺度内全部并行），在类别条件 ImageNet 上匹配或击败 DiT。文本条件 VAR (VARclip, HART) 是一个活跃的研究方向。

## 亲手构建

在 `code/main.py` 中，你将：
1. 在合成"图像"数据（二维高斯环）上构建一个微小的 **multi-scale VQ tokenizer (多尺度 VQ 分词器)**。
2. 训练一个 **VAR-style transformer (VAR 风格 transformer)** 来下一尺度预测 token。
3. 通过调用 transformer 4 次（4 个尺度）来采样并解码。
4. 验证尺度顺序训练使生成在尺度内并行。

这是一个玩具实现。重点是看到尺度结构化的 attention mask (注意力掩码) 和尺度内并行生成实际工作。

## 交付它

本节课产出 `outputs/skill-var-tokenizer-designer.md` —— 一个设计多尺度分词器的技能：尺度数量、尺度比例、码本大小、残差共享、解码器架构。

## 练习

1. **尺度数量消融。** 用 4、6、8、10 个尺度训练 VAR。测量重建质量 vs 自回归传播次数。更多尺度 = 更精细残差 = 更好质量但更多传播。
2. **码本大小。** 用码本大小 512、4096、16384 训练分词器。更大的码本给出更好重建但更难预测。找到拐点。
3. **尺度内并行检查。** 对于训练好的 VAR，显式测量注意力模式。在尺度 k 内，模型是否关注跨尺度位置而不关注尺度内？验证 mask 实现。
4. **VAR vs DiT 缩放。** 对于相同的 ImageNet 类别条件任务，在匹配参数预算下训练 VAR 和 DiT（例如 33M、130M、458M）。绘制 FID vs 计算量。VAR 在每个尺寸上应领先于 DiT —— 在小尺度上复现论文结果。
5. **文本条件。** 将 VAR 扩展为接受文本 embedding（CLIP pooled）作为通过 adaLN 的额外条件输入。这是 HART 的配方。FID 在文本对齐采样上改善了多少？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| VAR | "Visual AutoRegressive" | 通过在 VQ token 网格金字塔上的下一尺度预测进行图像生成 |
| Next-scale prediction | "先预测粗的，再预测细的" | 模型在分辨率递增的尺度上预测 token，以所有之前尺度为条件 |
| Multi-scale VQ tokenizer | "Residual VQ" | 产生 K 个分辨率递增 token 网格的 VQ-VAE，解码器累加所有尺度 |
| Scale k | "金字塔第 k 层" | K 个分辨率级别之一，从 k=1 的 1×1 到 k=K 的 (H/p)×(W/p) |
| Parallel-within-scale | "每个尺度一次前向" | 尺度 k 的所有 token 在一次 transformer 传播中预测，不是自回归地 |
| Causal-across-scales | "尺度顺序注意力" | 尺度 k 的 token 可以关注尺度 1..k 的全部，但不能关注尺度 k+1..K |
| Residual VQ | "Additive tokenization" | 每个尺度的 token 编码低尺度遗漏的残差；解码器累加所有尺度 embedding |
| VAR scaling law | "图像 GPT 缩放" | FID 在计算量上遵循可预测的幂律，像语言模型的困惑度一样 |
| HART | "混合 VAR + 文本" | 结合 MaskGIT 风格迭代解码与 VAR 尺度结构的文本条件 VAR 变体 |
| Scale position embedding | "(scale, row, col) 三元组" | 位置编码同时携带尺度索引和尺度内的空间坐标 |

## 延伸阅读

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) —— VAR 论文，权威参考
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) —— DiT，扩散比较基线
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) —— VQGAN，VAR 多尺度分词器扩展的分词器家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) —— VQ-VAE，离散图像 tokenization 的基础
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) —— 文本条件 VAR
