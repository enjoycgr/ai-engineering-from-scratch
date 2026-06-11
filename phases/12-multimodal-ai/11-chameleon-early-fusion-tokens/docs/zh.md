# Chameleon 与早期融合 Token-Only 多模态模型

> 到目前为止我们见过的每个 VLM 都保持图像和文本分离。视觉 token 来自视觉编码器，流经 projector，然后在 LLM 内与文本相遇。视觉和文本词汇从不重叠。Chameleon (Meta, 2024 年 5 月) 问道：如果它们重叠呢？训练一个 VQ-VAE，把图像变成来自共享词汇的离散 token 序列。每个多模态文档现在是一个序列——文本 token 和图像 token 交错，单个自回归 loss。副作用：模型可以生成混合模态输出——在单个推理调用中交替文本和图像 token。本课阅读早期融合论点并从头到尾构建一个玩具版本。

**类型：** Build
**语言：** Python (stdlib, VQ-VAE tokenizer + 交错 decoder)
**前置知识：** Phase 12 · 05, Phase 8 (Generative AI)
**时间：** ~180 分钟

## 学习目标

- 解释为什么共享词汇 + 单个 loss 改变模型能做什么。
- 描述 VQ-VAE 如何把图像 token 化为与 transformer next-token 目标兼容的离散序列。
- 说出 Chameleon 的训练稳定性技巧：QK-Norm、dropout 放置、LayerNorm 排序。
- 对比 Chameleon 与 BLIP-2 的 Q-Former 方法并描述何时各是正确选择。

## 问题

基于 adapter 的 VLM（LLaVA、BLIP-2、Qwen-VL）把文本和图像当作两种不同东西。文本 token 经过 `embed(text_token)`；图像经过 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有两个输入路径在中途合并。

三个后果：

1. LLM 只能消费图像，不能发出图像。输出只有文本。
2. 混合模态文档（段落和图像交替，如文章）很笨拙——你要么在模型外解析多模态输入，要么链式生成。
3. 分布不匹配。视觉 token 和文本 token 生活在 hidden space 的不同区域，造成微妙对齐问题。

Chameleon 拒绝前提：图像只是来自共享词汇的离散 token 序列。在交错文档上训练模型，一个 loss，一个自回归 decoder，你免费解锁混合模态生成。

## 概念

### VQ-VAE 作为图像 tokenizer

Tokenizer 是一个向量量化变分自编码器。架构：

- 编码器：CNN + ViT，把图像映射到空间特征图，如 32x32 的 dim 256 特征。
- Codebook：K 个向量（Chameleon 使用 8192）的学习词汇，也是 dim 256。
- 量化：对每个空间特征，按 L2 距离查找最近的 codebook 条目。用整数索引替换连续特征。
- 解码器：CNN，把量化特征还原为像素。

训练：VAE reconstruction loss + commitment loss + codebook loss。Codebook 索引形成图像的离散字母表。

对 Chameleon：一张图像变成 32*32 = 1024 个 token，从 8192 的词汇中抽取。与文本 token（来自 LLM 的 BPE 词汇，如 32000）拼接。最终词汇：40192。Transformer 看到一个序列，一个 loss。

### 共享词汇

Chameleon 的词汇组合文本 token、图像 token 和模态分隔符。每个 token 有单一 ID。输入 embedding 层把每个 ID 映射到 D-dim hidden 向量。输出投影把 hidden 映射回词汇 logits。Softmax 挑选下一个 token，无论什么模态。

分隔符重要：`<image>` 和 `</image>` 标签括住图像 token 序列。生成时，如果模型发出 `<image>`，下游软件知道接下来 1024 个 token 是发给 decoder 进行像素渲染的 VQ 索引。

### 混合模态生成

推理是共享词汇中的 next-token 预测。示例 prompt："Draw a cat and describe it." Chameleon 发出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自主挑选顺序——它可以先图像后文本、先文本后图像、或交错。相同 decoder，相同 loss。

对比只能文本生成的 adapter VLM。Chameleon 重新打开模型输出模态的问题。

### 训练稳定性——QK-Norm、dropout、LayerNorm 排序

早期融合训练在规模上不稳定。Chameleon 论文记录了三个技巧：

- QK-Norm。在 attention 内部对 query 和 key 投影应用 LayerNorm，在 dot product 之前。防止深度 logit 幅度爆炸。被多个 2024 年后大型模型使用。
- Dropout 放置。在每个残差-add 后 dropout，不只 attention 和 MLP 后。当图像 token 的梯度可以主导时，需要更多正则化。
- LayerNorm 排序。残差分支上的 Pre-LN（标准），加上最后 block skip 连接上的额外 LN。稳定最终层梯度流。

没有这些技巧，34B 参数 Chameleon 训练在多个 checkpoint 上发散。有它们时收敛。训练配方与架构一样重要。

### Tokenizer 的重建天花板

VQ-VAE 是有损的。8192 codebook 条目和每 512x512 图像 1024 token 时，重建 PSNR 封顶约 26-28 dB。这足够可识别的图像生成，但明显差于连续空间 diffusion（Stable Diffusion 3 达到 32+ dB）。

Tokenizer 是瓶颈。更好的 tokenizer（MAGVIT-v2、IBQ、SBER-MoVQGAN）提升天花板。Emu3（Lesson 12.12）单独通过更好的 tokenizer 达到 SDXL 质量生成。

### Chameleon vs BLIP-2 / LLaVA

Chameleon（早期融合，共享词汇）：
- 一个 loss，一个 decoder。
- 生成混合模态输出。
- Tokenizer 是质量天花板。
- 昂贵：推理路径上每生成图像需 VQ-VAE decoder。

BLIP-2 / LLaVA（晚期融合，分离塔）：
- 视觉进，文本出。
- 重用预训练 LLM。
- 理解无 tokenizer 瓶颈。
- 便宜：单次 forward pass。

按任务挑选。如果你需要图像生成，Chameleon 家族。如果你只需要理解，adapter-VLM 更简单且重用更多预训练计算。

### Fuyu 和 AnyGPT

Fuyu (Adept, 2023) 是相关方法：完全跳过单独视觉编码器，把原始图像 patch 通过 LLM 输入投影作为 token 喂入，无 tokenizer。比 Chameleon 更简单，失去共享词汇输出生成。

AnyGPT (Zhan 等人, 2024) 把 Chameleon 扩展到四种模态：文本、图像、语音、音乐。每种用相同的 VQ-VAE 技巧，共享 transformer。Any-to-any 生成。Lesson 12.16 更多覆盖。

## 使用它

`code/main.py` 构建一个玩具端到端早期融合模型：

- 一个微型 VQ-VAE 风格量化器，把 8x8 patch 映射到 codebook 索引（K=16）。
- 共享词汇：(text ids 0..31) + (image ids 32..47) + (separators 48, 49)。
- 在合成标题 + 图像 token 序列上训练的玩具自回归 decoder（bigram 表）。
- 给定 prompt 发出交替文本 + 图像 token 的采样循环。

代码故意保持 transformer 微小（bigrams），因此你可以从头到尾追踪信号流。

## 交付它

本课产生 `outputs/skill-tokenizer-vs-adapter-picker.md`。给定产品规格（仅理解 vs 理解 + 生成、所需图像质量、成本预算），它在 Chameleon 家族（早期融合）和 LLaVA 家族（晚期融合）之间挑选并用定量经验法则证明。

## 练习

1. Chameleon 使用 K=8192 codebook 条目和每 512x512 图像 1024 token。估算 vs 24-bit RGB 图像的压缩比。它是有损的吗？多损？

2. 4K 图像（3840x2160）在相同 VQ-VAE 密度下产生多少图像 token？Chameleon 风格模型能在一次推理调用中生成 4K 图像吗？什么先崩溃——上下文、tokenizer 质量还是 KV cache？

3. 在纯 Python 中实现 QK-Norm。给定 64-dim query 和 key，展示 LayerNorm 前后的 dot product。为什么深度幅度控制重要？

4. 阅读 Chameleon Section 2.3 关于训练稳定性。描述论文在 34B 无 QK-Norm 时观察到的精确失败模式。"norm explosion" 签名是什么？

5. 扩展玩具 decoder，给定纯文本 prompt 发出混合模态响应。测量模型在训练数据分布 60% text-first / 40% image-first 下挑选 image-first vs text-first 的频率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Early fusion | "Unified tokens" | 图像从第一步起就转换为与 transformer 词汇共享的离散 token |
| VQ-VAE | "Image tokenizer" | CNN + ViT + codebook，把图像映射为 transformer 可预测的整数索引 |
| Shared vocabulary | "One dictionary" | 覆盖文本 + 图像 + 模态分隔符的单个 token ID 空间 |
| QK-Norm | "Attention stabilizer" | 在 dot product 前对 query 和 key 应用 LayerNorm，防止 norm 爆炸 |
| Mixed-modality generation | "Text + image output" | 推理自主产生交错文本和图像 token |
| Codebook size | "K entries" | VQ-VAE 可量化到的离散向量数；交易压缩与保真度 |
| Tokenizer ceiling | "Reconstruction limit" | 解码 VQ token 可达到的最佳 PSNR；限制模型图像质量 |

## 延伸阅读

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan 等人 — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu 等人 — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan 等人 — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
