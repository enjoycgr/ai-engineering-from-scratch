# Flamingo 与门控交叉注意力用于 Few-Shot VLM

> DeepMind 的 Flamingo (2022) 在任何人之前做了两件事。它表明单个模型可以处理任意交错的图像、视频和文本序列。并且它表明 VLM 可以做 in-context (上下文内) 学习——给一个包含三个示例（图像、标题）对的 few-shot (少样本) prompt，模型无需任何梯度步骤就能为一张新图像生成标题。机制：门控 cross-attention (交叉注意力) 层，插入在冻结 LLM 的现有层之间，带有一个学习的 tanh gate，在初始化时从零开始，因此 LLM 的文本能力在初始化时被保留。本课走过 Flamingo 的 Perceiver resampler 和门控 cross-attention 架构——Gemini 交错输入和 Idefics2 视觉 token 的祖先。

**类型：** Learn
**语言：** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**前置知识：** Phase 12 · 03 (BLIP-2 Q-Former)
**时间：** ~120 分钟

## 学习目标

- 解释门控 cross-attention 如何通过 tanh(gate) = 0 在初始化时保留冻结 LLM 的文本能力。
- 走过 Perceiver resampler：N 个图像 patch → K 个固定"latent" query 通过 cross-attention。
- 描述 Flamingo 如何处理交错图像-文本序列，使用尊重图像位置的 causal masking (因果掩码)。
- 复现一个 few-shot 多模态 prompt 结构（3 个图像-标题示例然后一个查询图像）。

## 问题

BLIP-2 把 32 个视觉 token 喂给冻结 LLM 的输入层。对每张 prompt 一张图像有效。但如果你想喂*很多*交错的图像和文本，比如 "here is image A, caption it; here is image B, caption it; now here is image C, caption it"？LLM 的 self-attention 需要处理单一流中的图像 token 和文本 token，而哪些位置可以关注哪些图像的问题变得棘手。

Flamingo 的答案：完全不要改变 LLM 的输入流。在现有 LLM block 之间插入额外的 cross-attention 层。文本 token 仍然像往常一样流过 LLM 的 causal self-attention。在每几个 LLM block 之间，文本 token 也通过新的门控层 cross-attend 到图像特征。Gate（初始化为零）意味着在第 0 步新层是 no-op——模型表现得完全像预训练的 LLM。随着训练进行，gate 打开，视觉信息开始流动。

Flamingo 回答的第二个问题：如何处理每张 prompt 可变数量的图像（0、1 或多张）？一个 Perceiver resampler——一个小的 cross-attention 模块，接收任意数量的 patch 并产生固定数量的视觉 latent token。LLM cross-attention 层无论 prompt 中有多少图像，看到的形状都相同。

## 概念

### 冻结 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 开始。所有 70B 权重 untouched。现有的文本 self-attention 和 FFN 正常运作。

### Perceiver resampler

对于 prompt 中的每张图像，ViT 产生 N 个 patch token。Perceiver resampler 有 K 个固定可学习 latent（Flamingo 使用 K=64）。每个 resampler block 是两个子步骤：

1. Cross-attention：K 个 latent 关注 N 个 patch token（Q 来自 latent，K/V 来自 patch）。
2. Latent 内部的 self-attention + FFN。

6 个 resampler block 后，输出是 dim 1024 的 K=64 个视觉 token，无论 ViT 产生了多少 patch。224x224 图像（196 个 patch）和 480x480 图像（900 个 patch）都作为 64 个 resampler token 退出。

对于视频，resampler 在时序上应用：每帧的 patch 产生 64 个 latent，时序位置编码让模型区分 t=0 和 t=N。完整视频变成 T * 64 个视觉 token。

### 门控 cross-attention

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控 cross-attention block：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是一个初始化为零的可学习标量。
- `tanh(0) = 0`，因此 init 时门控分支贡献为零。
- 随着 `alpha` 偏离零，cross-attention 贡献平滑增长。
- 残差连接意味着即使完全打开的门也不会覆盖 LLM 的文本表示；它只是在其上添加视觉信息。

这是 Flamingo 中最重要的设计选择：视觉条件是加性的、门控的、在初始化时为零。第 0 步的 Flamingo 是文本-only 输入上完美的 Chinchilla 70B。

### 交错输入的掩码 cross-attention

在类似 "<image A> caption A <image B> caption B <image C> ?" 的 prompt 中，每个文本 token 应该只看到序列中排在它前面的图像。Cross-attention mask 强制：位置 `t` 的文本 token 只关注图像索引 `i < i_t` 的图像 resampler token，其中 `i_t` 是位置 `t` 之前最近的图像。"只看最后一个前面的图像"或"看所有前面图像"都是有效选择；Flamingo 选了前者。

### In-context few-shot 学习

Flamingo prompt 看起来像这样：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式并输出 "bird"（或 image3 显示的任何东西）。无梯度步骤。冻结 LLM 的 in-context 学习能力通过门控 cross-attention 传递——这是论文的 punchline 和为什么它重要。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb (M3W)：4300 万个带交错图像和文本的网页，重建阅读顺序。
2. 图像-文本对 (ALIGN + LTIP)：44 亿对。
3. 视频-文本对 (VTP)：2700 万个短视频片段。

OBELICS (2023) 是交错网页语料库的开源复现，Idefics、Idefics2 和大多数开源 "Flamingo 风格"模型在其上训练。

### OpenFlamingo 和 Otter

OpenFlamingo (2023) 是开源复现。架构相同（Perceiver resampler + 冻结 LLaMA 或 MPT 上的门控 cross-attention）。Checkpoint 在 3B、4B、9B。质量落后 Flamingo 由于更小的基础 LLM 和更少数据。

Otter (2023) 在 MIMIC-IT（多模态指令数据集）上基于 OpenFlamingo 做指令微调，表明门控 cross-attention 也适用于指令遵循。

### 后代

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控 cross-attention 谱系，逐步简化（Idefics2 去掉 resampler，改用直接 patch token 加 adaptive pooling）。
- Flamingo 到 Chameleon 的过渡：到 2024 年许多团队转向 early-fusion (早期融合)（Lesson 12.11）；Flamingo 风格门控 cross-attention 在需要冻结主干的生产中保留。
- Gemini 的交错输入：概念上继承 Flamingo 的交错格式灵活性，尽管确切机制是专有的。

### 与 BLIP-2 对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥 | Q-Former 只在输入处 | 每 M 层的门控 cross-attention |
| 视觉 token | 每张图像 32 个 | 每张图像每 cross-attn 层 64 个 |
| 冻结 LLM | 是 | 是 |
| Few-shot in-context | 弱 | 强——论文的核心亮点 |
| 交错输入 | 无原生支持 | 是，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数量 | 188M 训练 | ~10B 训练（cross-attn 层） |
| 计算 | 8 A100 天 | 数千 TPUv4 周 |

预算内单图像 VQA 选 BLIP-2。交错、few-shot、或多图像推理选 Flamingo/Idefics2。

## 使用它

`code/main.py` 演示：

1. 36 个 fake patch token 上的 Perceiver resampler，8 个可学习 latent（纯 Python cross-attention）。
2. 一个门控 cross-attention 步骤，`alpha = 0` → 输出等于输入（LLM 未变），然后 `alpha = 2.0` → 视觉贡献混入。
3. 交错 mask builder，为 "(image 1) (text 1) (image 2) (text 2)" 序列产生 2D attention mask。

## 交付它

本课产生 `outputs/skill-gated-bridge-diagnostic.md`。给定一个开放 VLM 的配置（resampler Y/N、cross-attn 频率、gate 方案），它识别 Flamingo 谱系元素并解释冻结策略。对调试为什么 fine-tune 降低了文本性能有用（答案：gate 开得太快太宽）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：9B LLM + 1.4B 门控 cross-attention 层 + 64M resampler。总参数中多大比例是训练的？

2. 在 PyTorch 中实现门控残差 `y = tanh(alpha) * cross + x`。实验表明 `alpha=0` 时，`y==x` 在 init 时精确成立。

3. 阅读 OpenFlamingo Section 3.2（arXiv:2308.01390）关于他们如何处理一个 batch 中每张 prompt 图像数量不同。描述 padding 策略。

4. 为什么 Flamingo 的 cross-attention mask 让文本 token 只关注*最近的*前面图像，而不是所有前面图像？阅读 Flamingo 论文 Section 2.4 并解释权衡。

5. In-context few-shot：为一个新的 Flamingo 变体构造一个包含 4 个 "image → color of main object" 示例的 prompt。描述当你把示例数量从 0 变到 8 时的预期准确率模式。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Perceiver resampler | "Fixed-latent cross-attention" | 从可变数量输入 patch 产生 K 个固定 token 的模块 |
| Gated cross-attention | "Tanh-gated bridge" | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，init 为 0 |
| Interleaved input | "Mixed sequence" | 图像和文本按阅读顺序自由混合的 prompt 格式 |
| Frozen LLM | "No LLM gradients" | 文本 LLM 权重不更新；只有 resampler + cross-attn 层训练 |
| Few-shot | "In-context examples" | 在 prompt 中给几个（图像、答案）对；模型无需 fine-tuning 即可泛化 |
| OBELICS | "Interleaved web corpus" | 1.41 亿个带图像和文本按阅读顺序排列的网页的开源数据集 |
| Chinchilla | "70B frozen base" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| Gate schedule | "How alpha moves" | 训练期间 cross-attention gate 打开的速度 |
| Cross-attn frequency | "Every M layers" | 门控 cross-attention block 插入频率；Flamingo 使用 M=4 |
| OpenFlamingo | "Open reproduction" | MosaicML/LAION 在 3-9B 的开源 checkpoint；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac 等人 — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla 等人 — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开源复现。
- [Laurençon 等人 — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网页语料库。
- [Jaegle 等人 — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li 等人 — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后代。
- [Laurençon 等人 — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化。
