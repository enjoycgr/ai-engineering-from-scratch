# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP (2021) 证明了一个简单想法足以驱动接下来五年：用对比 loss (损失函数) 将图像编码器和文本编码器对齐到同一个向量空间，只使用嘈杂的网络图像-标题对。零监督标签。4 亿对。得到的 embedding (嵌入) 空间可以做 zero-shot (零样本) 分类、图像-文本检索，并作为每一个 2026 年 VLM 的视觉塔。SigLIP 2 (2025) 用 sigmoid 取代了 softmax，以更低成本超越了 CLIP。本课从 InfoNCE 数学走到 sigmoid 成对 loss，并用 stdlib Python 构建训练步骤。

**类型：** Build
**语言：** Python (stdlib, InfoNCE + sigmoid loss 实现)
**前置知识：** Phase 12 · 01 (ViT patch), Phase 7 (Transformers)
**时间：** ~180 分钟

## 学习目标

- 从互信息推导 InfoNCE loss，并实现一个数值稳定的向量化版本。
- 解释为什么 sigmoid 成对 loss (SigLIP) 能在 batch size (批量大小) 32768+ 下扩展，而没有 softmax 要求的 all-gather 开销。
- 通过构建文本模板（`a photo of a {class}`）并在余弦相似度上取 argmax 来运行 zero-shot ImageNet 分类。
- 说出 CLIP / SigLIP 预训练给你的四个杠杆：batch size、temperature (温度)、prompt template、数据质量。

## 问题

CLIP 之前的视觉是监督的。收集标注数据集（ImageNet：120 万张图像，1000 个类别），训练 CNN，发布。标签昂贵，标签偏向标注者能达成一致的东西，标签不转移到新任务除非 fine-tuning (微调)。

网络图像-标题有十亿多个松散标注的对，免费。一张金毛猎犬的照片配上 alt 文本 "my dog Max in the park" 携带了一个监督信号——文本描述了图像。问题是：你能把它变成有用的训练吗？

CLIP 的答案：把图像-标题对当作匹配任务。给定 N 张图像和 N 个标题，学习把每张图像与自身标题匹配，对抗 N-1 个干扰项。监督是"这两个东西属于一起；那 N-1 个不属于。"没有类别标签。没有人工标注。只有对比 loss。

得到的 embedding 空间做的比 CLIP 训练的还多。ImageNet zero-shot 有效，因为 "a photo of a cat" 嵌入在猫图片附近，而这些猫从未被显式标注为猫。这个赌注催生了每一个 2026 年的 VLM。

## 概念

### 双编码器 (Dual encoder)

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 transformer，每个标题输出一个 D 维向量。

两个塔都把输出归一化为单位长度。相似度是 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于 N 个（图像、标题）对的 batch，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是一个可学习的 temperature (温度)（CLIP 初始化为 0.07；以对数空间学习）。

### InfoNCE loss

CLIP 在行列上使用对称 cross-entropy (交叉熵)：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正例是它自己的标题
loss_t2i = CE(S^T, labels=identity)   # 每个标题的正例是它自己的图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像比 batch 中所有其他标题更匹配自己的标题。"负例"是 batch 中所有其他项。更大的 batch = 更多负例 = 更强的信号。CLIP 在 batch 32k 下训练；规模很重要。

### Temperature (温度)

`tau` 控制 softmax 的锐度。低 tau → 尖锐分布，硬负例挖掘效果。高 tau → 柔和，所有样本都贡献。CLIP 学习 `log(1/tau)`，裁剪以防止 collapse。SigLIP 2 固定初始 tau，改用可学习的 bias。

### 为什么 sigmoid 扩展更好 (SigLIP)

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须 all-gather 每个 embedding 到每个副本，然后做 softmax。这在通信量上与世界大小成二次方。

SigLIP 用逐元素 sigmoid 取代 softmax：对于每对 `(i, j)`，loss 是一个二元分类"这是匹配对吗？"正例类标签是对角线，其他一切是负例。Loss 是：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 如果 `i == j`，否则为 0。每对的 loss 是独立的。不需要 all-gather。每个 GPU 计算其本地块并求和。SigLIP 2 在 batch 32k-512k 下廉价扩展，而 CLIP 需要成比例的更多通信。

### Zero-shot 分类

给定 N 个类别名，对每个类别构建一个文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。Argmax 余弦相似度 = 预测类别。对目标类别没有训练。

Prompt template 很重要。CLIP 原始论文每个类别用了 80 个模板（plain、artistic、photo、painting 等）并平均 embedding。+3 ImageNet 分。现代使用通常选一个或两个模板。

### 线性探针和微调

Zero-shot 是基线。线性探针（在冻结 CLIP 特征上为你的目标类别训练一个线性层）在领域内任务上击败 zero-shot。完全 fine-tuning 在领域内击败线性探针，但会损害 zero-shot 迁移。三种方案，三种权衡。

### SigLIP 2：NaFlex 和密集特征

SigLIP 2 (2025) 添加了：
- NaFlex：单个模型处理可变宽高比和分辨率。
- 更好的密集特征用于分割和深度估计，针对作为 VLM 冻结主干的用途。
- 多语言：在 100+ 语言上训练，而 CLIP 仅英语。
- 1B 参数规模，CLIP 最高到 400M。

2026 年开源 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。CLIP 在纯图像-文本检索中仍是默认，其中特定的 LAION-2B 训练分布匹配你的查询模式。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN (Google, 2021)：与 CLIP 相同想法，18 亿对规模，90% 嘈杂。证明嘈杂数据可以扩展。OpenCLIP (LAION)：在 LAION-400M / 2B 上开源复现 CLIP，多种规模，首选开源 checkpoint。EVA-CLIP：从 masked image modeling 初始化；VLM 的强主干。BASIC：Google 的 CLIP+ALIGN 混合。都是同一家族，不同数据和调优。

### Zero-shot 天花板

CLIP 类模型在 ImageNet zero-shot 上封顶约 76%（CLIP-G、OpenCLIP-G）。超越需要要么更多数据（SigLIP 2 达到 80%+）要么架构改变（监督头、更多参数）。基准正在饱和；真正的价值是下游 VLM 消费的 embedding 空间。

## 使用它

`code/main.py` 实现：

1. 一个玩具双编码器（基于 hash 的图像特征、文本字符特征），让你无需 numpy 就能看到 InfoNCE 的形状。
2. 纯 Python 中的 InfoNCE loss（通过 log-sum-exp 数值稳定）。
3. 用于比较的 Sigmoid 成对 loss。
4. Zero-shot 分类例程：计算与一组文本 prompt 的余弦相似度，argmax 预测。

运行它并观察 loss 曲线。绝对数字是玩具级的；形状匹配真实 CLIP 训练器输出的。

## 交付它

本课产生 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，它用 CLIP 模板构建文本 prompt，用声明的 checkpoint（如 `openai/clip-vit-large-patch14`）嵌入两边，并返回 top-1 / top-5 预测和相似度分数。该 skill 拒绝对 prompt 列表中没有的类别做出声明。

## 练习

1. 手工实现 4 对 batch 的 InfoNCE。构建 4x4 相似度矩阵，运行 softmax，挑出对角线，计算 cross-entropy。验证你的 Python 实现与这个手工计算。

2. SigLIP 使用一个 bias 参数 `b` 加在 temperature 之外：`S'[i,j] = S[i,j]/tau + b`。当 batch 有大的类别不平衡（每行正例远少于负例）时，`b` 起什么作用？阅读 SigLIP Section 3（arXiv:2303.15343）。

3. 为猫 vs 狗构建一个 zero-shot 分类器。试两个 prompt template：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板 ensemble 是否击败单一模板？

4. 计算 softmax InfoNCE vs sigmoid 成对 loss 在 512-GPU、batch 32k 运行下的通信代价。哪个扩展为 O(N)，哪个为 O(N^2)？引用 SigLIP Section 4。

5. 阅读 OpenCLIP scaling-laws 论文（arXiv:2212.07143，Cherti 等人）。从图中复现他们对数据扩展的结论：在固定模型大小下，ImageNet zero-shot 准确率与训练数据大小之间是什么对数-线性关系？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| InfoNCE | "对比 loss" | Batch 相似度矩阵上的 cross-entropy；每个项的正例是其配对项，负例是所有其他项 |
| Sigmoid loss | "SigLIP loss" | 逐对二元 cross-entropy；无 softmax，无 all-gather，分布式训练中廉价扩展 |
| Temperature | "tau" | 在 softmax/sigmoid 前缩放 logits 的标量；控制分布锐度 |
| Zero-shot | "无微调分类" | 用文本 prompt 构建类别 embedding 并通过余弦相似度分类；对目标类别无训练 |
| Prompt template | "a photo of a ..." | 围绕类别名的文本支架；影响 zero-shot 准确率 1-5 分 |
| Dual encoder | "Two-tower" | 一个图像编码器 + 一个文本编码器，输出在共享 D 维空间中 |
| Hard negative | "困难干扰项" | 足够接近正例的负例，模型必须努力才能分开它们 |
| Linear probe | "冻结 + 一层" | 只在冻结特征上训练一个线性分类器；衡量特征质量 |
| NaFlex | "Native flexible resolution" | SigLIP 2 能力：无需 resize 即可摄入任意宽高比和分辨率的图像 |
| Temperature scaling | "对数参数化 tau" | CLIP 参数化 `log(1/tau)` 以便梯度表现良好；裁剪以防止 collapse 到接近零的 tau |

## 延伸阅读

- [Radford 等人 — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai 等人 — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia 等人 — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用嘈杂网络数据扩展。
- [Cherti 等人 — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling laws。
