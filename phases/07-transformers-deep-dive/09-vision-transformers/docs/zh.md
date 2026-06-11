# Vision Transformers (ViT, 视觉 Transformer)

> 图像是一组 patch 的网格。句子是一组 token 的网格。同一个 transformer 可以处理两者。

**类型:** 构建
**语言:** Python
**前置知识:** Phase 7 · 05 (完整 Transformer), Phase 4 · 03 (CNN), Phase 4 · 14 (Vision Transformers 入门)
**时间:** ~45 分钟

## 问题背景

在 2020 年之前，计算机视觉意味着卷积。ImageNet、COCO 和检测基准上的每个 SOTA（State-of-the-Art，最先进模型）都使用 CNN 骨干网络。Transformer 则是用于语言的。

Dosovitskiy 等人（2020）——"An Image is Worth 16x16 Words"（一张图像值 16x16 个词）—— 展示了你可以完全抛弃卷积。将图像切成固定大小的 patch，将每个 patch 线性投影到 embedding（嵌入向量）中，然后将序列送入一个普通的 transformer encoder（编码器）。在足够大的规模下（ImageNet-21k 预训练或更大），ViT 可以匹敌或超越基于 ResNet 的模型。

ViT 开启了 2026 年一个更广泛的范式：一种架构，多种模态。Whisper 将音频 token 化。ViT 将图像 token 化。机器人使用 action token（动作 token）。视频使用 pixel token（像素 token）。Transformer 并不关心 —— 给它一个序列，它就能学习。

到 2026 年，ViT 及其衍生模型（DeiT、Swin、DINOv2、ViT-22B、SAM 3）主导了大部分视觉领域。CNN 在边缘设备和延迟敏感任务上仍有优势。其他所有领域都在某个地方使用了 ViT。

## 核心概念

![图像 → patch → token → transformer](../assets/vit.svg)

### 步骤 1 — patchify（分块）

将 `H × W × C` 的图像分割成 `N × (P·P·C)` 的扁平 patch 序列。典型配置：`224 × 224` 图像，`16 × 16` patch → 196 个 patch，每个 768 维。

```
image (224, 224, 3) → 14 × 14 网格的 16x16x3 patch → 196 个长度为 768 的向量
```

Patch size（patch 大小）是一个关键杠杆。更小的 patch = 更多 token、更好的分辨率、二次方的注意力成本。更大的 patch = 更粗糙、更便宜。

### 步骤 2 — 线性嵌入（linear embedding）

一个单一的可学习矩阵将每个扁平 patch 投影到 `d_model`。等价于一个卷积核大小为 `P`、步幅为 `P` 的卷积。在 PyTorch 中，这 literally（ literally ）就是 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` —— 两行代码的实现。

### 步骤 3 — 前置 `[CLS]` token，添加位置嵌入（positional embeddings）

- 前置一个可学习的 `[CLS]` token。它的最终隐藏状态是用于分类的图像表示。
- 添加可学习的位置嵌入（ViT 原版）或二维正弦位置编码（后续变体）。
- 2024 年后，RoPE（Rotary Position Embedding，旋转位置编码）扩展到 2D 用于位置编码，有时不需要显式的嵌入。

### 步骤 4 — 标准 transformer encoder

堆叠 L 个 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 块。与 BERT 完全相同。没有视觉特定的层。这是该论文的教学核心。

### 步骤 5 — 头部（head）

对于分类：取 `[CLS]` 隐藏状态 → 线性层 → softmax。对于 DINOv2 或 SAM，丢弃 `[CLS]`，直接使用 patch embedding。

### 重要的变体

| 模型 | 年份 | 变化 |
|-------|------|--------|
| ViT | 2020 | 原版。固定 patch 大小，全局注意力。 |
| DeiT | 2021 | 蒸馏（distillation）；仅在 ImageNet-1k 上可训练。 |
| Swin | 2021 | 分层结构 + 移位窗口。固定亚二次方成本。 |
| DINOv2 | 2023 | 自监督（无标签）。最佳通用视觉特征。 |
| ViT-22B | 2023 | 22B 参数；适用缩放定律（scaling laws）。 |
| SigLIP | 2023 | ViT + 语言配对，sigmoid 对比损失。 |
| SAM 3 | 2025 | 分割一切；ViT-Large + 可提示的 mask decoder（掩码解码器）。 |

### 为什么 ViT 花了一段时间才普及

ViT 需要*大量*数据才能匹敌 CNN，因为它没有 CNN 的归纳偏置（平移不变性、局部性）。没有超过 1 亿张标注图像或强自监督预训练，CNN 在相同计算量下仍然获胜。DeiT 在 2021 年通过蒸馏技巧解决了这个问题；DINOv2 在 2023 年通过自监督永久解决了这个问题。

## 动手构建

参见 `code/main.py`。纯标准库的 patchify + 线性嵌入 + 合理性检查。没有训练 —— 任何现实规模的 ViT 都需要 PyTorch 和数小时的 GPU 时间。

### 步骤 1: 伪造图像

一个 24 × 24 的 RGB 图像，表示为行的列表，每行是 `(R, G, B)` 元组。我们使用 6×6 的 patch → 16 个 patch，每个 108 维的嵌入向量。

### 步骤 2: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

光栅顺序：网格上的行优先。每个 ViT 都使用这种顺序。

### 步骤 3: 线性嵌入

将每个扁平 patch 乘以一个随机的 `(patch_flat_size, d_model)` 矩阵。验证前置 `[CLS]` 后的输出形状为 `(N_patches + 1, d_model)`。

### 步骤 4: 计算现实 ViT 的参数数量

打印 ViT-Base 的参数数量：12 层、12 头、d=768、patch=16。与 ResNet-50（~25M）比较。ViT-Base 约为 ~86M。ViT-Large ~307M。ViT-Huge ~632M。

## 如何使用

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 个 patch
cls_emb = out[:, 0]                       # 图像表示
```

**DINOv2 嵌入是 2026 年图像特征的默认选择。** 冻结骨干网络，训练一个微小的 head。适用于分类、检索、检测、描述。Meta 的 DINOv2 检查点在每个非文本视觉任务上都优于 CLIP。

**Patch size 选择。** 小模型使用 16×16（ViT-B/16）。密集预测（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 落地应用

参见 `outputs/skill-vit-configurator.md`。该 skill 根据数据集大小、分辨率和计算预算，为新视觉任务选择 ViT 变体和 patch size。

## 练习

1. **简单。** 运行 `code/main.py`。验证 patch 数量等于 `(H/P) * (W/P)`，扁平 patch 维度等于 `P*P*C`。
2. **中等。** 实现二维正弦位置嵌入 —— 为每个 patch 的 `row`（行）和 `col`（列）分别生成两个独立的正弦编码，然后拼接。将其输入一个微型 PyTorch ViT，并与 CIFAR-10 上的可学习位置嵌入比较准确率。
3. **困难。** 构建一个 3 层 ViT（PyTorch），在 1,000 张 MNIST 图像上训练，使用 4×4 patch。测量测试准确率。现在在同一 1,000 张图像上添加 DINOv2 预训练（简化版：仅训练 encoder 从 masked patch 预测 patch embedding）。准确率有提升吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Patch | "视觉 transformer 的 token" | 图像中 `P × P × C` 区域的像素值扁平向量。 |
| Patchify | "切碎 + 扁平化" | 将图像切片成不重叠的 patch，每个扁平化为向量。 |
| `[CLS]` token | "图像摘要" | 前置的可学习 token；其最终嵌入是图像表示。 |
| Inductive bias（归纳偏置） | "模型假设了什么" | ViT 比 CNN 的先验更少；需要更多数据来弥补差距。 |
| DINOv2 | "自监督 ViT" | 使用图像增强 + 动量教师（momentum teacher）进行无标签训练。2026 年最佳通用图像特征。 |
| SigLIP | "CLIP 的继任者" | 使用 sigmoid 对比损失训练的 ViT + 文本编码器；在相同计算量下优于 CLIP。 |
| Swin | "窗口化 ViT" | 分层 ViT，局部注意力 + 移位窗口；亚二次方成本。 |
| Register tokens（寄存器 token） | "2023 年的技巧" | 几个额外的可学习 token，用于吸收注意力汇点（attention sinks）；改善 DINOv2 特征。 |

## 延伸阅读

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 论文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2 的 register token 修复。
