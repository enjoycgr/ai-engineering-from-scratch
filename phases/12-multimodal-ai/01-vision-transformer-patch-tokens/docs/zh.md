# Vision Transformers 与 Patch-Token 原语

> 在一切多模态之前，图像必须先变成 transformer 能够处理的 token 序列。2020 年的 ViT 论文用 16x16 像素 patch、线性投影和位置 embedding (嵌入) 回答了这个问题。五年后，每一个 2026 年的前沿模型（Claude Opus 4.7 的 2576px 原生分辨率、Gemini 3.1 Pro、Qwen3.5-Omni）仍然从这里开始——编码器从 ViT 演进到了 DINOv2 再到 SigLIP 2，加入了 register tokens (寄存器 token)，位置方案变成了 2D-RoPE，但原语始终未变。本课从头到尾阅读 patch-token 流水线，并用 stdlib Python 构建它，让 Phase 12 的其余部分对"视觉 token"有一个具体的心智模型。

**类型：** Learn
**语言：** Python (stdlib, patch tokenizer + geometry calculator)
**前置知识：** Phase 7 (Transformers), Phase 4 (Computer Vision)
**时间：** ~120 分钟

## 学习目标

- 将一张 HxWx3 的图像转换为带有正确位置编码 (positional encoding) 的 patch token 序列。
- 为一个给定配置（patch size、分辨率、hidden dim、depth）的 ViT 计算序列长度、参数量和 FLOPs。
- 说出让 ViT 从 2020 年研究变成 2026 年生产的三大升级：自监督预训练 (self-supervised pretraining)（DINO / MAE）、register tokens、原生分辨率打包 (native-resolution packing)。
- 针对下游任务，在 CLS pooling、mean pooling 和 register tokens 之间做出选择。

## 问题

Transformers 处理向量序列。文本已经是序列（字节或 token）。图像是一个二维像素网格加三个颜色通道——不是序列。如果你展平每个像素，一张 224x224 的 RGB 图像变成 150,528 个 token，而 self-attention (自注意力) 在这个长度下是不可行的（序列长度的二次方）。

2020 年之前的方法是在前面接一个 CNN 特征提取器：ResNet 生成一个 7x7 的 2048 维特征图，把 49 个 token 喂给 transformer。这可行，但继承了 CNN 的偏差（平移等变性、局部感受野），也失去了 transformer 对规模的胃口。

Dosovitskiy 等人（2020）提出了一个直白的问题：如果我们跳过 CNN 呢？把图像切成固定大小的 patch（比如 16x16 像素），将每个 patch 线性投影成一个向量，加上位置 embedding，然后把序列喂给一个 vanilla transformer。这在当时被视为异端——没有卷积的计算机视觉。有了足够的数据（JFT-300M，然后是 LAION），它在 ImageNet 上击败了 ResNet 并且持续改进。

到 2026 年，ViT 原语是毋庸置疑的基础。每个开源权重的 VLM 的视觉塔都是某种后代（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是"应该用 patch 吗？"而是"什么 patch size、什么分辨率方案、什么预训练目标、什么位置编码。"

## 概念

### Patch 作为 token

给定一张形状为 `(H, W, 3)` 的图像 `x` 和一个 patch size `P`，你把图像切成 `(H/P) x (W/P)` 个不重叠的 patch 网格。每个 patch 是一个 `P x P x 3` 的像素立方体。将每个立方体展平为一个 `3 P^2` 的向量。应用一个共享的线性投影 `W_E`，形状为 `(3 P^2, D)`，把每个 patch 映射到模型的 hidden dimension `D`。

对于 ViT-B/16 的标准配置：
- 分辨率 224，patch size 16 → 网格 14x14 → 196 个 patch token。
- 每个 patch 是 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` token → 序列长度 197。

Patch 投影在数学上等同于一个 2D 卷积，kernel size `P`，stride `P`，`D` 个输出通道。这就是生产代码实际实现它的方式——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"的视角是概念性的；kernel 视角是高效的。

### 位置嵌入 (Positional embeddings)

Patch 没有固有顺序——transformer 把它们看作一个集合。早期的 ViT 添加了可学习的 1D 位置 embedding（每个位置一个 768 维向量，共 197 个）。可行，但把模型绑在训练分辨率上：推理时如果改变网格大小，必须插值位置表。

现代的视觉主干使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或分解式 2D 位置。2D-RoPE 基于 patch 的（行、列）索引旋转 query 和 key 向量，因此模型从旋转角度推断相对 2D 位置。没有位置表。模型在推理时处理任意网格大小。

### CLS token、池化输出和 register tokens

图像级别的表示是什么？三种选择并存：

1. `[CLS]` token。在 patch 序列前添加一个可学习向量。经过所有 transformer block 后，CLS token 的 hidden state 就是图像表示。继承自 BERT。被原始 ViT、CLIP 使用。
2. Mean pool (均值池化)。对 patch token 的输出 hidden state 取平均。被 SigLIP、DINOv2、大多数现代 VLM 使用。
3. Register tokens (寄存器 token)。Darcet 等人（2023）观察到，没有显式 sink token 训练的 ViT 会发展出高范数的"伪影"patch，劫持 self-attention。添加 4-16 个可学习的 register token 吸收这个负载，提升密集预测质量（分割、深度）。DINOv2 和 SigLIP 2 都带有 register。

这个选择对下游任务很重要。CLS 对分类足够好。对于把 patch token 喂给 LLM 的 VLM，你完全跳过 pooling——每个 patch 变成 LLM 的输入 token。Register 在交接前被丢弃（它们是脚手架，不是内容）。

### 预训练：监督、对比、掩码、自蒸馏

2020 年的 ViT 用 JFT-300M 上的监督分类预训练。很快被取代：

- CLIP (2021)：4 亿对上的对比图像-文本。Lesson 12.02。
- MAE (2021, He 等人)：mask 75% 的 patch，重建像素。自监督，纯图像即可。
- DINO (2021) / DINOv2 (2023)：学生-教师的自蒸馏，无标签，无标题。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉主干，是"密集特征"用例的默认选择。
- SigLIP / SigLIP 2 (2023, 2025)：用 sigmoid loss 和 NaFlex 原生宽高比的 CLIP。2026 年开源 VLM（Qwen、Idefics2、LLaVA-OneVision）的主导视觉塔。

你选择的预训练决定了主干擅长什么：CLIP/SigLIP 用于与文本的语义匹配，DINOv2 用于密集视觉特征，MAE 作为下游 fine-tuning (微调) 的起点。

### 扩展定律 (Scaling laws)

ViT 扩展（Zhai 等人 2022）确立了 ViT 的质量在模型大小、数据大小和计算量上遵循可预测的规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch size 是序列长度与保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型值）每张图像产生更多 token；对 OCR 和密集任务更好，速度更慢。
- 分辨率是另一个大杠杆。从 224 到 384 到 512 几乎总是有帮助，但 FLOPs 是二次增长的。

ViT-g/14（1B 参数，patch 14，分辨率 224 → 256 个 token）和 SigLIP SO400m/14（400M 参数，patch 14）是 2026 年开源 VLM 的两个主力编码器。

### ViT 的参数量

完整计算在 `code/main.py` 中。对于 ViT-B/16 @ 224：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载 checkpoint 之前这样估算每个 ViT。主干大小设置了任何下游 VLM 的 VRAM 下限。

### 2026 年生产配置

2026 年大多数开源 VLM 配备的编码器是原生分辨率的 SigLIP 2 SO400m/14（NaFlex）。它有：
- 400M 参数。
- Patch size 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 图像级任务用 mean pool；VQA 时全部 729 个 patch 流入 LLM。
- 4 个 register token，在 LLM 交接前丢弃。
- 2D-RoPE 带图像级缩放，支持原生宽高比。

那个配置中的每个决定都可以追溯到一篇你可以阅读的论文。

## 使用它

`code/main.py` 是一个 patch tokenizer 和几何计算器。给定（图像 H, W, patch P, hidden D, depth L），它报告：

- Patch 后的网格形状和序列长度。
- 一张合成 8x8 像素玩具图像的 token 序列（走过展平+投影路径）。
- 按 patch embed、位置 embed、transformer block 和 head 分解的参数量。
- 目标分辨率下每次 forward pass 的 FLOPs。
- 跨 ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的比较表。

运行它。把参数量匹配到公布的数字。玩弄 patch size 和分辨率，感受 token 数量的代价。

## 交付它

本课产生 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch size、分辨率、hidden dim、depth），它给出 token 数量、参数量估计和 VRAM 估计并附带理由。每当你为一个 VLM 选择视觉主干时使用这个 skill——它防止"token 爆炸了，我的 LLM 上下文被占满"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch size 14 时的 patch-token 序列长度。与仅 CLS 表示相比如何？

2. 一帧 1080p（1920x1080）在 patch 14 下产生多少 token？在 30 FPS、5 分钟视频下，总共多少视觉 token？哪种代价节省最多：pooling、帧采样还是 token 合并？

3. 在纯 Python 中实现 patch token 上的 mean pooling。验证对 DINOv2 输出 196 个 token 的 mean-pool 与模型 `forward` 请求 pooled embedding 时返回的一致。

4. 阅读 "Vision Transformers Need Registers" 的 Section 3（arXiv:2309.16588）。用两句话描述 register 吸收了什么伪影，以及为什么这对下游密集预测很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定不同分辨率图像列表，产生一个打包序列和块对角 attention mask。到达 Lesson 12.06 时与之验证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch | "16x16 像素方块" | 输入图像的固定大小不重叠区域；变成一个 token |
| Patch embedding | "线性投影" | 共享学习矩阵（或 stride=P 的 Conv2d），把展平 patch 像素映射到 D 维向量 |
| CLS token | "Class token" | 前置的可学习向量，其最终 hidden state 代表整张图像；2026 年可选 |
| Register token | "Sink token" | 额外可学习 token，吸收 ViT 预训练中发展出的高范数 attention 伪影 |
| Position embedding | "位置信息" | 每个位置的向量或旋转，让序列有序；2D-RoPE 是现代默认 |
| Grid | "Patch grid" | 给定分辨率和 patch size 的 (H/P) x (W/P) 二维 patch 数组 |
| NaFlex | "Native flexible resolution" | SigLIP 2 特性：单个模型服务多种宽高比和分辨率，无需重训练 |
| Backbone | "Vision tower" | 预训练图像编码器，其 patch-token 输出在 VLM 中喂给 LLM |
| Pooling | "图像级摘要" | 把 patch token 变成一个向量的策略：CLS、mean、attention pool、register-based |
| Patch 14 vs 16 | "更细 vs 更粗网格" | Patch 14 每张图产生更多 token，OCR 保真度更好，更慢；patch 16 是经典默认 |

## 延伸阅读

- [Dosovitskiy 等人 — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT。
- [He 等人 — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab 等人 — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无标签。
- [Darcet 等人 — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 与伪影分析。
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认视觉塔。
- [Zhai 等人 — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验扩展定律。
