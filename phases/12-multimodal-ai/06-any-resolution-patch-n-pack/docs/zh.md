# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。一张收据是 9:16，一张图表是 16:9，一张医学扫描可能是 4096x4096，一张手机截屏是 9:19.5。2024 年前 VLM 的答案——把所有东西 resize 到固定正方形——丢掉了让 OCR、文档理解和高分辨率场景解析工作的信号。NaViT (Google, 2023) 表明你可以把可变分辨率 patch 打包进单个 transformer batch，使用块对角 masking。Qwen2-VL 的 M-RoPE (2024) 完全丢弃了绝对位置表。LLaVA-NeXT 的 AnyRes 把高分辨率图像铺成基础 + 子图像。SigLIP 2 的 NaFlex 变体 (2025) 现在是想要单个 checkpoint 服务每种宽高比的开源 VLM 的默认编码器。本课从头到尾实现 patch-n'-pack。

**类型：** Build
**语言：** Python (stdlib, patch packer + block-diagonal mask)
**前置知识：** Phase 12 · 01 (ViT patch), Phase 12 · 05 (LLaVA)
**时间：** ~120 分钟

## 学习目标

- 把一批可变分辨率图像的 patch 打包到一个序列中，并构建块对角 attention mask。
- 在 AnyRes tiling (LLaVA-NeXT)、NaFlex (SigLIP 2) 和 M-RoPE (Qwen2-VL) 之间为给定任务做选择。
- 在不 resize 的情况下为 OCR、图表和摄影计算 token 预算。
- 说出正方形 resize 的三种失败模式：压扁的文本、裁剪的内容、padding 上浪费的 token。

## 问题

Transformers 期望序列。Batch 是相同长度序列的堆栈。如果你的图像都是 224x224，你每次都得到 196 个 patch token，不需要 padding，任务完成。在 224 上训练，在 224 上推理，再也不用想分辨率。

世界不配合。文档是 portrait (8.5x11 英寸，约 2:3)。图表截屏是 landscape (16:9)。收据又高又细 (1:3)。医学影像发货为 2048x2048 或更大。手机设备截屏是 1170x2532 (0.46:1)。

三种 2024 年前的选项及为什么每个都失败：

1. Resize 到固定正方形 (224x224 或 336x336)。挤压扭曲了文本和面孔。下采样破坏了图表标签和 OCR 内容。标准做法直到 LLaVA-1.5。
2. 裁剪到固定宽高比。你丢掉了图像的大部分，选择裁剪位置是它自己的视觉问题。
3. Padding 到最长边。修复扭曲但 portrait 图像上浪费 50%+ 的 token 在 padding 上。所有这些 pad token 的二次 attention 成本。

2024-2025 年的答案：让 transformer 以图像原生分辨率吃 patch，并找出一个 batch 中如何把异构序列打包进一个序列而不浪费计算。

## 概念

### NaViT 和 patch-n'-pack

NaViT (Dehghani 等人, 2023) 是表明这在规模上有效的论文。想法是机械的：

1. 对 batch 中的每张图像，在给定 patch size（如 14）下计算其原生 patch 网格。
2. 把每张图像的 patch 展平成自己的可变长度序列。
3. 把所有图像的 patch 拼接进一个长序列。
4. 构建块对角 attention mask，使图像 A 的 patch 只关注图像 A 内部。
5. 携带每 patch 位置信息（2D RoPE 或分数位置 embedding）。

三幅图像的 batch：336x336 (576 token)、224x224 (256 token)、448x336 (768 token) 变成一个 1600-token 序列，带 1600x1600 块对角 mask。无 padding。无浪费计算。Transformer 处理任意宽高比。

NaViT 还引入了训练中的分数 patch 随机丢弃——在 batch 中随机丢弃 50% 的 patch——既正则化又加速训练。SigLIP 2 继承了这一点。

### AnyRes (LLaVA-NeXT)

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一个高分辨率图像和一个固定编码器（336 的 CLIP 或 SigLIP），把图像铺成瓦片：

1. 从预定义集合中挑选一个网格布局——(1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等——最匹配图像宽高比。
2. 把完整图像铺进网格；每个瓦片变成 336x336 裁剪。
3. 还生成一个缩略图：整幅图像 resize 到 336x336 作为全局上下文 token。
4. 通过冻结的 336-编码器编码每个瓦片。拼接瓦片 token + 缩略图 token。

对于 672x672 图像在 2x2 网格加缩略图：4 * 576 + 576 = 2880 视觉 token。昂贵但有效——LLM 同时看到局部细节和全局上下文。

AnyRes 是编码器冻结且只支持一种分辨率时的首选路线。对大图像爆炸 token 数（1344x1344 图像在 4x4 网格上是 9216 + 576 ≈ 9800 token，填满大部分 8k LLM 上下文）。

### M-RoPE (Qwen2-VL)

Qwen2-VL 引入了 Multimodal Rotary Position Embedding。取代 NaViT 的分数位置或 AnyRes 的瓦片+缩略图，每个 patch 携带 3D 位置（时间、高度、宽度）。Query/key 旋转处理任意的 H、W 和时序长度。

M-RoPE 原生发货动态分辨率无需重训练。推理时你喂任意 HxW 图像，patch embedder 产生 H/14 x W/14 个 token，每个 token 得到其 (t=0, r=行, c=列) 位置，RoPE 用正确频率旋转 attention，完成。Qwen2.5-VL 和 Qwen3-VL 延续这个。InternVL3 的 V2PE 是相同想法，带每模态可变编码。

与 AnyRes 不同，M-RoPE 在原生分辨率下是 O(H x W / P^2) token——无乘法瓦片开销。与 NaViT 不同，它仍然期望每次 forward 单张图像。跨分辨率 batching 仍然需要在上面做 patch-n'-pack。

### NaFlex (SigLIP 2)

NaFlex 是 SigLIP 2 checkpoint 的原生-flex 模式。单个模型在推理时服务多种序列长度（256、729、1024 token）。内部它在训练中使用 NaViT 风格 patch-n'-pack 和每 patch 绝对分数位置。卖点：一个 checkpoint，基于任务在推理时选择你的 token 预算。

语义任务（分类、检索）用 256 token。OCR 或图表理解用 1024 token。无需重训练。

### 打包 mask

块对角 mask 是大多数实现绊倒的地方。对覆盖图像 `i=0..B-1` 且长度 `n_i` 的打包序列总长度 `N_total`，形状为 `(N_total, N_total)` 的 mask `M` 在两个索引落在同一块时为 1，否则为 0。你可以从累积长度列表构建：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff 存在 b 使得 offsets[b] <= i < offsets[b+1] 且 offsets[b] <= j < offsets[b+1]
```

这在 PyTorch 中用 `torch.block_diag` 或显式 gather 是一行。FlashAttention 的变长路径 (`cu_seqlens`) 完全跳过 mask，用累积长度张量直接在序列内 attention——比典型 batch 的密集 mask 快约 10 倍。

### Token 预算

按任务选策略：

- OCR / 文档：1024-4096 token。SigLIP 2 NaFlex 在 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：729-1024 token 在 384-448 原生。Qwen2.5-VL 动态分辨率带最大像素上限。
- 自然照片：256-576 token 足够。下游 LLM 看到足够。只在内容密度高的地方为 token 付费。
- 视频：空间 pooling 后每帧 64-128 token，2-8 FPS。Lesson 12.17 覆盖这个。

2026 年生产规则：选一个每任务最大像素上限，原生宽高比编码到该上限，打包 batch，跳过 padding。Qwen2.5-VL 为精确这个旋钮暴露 `min_pixels` 和 `max_pixels`。

## 使用它

`code/main.py` 为一批带整数像素坐标的异构分辨率图像实现 patch-n'-pack。它：

- 取 (H, W) 图像尺寸列表。
- 在 patch size 14 下计算每张图像的 patch 序列长度。
- 把它们打包进总长度 `sum(n_i)` 的单个序列。
- 构建块对角 attention mask（密集，为清晰起见）。
- 对比打包成本与正方形 resize 和 AnyRes tiling。
- 为混合 batch（收据、图表、截屏、照片）打印 token 预算表。

运行它。输出的数字是每个 2026 年开源 VLM 使用 patch-n'-pack 的原因。

## 交付它

本课产生 `outputs/skill-resolution-budget-planner.md`。给定一个混合宽高比工作负载（OCR、图表、照片、视频帧）和总 token 预算，它选择正确策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并发出每请求配置。当你为产品调整 VLM 大小时使用这个 skill——它防止静默的 10x token 爆炸杀死延迟预算。

## 练习

1. 一张收据是 600x1500 (1:2.5)。在 patch size 14 下，原生分辨率多少 token？正方形 resize 到 336 后多少？实践中哪个丢失更多 OCR 准确率？

2. 为四个长度 256、576、729、1024 的图像 batch 构建块对角 mask。验证 attention 矩阵是 2585x2585 且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零项。

3. 对于 1792x896 图像在 patch 14 下，对比：(a) 正方形 resize 到 336 然后编码，(b) AnyRes 2x1 + 缩略图，(c) M-RoPE 原生。哪个用最少 token？哪个保留最多细节？

4. 实现分数 patch 丢弃：给定打包序列，均匀随机丢弃 50% token，并相应更新块对角 mask。测量 mask 稀疏度变化。

5. 阅读 Qwen2-VL 论文 Section 3.2（arXiv:2409.12191）。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么以及为什么两个边界都重要。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Patch-n'-pack | "NaViT-style packing" | 把不同图像的可变长度 patch 序列拼接进一个 batch 维度 |
| Block-diagonal mask | "Packing mask" | 限制每张图像的 patch 只关注自身的 attention mask，不关注打包邻居 |
| AnyRes | "LLaVA-NeXT tiling" | 把高分辨率图像分成固定大小瓦片网格加一个全局缩略图；用固定编码器编码每个瓦片 |
| NaFlex | "SigLIP 2 native-flex" | 单个 SigLIP 2 checkpoint 在推理时服务 256/729/1024 token 预算，无需重训练 |
| M-RoPE | "Multimodal RoPE" | 3D 旋转位置编码（时间、行、列），无需位置表即可处理任意 H、W、T |
| cu_seqlens | "FlashAttention packing" | FlashAttention 变长路径使用的累积长度张量，替代密集块对角 mask |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL 每请求旋钮，限制非常小或非常大输入上的总像素数 |
| Visual token budget | "每张图像多少 token" | 每张图像发出的 patch token 粗略计数；设置 LLM prompt 预算和 attention 成本 |

## 延伸阅读

- [Dehghani 等人 — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang 等人 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon 等人 — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen 等人 — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
