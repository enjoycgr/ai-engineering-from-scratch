# Transfusion：一个 Transformer 中的自回归文本 + Diffusion 图像

> Chameleon 和 Emu3 把所有赌注押在离散 token 上。它们有效，但量化瓶颈可见——图像质量在连续空间 diffusion 模型之下达到平台期。Transfusion (Meta, Zhou 等人, 2024 年 8 月) 采取相反赌注：保持图像连续，完全丢弃 VQ-VAE，用一个 transformer 训练两个 loss。文本 token 获得 next-token-prediction。图像 patch 获得 flow-matching / diffusion loss。两个目标优化相同权重。Stable Diffusion 3 (MMDiT) 的底层架构是近亲。本课阅读 Transfusion 论点，构建玩具两 loss 训练器，并追踪让一个 transformer 做两份工作的 attention mask。

**类型：** Build
**语言：** Python (stdlib, MNIST 规模玩具上的两 loss 训练器)
**前置知识：** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**时间：** ~180 分钟

## 学习目标

- 连接一个运行两个 loss（文本 token 上的 NTP，图像 patch 上的 diffusion MSE）的 transformer 在一个主干上。
- 解释为什么图像 patch 间的双向 attention 加文本 token 上的因果 attention 是正确的 mask 选择。
- 在计算、质量和代码复杂度上对比 Transfusion 风格（连续图像、diffusion loss）与 Chameleon 风格（离散图像、NTP）。
- 说出 MMDiT 的贡献：每 block 的模态特定权重，残差流中的联合 attention。

## 问题

离散 vs 连续图像 token 的争论比 LLM 更古老。连续表示（原始像素、VAE latent）保留细节。离散 token（VQ 索引）适合 transformer 原生词汇但在量化步骤丢失细节。

Chameleon / Emu3 走向离散：一个 loss，一个架构，但图像保真度受 tokenizer 质量限制。

Diffusion 模型走向连续：卓越的图像质量，但与 LLM 分离的模型、复杂的 noise-schedule 工程、与文本生成无干净整合。

Transfusion 问道：我们能两者兼得吗？保持图像连续，仍然训练一个模型，将两个 loss 缝入一个梯度步骤。

## 概念

### 两 loss 架构

单个 decoder-only transformer 处理包含以下内容的序列：

- 文本 token（离散，来自 BPE 词汇）。
- 图像 patch（连续，16x16 像素块通过线性 embedding 投影到 hidden dim——与 ViT 编码器输入相同）。
- `<image>` 和 `</image>` 标签标记连续 patch 的位置。

Forward pass 运行一次。Loss 按 token 挑选两个 head 之一：

- 文本 token：vocab-logits head 上的标准 cross-entropy。
- 图像 patch：连续 patch 上的 diffusion loss——预测加到每个 patch 的 noise。

梯度流过共享 transformer body。两个 loss 同时改善共享权重。

### Attention mask：因果文本 + 双向图像

文本 token 必须是因果的——不能让文本 token 关注未来文本，否则 teacher forcing 崩溃。图像 patch 然而代表一个快照；它们应该在同一块内双向关注彼此。

Mask：

```
M[i, j] = 1 if:
  (i 是文本 and j 是文本 and j <= i)   # 文本因果
  OR (i 是图像 and j 是图像 and same_image_block(i, j))   # 图像块内双向
  OR (i 是文本 and j 是图像 and j < i_image_end)   # 文本关注前面图像
  OR (i 是图像 and j 是文本 and j < i_image_start)   # 图像关注前面文本
```

在训练和推理中实现为块三角 mask。

### Transformer 内的 Diffusion loss

Diffusion loss 是标准的：给图像 patch 加 noise，让模型预测 noise（或等价地，干净 patch）。Transfusion 的版本使用 flow matching——从 noisy 到 clean 预测速度场。

训练期间：
1. 对每个图像 patch x0，采样随机 timestep t。
2. 采样 noise ε，计算 xt = (1-t) * x0 + t * ε（flow matching 的线性插值）。
3. Transformer 预测 v_theta(xt, t)；loss = MSE(v_theta(xt, t), ε - x0)。
4. 与同一序列的文本 NTP loss 一起反向传播。

推理时，生成是：
- 文本 token：标准自回归采样。
- 图像 patch：条件于前面文本 token 的 diffusion 采样循环（典型 10-30 步）。

### MMDiT：Stable Diffusion 3 的变体

Stable Diffusion 3 (Esser 等人, 2024 年 3 月) 与 Transfusion 大约同时发货 MMDiT (Multimodal Diffusion Transformer)。架构是兄弟姐妹。

MMDiT 的关键差异：

- 每 block 模态特定权重。每个 transformer block 对文本 token vs 图像 patch 有独立的 Q、K、V 和 MLP 权重。Attention 是联合的（跨模态）；其他都是模态特定的。
- Rectified flow 训练。一种特定的 flow-matching 变体，已知采样和比 DDPM 更简单的数学。
- 规模。MMDiT 是 SD3 的主干（2B 和 8B 参数变体）。Transfusion 论文扩展到 7B。

两者收敛于相同核心思想：一个 transformer 在文本上运行 NTP，在连续图像表示上运行 diffusion。

### 为什么这击败 Chameleon 风格

连续-diffusion 与离散-NTP 在图像生成上的质量差距是可测量的。Transfusion 论文报告：

- 7B 参数下，在 FID 上击败相同规模 Chameleon 风格模型 3-5 分。
- 无需 tokenizer 训练——图像编码器更简单（线性投影到 hidden，与 ViT 输入层相同）。
- 推理可以并行化图像 patch 去噪，不像自回归图像 token。

缺点：Transfusion 是双 loss 模型，使训练动态更棘手。Loss 权重需要调优。NTP 和 diffusion 之间的 schedule 不匹配可导致一个 head 主导。

### 下游是什么

Janus-Pro (Lesson 12.15) 通过为理解和生成解耦视觉编码器——理解用 SigLIP，生成用 VQ——同时共享 transformer body，细化了 Transfusion 的想法。Show-o (Lesson 12.14) 用离散-diffusion（masked prediction）交换 diffusion。统一生成家族在 Transfusion 后快速分支。

2026 年发出图像的生产 VLM——Gemini 3 Pro、GPT-5、Claude Opus 4.7 的图像生成路径——几乎确定使用这个家族的某种后代。细节是专有的。

## 使用它

`code/main.py` 在微型 MNIST 类问题上构建玩具 Transfusion：

- 文本标题是描述数字（0-9）的短整数序列。
- 图像是 4x4 字节网格。
- 一对共享权重线性投影作为 transformer 替身；文本上 NTP loss，noisy patch 上 MSE loss。
- 训练循环交替两个 loss，attention mask 显式。
- 生成在一次 forward pass 中产生文本标题和 4x4 图像。

Transformer 是玩具。两 loss 管道、attention mask 构建和推理循环是真正的产物。

## 交付它

本课产生 `outputs/skill-two-loss-trainer-designer.md`。给定新多模态训练任务（文本 + 图像、文本 + 音频、文本 + 视频），它设计两 loss schedule（loss 权重、mask 形状、共享 vs 模态特定 block）并标记实现风险。

## 练习

1. Transfusion 风格模型训练 70% 文本 token 和 30% 图像 patch。图像 diffusion loss 在幅度上约是文本 NTP loss 的 10 倍。什么 loss 权重平衡它们？

2. 为序列 `[T, T, <image>, P, P, P, P, </image>, T]` 实现块三角 mask。标记每项 0 或 1。

3. MMDiT 有模态特定 QKV 权重。这 vs Transfusion 完全共享 transformer 增加多少参数开销？7B 参数下，值得吗？

4. 生成：给定文本 prompt，模型为 50 token 运行 NTP，然后命中 `<image>`，然后在 20 去噪步上为 256 patch 运行 diffusion。总共多少次 forward pass？

5. 阅读 SD3 论文 Section 3。描述 rectified flow 及为什么它在更少推理步上收敛于 DDPM。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Two-loss training | "NTP + diffusion" | 单个 transformer 在同一步梯度中同时优化文本 token 上的 cross-entropy 和连续图像 patch 上的 MSE |
| Flow matching | "Rectified flow" | Diffusion 变体，预测从 noise 到 clean data 的速度场；比 DDPM 数学更简单 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3 的架构：联合 attention，模态特定 MLP 和 norm |
| Block-triangular mask | "Causal text + bidirectional image" | 跨文本因果但在图像区域内双向的 attention mask |
| Continuous image representation | "No VQ" | 图像 patch 作为实值向量，而非整数 codebook 索引 |
| Velocity prediction | "v-parameterization" | 网络输出是 noise 与 data 之间的速度场，而非 noise 本身 |

## 延伸阅读

- [Zhou 等人 — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser 等人 — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao 等人 — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
