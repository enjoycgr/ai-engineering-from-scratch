# 视频生成 (Video Generation)

> 图像是一个二维张量 (2-D tensor)。视频是一个三维张量 (3-D tensor)。理论相同；计算量却大了 10–100 倍。OpenAI 的 Sora（2024 年 2 月）证明了这是可行的。到 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 都能从文本生成 1080p 的生产级视频；而开源权重模型栈（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）仅落后约 12 个月。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 分钟

## 问题所在 (The Problem)

一段 10 秒 1080p、24fps 的视频包含 240 帧，每帧 1920×1080×3 像素。每段剪辑的原始数据约 1.5 GB。在像素空间 (pixel-space) 上做扩散 (diffusion) 是不可行的。你需要：

1. **时空压缩 (Spatiotemporal compression)。** 一个 VAE（变分自编码器），它把视频（而非单帧）编码成一系列时空块 (spatial-temporal patches)。
2. **时间一致性 (Temporal coherence)。** 帧与帧之间需要在数秒内共享内容、光照和物体身份。网络必须对运动进行建模。
3. **计算预算 (Compute budget)。** 视频训练比同等模型尺寸下的图像训练昂贵 10–100 倍。
4. **条件控制 (Conditioning)。** 文本、图像（首帧）、音频，或另一段视频。大多数生产级模型同时接受这四种输入。

解决这一问题的架构是将 **Diffusion Transformer (DiT，扩散 Transformer)** 应用于时空块 (spatiotemporal patches)，并在海量（提示词、字幕、视频）数据集上训练。其扩散损失 (diffusion loss) 与第 06 课相同。

## 核心概念 (The Concept)

![视频扩散：块化、DiT、解码](../assets/video-generation.svg)

### 块化 (Patchify)

用 3D VAE（学习的时空压缩）对视频进行编码。隐变量 (latent) 的形状为 `[T_latent, H_latent, W_latent, C_latent]`。将其切分为大小为 `[t_p, h_p, w_p]` 的块 (patches)。对于 Sora 风格的模型，`t_p = 1`（每帧块化）或 `t_p = 2`（每两帧）。一段 10 秒 1080p 的视频压缩后约有 20,000–100,000 个块。

### 时空 DiT (Spatiotemporal DiT)

Transformer 处理展平后的块序列。每个块都有一个三维位置嵌入 (positional embedding)（时间 + y + x）。注意力 (Attention) 通常被分解 (factorized)：

- **空间注意力 (Spatial attention)**：在每一帧的块内运行。
- **时间注意力 (Temporal attention)**：在相同空间位置的不同帧之间运行。
- **完整 3D 注意力 (Full 3D attention)** 的计算量高出 16–100 倍；仅用于低分辨率或研究中。

### 文本条件控制 (Text conditioning)

通过交叉注意力 (cross-attention) 接入大型文本编码器（Sora 使用 T5-XXL，CogVideoX-5B 也使用 T5-XXL）。长提示词很重要——Sora 的训练集使用了 GPT 生成的密集重标注 (dense re-captions)，平均每段剪辑约 200 个 token。

### 训练 (Training)

标准的扩散损失（ε 预测或 v 预测）作用于时空隐变量 (spatiotemporal latents)。数据：网络视频 + ~1 亿条精选剪辑 + 合成文本字幕。计算成本：即使是很小的研究级运行也需要 10,000+ GPU 小时；Sora 级别的规模需要 100,000+。

## 2026 年生产级生态 (The 2026 production landscape)

| 模型 | 发布时间 | 最大时长 | 最大分辨率 | 开源权重？ | 亮点 |
|------|----------|----------|------------|-----------|------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | 否 | 首个在规模上展现世界模拟器 (world simulator) 特性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 生产级 Sora，推理速度提升 5 倍 |
| Veo 2 (Google) | 2024-12 | 8s | 4K | 否 | 2025 年最高画质 + 物理模拟 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音频支持与更强的镜头控制 |
| Kling 1.5 / 2.1 (快手) | 2024-2025 | 10s | 1080p | 否 | 2025 Q1 最佳人体运动 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 专业视频工具集成 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 角色一致性最强 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | 是 (2B, 5B) | 首个开源 5B 级视频模型 |
| HunyuanVideo (腾讯) | 2024-12 | 5s | 720p | 是 (13B) | 2024 年末开源 SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | 是 (10B) | 许可最宽松 |
| WAN 2.2 (阿里巴巴) | 2025-07 | 5s | 720p | 是 | 2025 年中开源最强模型 |

开源权重正在比图像领域更快地缩小差距：到 2026 年中，HunyuanVideo + WAN 2.2 的 LoRA 已经支撑了大多数开源工作流。

## 动手构建 (Build It)

`code/main.py` 模拟了时空 DiT 的核心思想：对一个小型合成视频进行块化 (patchify)，添加每个块的位置嵌入 (position embedding)，并用类 Transformer 的注意力对整段序列进行去噪。不使用 numpy；纯 Python。我们展示了即使在 1-D 情况下，当相邻帧的块共享同一个去噪器 (denoiser) 和位置嵌入时，时间一致性 (temporal coherence) 也会自然涌现。

### 步骤 1：对一个合成 1-D "视频" 进行块化

```python
def make_video(T_frames=8, rng=None):
    # "视频" 是一系列遵循平滑轨迹的一维数值
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 步骤 2：每帧的位置嵌入 (position embedding)

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 步骤 3：去噪器 (denoiser) 看到整个序列

我们不再独立地对每一帧去噪，而是让微型网络把所有帧的数值 + 它们的位置嵌入拼接起来，然后联合预测所有帧的噪声。

### 步骤 4：时间一致性测试

训练完成后，采样一段视频。测量帧间差值 (frame-to-frame delta)。如果模型学到了时间结构，这些差值会比独立采样每一帧时更小。

## 常见陷阱 (Pitfalls)

- **独立逐帧采样 = 闪烁 (flicker)。** 如果你在每一帧上单独运行图像扩散，输出会闪烁，因为每帧的噪声都是独立的。视频扩散通过注意力 (attention) 或共享噪声把帧耦合起来，从而修复这个问题。
- **朴素 3D 注意力 = 显存溢出 (OOM)。** 在 10 秒 1080p 隐变量上运行完整 3D 注意力需要数千亿次运算。将其分解为空间 + 时间注意力。
- **数据字幕比数据量更重要。** Sora 相比之前工作的主要升级，是使用了约 10 倍更详细的字幕进行训练（GPT-4 重标注的剪辑）。OpenAI 的技术报告明确指出了这一点。
- **首帧条件控制 (First-frame conditioning)。** 大多数生产级模型也接受一张图像作为首帧。这就是"图生视频" (image-to-video, I2V) 模式；训练包含这一变体。
- **物理漂移 (Physics drift)。** 长剪辑（>10 秒）会累积细微的不一致。滑动窗口生成 (sliding-window generation) + 关键帧锚定 (keyframe anchoring) 可以缓解。

## 拿来即用 (Use It)

| 使用场景 | 2026 年推荐 |
|----------|------------|
| 最高画质文本生视频，托管服务 | Veo 3 或 Sora |
| 镜头控制的电影级视频 | Runway Gen-3 + 运动笔刷 (motion brushes) |
| 跨剪辑的角色一致性 | Pika 2.0 或 Kling 2.1 |
| 开源权重，快速微调 (fine-tune) | WAN 2.2 + LoRA |
| 图生视频 (Image-to-video) | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音频驱动视频口型同步 | Veo 3（原生音频）或专用口型同步模型 |
| 视频编辑 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（静帧） |

2024 年至 2026 年间，同等画质下每秒视频的成本下降了 20 倍。

## 交付成果 (Ship It)

保存 `outputs/skill-video-brief.md`。该 skill 接收一个视频需求简报（时长、宽高比、风格、镜头计划、主体一致性、音频），并输出：模型 + 托管方案、提示词脚手架（镜头语言、主体描述、运动描述符）、种子 + 可复现性协议，以及帧级 QA 检查清单。

## 练习题 (Exercises)

1. **简单。** 在 `code/main.py` 中，对比 (a) 独立逐帧采样 和 (b) 联合序列采样 的帧间差值。报告差值的均值和方差。
2. **中等。** 添加首帧条件控制：将第 0 帧固定为给定值，然后采样其余部分。测量固定值如何传播。
3. **困难。** 使用 HuggingFace diffusers 在本地 GPU 上运行 CogVideoX-2B。对一段 720p 6 秒剪辑执行 20 步推理。剖析时空注意力以找出瓶颈。

## 关键术语 (Key Terms)

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Video VAE | "3-D VAE" | 将 `(T, H, W, C)` 压缩为时空隐变量的编码器。 |
| Patches | "The tokens" | 隐变量中固定大小的三维块；DiT 的输入。 |
| Factorized attention | "Spatial + temporal" | 先对空间做注意力，再对时间做注意力；跳过完整 3D 注意力。 |
| Image-to-video (I2V) | "Animate this photo" | 模型接收一张图像 + 文本，输出以该图像为首帧的视频。 |
| Keyframe conditioning | "Anchor frames" | 固定特定帧以控制视频的整体走向。 |
| Motion brush | "Directional hint" | 用户在图像上绘制运动向量的 UI 输入。 |
| Re-captioning | "Dense captions" | 使用 LLM 为训练剪辑重新标注详细的提示词。 |
| Flicker | "Temporal artifact" | 帧间不一致；通过耦合去噪 (coupled denoising) 修复。 |

## 生产级备注：视频隐变量是内存带宽问题 (Production note: video latents are a memory-bandwidth problem)

一段 10 秒 1080p 24fps 的剪辑，共有 240 帧 × 1920 × 1080 × 3 ≈ 1.5 GB 原始像素。经过 4× 视频 VAE 压缩（`2× 空间 × 2× 时间`）后，单次请求的隐变量约为 100 MB。以 batch size 1 运行 30 步的时空 DiT，每步需要在 HBM 中传输约 3 GB 数据——瓶颈是内存带宽 (memory bandwidth)，而非 FLOPs。

三个来自生产级推理文献的调参旋钮：

- **DiT 的 TP (Tensor Parallelism)。** 文本生视频模型通常 ≥10B 参数。在 4 张 H100 上使用 TP=4 是标准做法；405B 级模型使用 PP=2 × TP=2。每步延迟随 TP 大致线性下降，直到受 all-reduce 墙限制。
- **帧批处理 = 连续批处理 (continuous batching)。** 在生成时，视频在概念上是通过注意力 (attention) 关联起来的帧批次。连续批处理（在飞调度，in-flight scheduling）适用：如果模型架构允许滑动窗口生成，可以在返回第 `t-1` 帧的同时开始渲染第 `t+1` 帧。
- **剪辑级预填充缓存 (Clip-level prefill cache)。** 对于图生视频，首帧条件控制类似于 LLM 的 prompt prefill：只计算一次，然后在时间解码器的多次传递中复用。这实际上是一种视频版的 KV-cache。

## 延伸阅读 (Further Reading)

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技术报告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中开源 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 开创性视频扩散论文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的前身。
