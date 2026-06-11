# 音频生成 (Audio Generation)

> 音频是采样率为 16–48 kHz 的一维信号。一段 5 秒的剪辑包含 8–24 万个样本。没有任何 transformer 能直接在这样的序列长度上做注意力 (attention)。到 2026 年，每一个生产级音频模型的解决方案都一样：神经编解码器 (neural codec，如 EnCodec、SoundStream、DAC) 将音频压缩为 50–75 Hz 的离散 token，然后由 transformer 或扩散 (diffusion) 模型生成这些 token。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 分钟

## 问题所在 (The Problem)

三类音频生成任务：

1. **文本转语音 (Text-to-speech, TTS)。** 给定文本，生成语音。清晰语音是窄带的且具有强音素结构——由基于 token 的 transformer 很好地解决。VALL-E (Microsoft)、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成 (Music generation)。** 给定提示词（文本、旋律、和弦进行、风格），生成音乐。分布宽得多。MusicGen (Meta)、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音频效果 / 音效设计 (Audio effects / sound design)。** 给定提示词，生成环境声或拟音 (Foley)。AudioGen、AudioLDM 2、Stable Audio Open。

这三者都运行在同一套底层架构上：神经音频编解码器 (neural audio codec) + token-自回归 (token-AR) 或扩散 (diffusion) 生成器。

## 核心概念 (The Concept)

![音频生成：编解码器 token + transformer 或扩散](../assets/audio-generation.svg)

### 神经音频编解码器 (Neural audio codecs)

EnCodec (Meta, 2022)、SoundStream (Google, 2021)、Descript Audio Codec (DAC, 2023)。一个卷积编码器 (convolutional encoder) 将波形 (waveform) 压缩为每时间步的向量；残差向量量化 (Residual Vector Quantization, RVQ) 将每个向量转化为 K 个码本 (codebook) 索引的级联。解码器 (decoder) 将其还原。使用 8 个 RVQ 码本在 75 Hz 下对 24 kHz 音频进行 2 kbps 压缩 = 每秒 600 个 token。

```
波形 (waveform, 16000 样本/秒)
    └─ 编码器卷积 (encoder conv) ─┐
                     ├─ RVQ layer 1 → 75 Hz 的索引
                     ├─ RVQ layer 2 → 75 Hz 的索引
                     ├─ ...
                     └─ RVQ layer 8
```

### 两种生成范式 (Two generative paradigms on top)

**Token-自回归 (Token-autoregressive)。** 将 RVQ token 展平为序列，运行仅解码器的 transformer (decoder-only transformer)。MusicGen 使用 "延迟并行 (delayed parallel)" 方式，以每流偏移的方式并行发出 K 个码本流。VALL-E 从文本提示词 + 3 秒语音样本生成语音 token。

**隐变量扩散 (Latent diffusion)。** 将编解码器 token 打包为连续隐变量 (continuous latents)，或用分类扩散 (categorical diffusion) 对其进行建模。Stable Audio 2.5 在连续音频隐变量上使用流匹配 (flow matching)。AudioLDM 2 使用文本到梅尔频谱 (text-to-mel) 再到音频的扩散。

2024–2026 年的趋势：流匹配 (flow matching) 正在音乐领域取胜（推理更快、样本更干净），而 token-自回归 (token-AR) 仍然主导语音领域，因为它天然是因果的 (causal) 且适合流式传输 (streaming)。

## 生产级生态 (Production landscape)

| 系统 | 任务 | 骨干网络 | 延迟 |
|------|------|----------|------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 (neural vocoder) | ~300ms 首 token |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态自回归 (end-to-end multimodal AR) | ~200ms |
| NaturalSpeech 3 | TTS | 隐变量流匹配 (Latent flow matching) | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 (SFX) | DiT + 音频隐变量上的流匹配 | ~10s 生成 1 分钟剪辑 |
| Suno v4 | 完整歌曲 | 未公开；疑似 token-AR | ~30s 每首歌 |
| Udio v1.5 | 完整歌曲 | 未公开 | ~30s 每首歌 |
| MusicGen 3.3B | 音乐 | EnCodec 32kHz 上的 token-AR | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 (Flow matching) | ~5s 生成 5s 剪辑 |
| Riffusion v2 | 音乐 | 频谱图扩散 (Spectrogram diffusion) | ~10s |

## 动手构建 (Build It)

`code/main.py` 模拟了核心思想：在一个由两种不同"风格"生成的合成"音频 token" 序列上训练一个微型 next-token transformer（风格 A 是交替的低高 token，风格 B 是单调递增 ramp）。以风格为条件进行训练和采样。

### 步骤 1：合成音频 token (synthetic audio tokens)

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "类语音"：交替
        return [i % vocab_size for i in range(length)]
    # "类音乐"：递增 ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步骤 2：训练一个微型 token 预测器

一个以风格为条件的二元组 (bigram) 风格预测器。重点在于这个模式：编解码器 token (codec tokens) → 交叉熵 (cross-entropy) 训练 → 自回归 (autoregressive) 采样。

### 步骤 3：条件采样 (sample conditionally)

给定风格 token 和一个起始 token，从预测分布中采样下一个 token。继续生成 20–40 个 token。

## 常见陷阱 (Pitfalls)

- **编解码器 (codec) 质量决定了输出质量上限。** 如果编解码器无法忠实还原某种声音，生成器再好也无济于事。DAC 是目前开源最佳。
- **RVQ 误差累积 (RVQ error accumulation)。** 每个 RVQ 层建模前一层的残差 (residual)。第 1 层的误差会传播。对高层使用 temperature 0 采样有助于缓解。
- **音乐结构 (Musical structure)。** 30 秒的 token 序列在 75 Hz 下超过 2 万个 token。对 transformer 来说很难。MusicGen 使用滑动窗口 (sliding window) + 提示词续写 (prompt continuation)；Stable Audio 使用更短的剪辑 + 交叉淡入淡出 (crossfading)。
- **边界处的伪影 (Artifacts at boundaries)。** 在生成剪辑之间做交叉淡入淡出需要仔细的叠加相加 (overlap-add)。
- **对干净数据的渴求 (Clean-data appetite)。** 音乐生成器需要数万小时的授权音乐。Suno / Udio 与 RIAA 的诉讼（2024 年）将这一问题推向了公众视野。
- **语音克隆伦理 (Voice cloning ethics)。** 一段 3 秒样本 + 文本提示词就足以让 VALL-E / XTTS / ElevenLabs 克隆一个声音。每个生产级模型都需要滥用检测 + 退出名单 (opt-out lists)。

## 拿来即用 (Use It)

| 任务 | 2026 年推荐栈 |
|------|--------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 语音克隆（已验证同意） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，快速生成 | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 (SFX) | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音智能体 | GPT-4o realtime 或 Gemini Live |
| 开源权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付成果 (Ship It)

保存 `outputs/skill-audio-brief.md`。该 skill 接收一个音频需求简报（任务、时长、风格、声音、许可证），并输出：模型 + 托管方案、提示词格式（风格标签、风格描述符、结构标记）、编解码器 (codec) + 生成器 + 声码器 (vocoder) 链路、种子协议，以及评估计划（TTS 用 MOS / CLAP 分数 / CER，SFX 用用户 A/B）。

## 练习题 (Exercises)

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成序列是否符合该风格的模式。
2. **中等。** 添加延迟并行解码 (delayed parallel decoding)：模拟 2 条必须保持 1 步偏移的 token 流。训练一个联合预测器。
3. **困难。** 使用 HuggingFace transformers 在本地运行 MusicGen-small。用三个不同提示词生成 10 秒剪辑；进行风格遵循度的 A/B 对比。

## 关键术语 (Key Terms)

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Codec | "Neural compression" | 音频的编码器 / 解码器；典型输出为 50–75 Hz 的 token。 |
| RVQ | "Residual VQ" | K 个量化器 (quantizers) 的级联；每个建模前一层的残差。 |
| Token | "One codec symbol" | 码本 (codebook) 中的离散索引；典型为 1024 或 2048。 |
| Delayed parallel | "Offset codebooks" | 以交错偏移的方式发出 K 条 token 流，以缩短序列长度。 |
| Flow matching | "The 2024 win for audio" | 扩散 (diffusion) 的直线路径替代方案；采样更快。 |
| Voice prompt | "3-second sample" | 说话人嵌入 (speaker embedding) 或 token 前缀，用于引导克隆声音。 |
| Mel spectrogram | "The visual" | 对数幅度感知频谱图 (log-magnitude perceptual spectrogram)；许多 TTS 系统使用它。 |
| Vocoder | "Mel to wave" | 将梅尔频谱 (mel spectrograms) 还原为音频的神经组件。 |

## 生产级备注：音频是一个流式问题 (Production note: audio is a streaming problem)

音频是唯一一种用户期望*边生成边播放*的输出模态，而非全部生成完再播放。在生产级术语中，这意味着 TPOT（每输出 token 时间）很重要，因为用户的聆听速度就是目标吞吐——而不是阅读速度。对于 16kHz 音频使用 ~75 token/秒 的 EnCodec，服务器必须每秒为每位用户生成 ≥75 个 token 才能保持播放流畅。

两个架构层面的后果：

- **流匹配 (flow matching) 音频模型无法天然流式传输。** Stable Audio 2.5 和 AudioCraft 2 以单次前向传播渲染固定长度的剪辑。要实现流式，需要把剪辑分块并在边界处重叠——类似于滑动窗口扩散 (sliding-window diffusion)——相比编解码器自回归 (codec AR) 模型会增加 100–300ms 的延迟开销。

如果产品是"实时语音聊天"或"实时音乐续写"，选择编解码器自回归 (codec AR) 路径。如果是"提交后渲染一段 30 秒剪辑"，流匹配 (flow matching) 在质量和总延迟上胜出。

## 延伸阅读 (Further Reading)

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 编解码器 (codec) 标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 首个广泛使用的神经音频编解码器 (neural audio codec)。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025 年基于流匹配 (flow matching) 的文本到音乐生成。
