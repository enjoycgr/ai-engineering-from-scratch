# 音频 Transformer — Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个 ViT（Vision Transformer，视觉 Transformer），它接收 mel 频谱图并输出文本。

**类型:** 学习
**语言:** Python
**前置知识:** Phase 7 · 05 (完整 Transformer), Phase 7 · 08 (编码器-解码器), Phase 7 · 09 (ViT)
**时间:** ~45 分钟

## 问题背景

在 Whisper（OpenAI, Radford 等, 2022）出现之前，最先进的自动语音识别（ASR, Automatic Speech Recognition）是 wav2vec 2.0 和 HuBERT —— 自监督特征提取器加上一个微调后的输出头。质量高，但数据流水线昂贵，且对领域变化敏感。多语言语音识别需要为每个语系单独训练模型。

Whisper 下了三个赌注：

1. **用所有数据训练。** 从互联网上抓取了 68 万小时的弱标注音频，涵盖 97 种语言。没有干净的学术语料库，没有音素标签。
2. **单一多任务模型。** 一个解码器同时训练转录（transcription）、翻译（translation）、语音活动检测（voice activity detection）、语言识别（language ID）和时间戳（timestamping），通过任务 token（task token）来区分。
3. **标准编码器-解码器 Transformer。** 编码器消费 log-mel 频谱图（log-mel spectrogram），解码器自回归地生成文本 token。没有声码器（vocoder），没有 CTC（Connectionist Temporal Classification），没有 HMM（Hidden Markov Model）。

结果：Whisper large-v3 在口音、噪声和零干净标注数据的语言上都表现出色。到 2026 年，它已成为每一个开源语音助手和大多数商业语音助手的默认语音前端。

## 核心概念

![Whisper 流水线：音频 → mel → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 分帧

音频采样率为 16 kHz。裁剪/填充至 30 秒。计算 log-mel 频谱图：80 个 mel 频带（mel bins），10 ms 步长（stride）→ 约 3,000 帧 × 80 维特征。这就是 Whisper 看到的"输入图像"。

### 步骤 2 — 卷积主干（convolutional stem）

两个 Conv1D 层，卷积核大小为 3，步长为 2，将 3,000 帧缩减到 1,500。在不增加大量参数的情况下将序列长度减半。

### 步骤 3 — 编码器

一个 24 层（large 版本）的 Transformer 编码器，处理 1,500 个时间步。使用正弦位置编码（sinusoidal positional encoding）、自注意力（self-attention）、GELU 前馈网络（FFN, Feed-Forward Network）。输出 1,500 × 1,280 的隐状态。

### 步骤 4 — 解码器

一个 24 层的 Transformer 解码器。它自回归地从 BPE（Byte Pair Encoding）词表中生成 token，该词表是 GPT-2 词表的超集，并增加了一些音频专用的特殊 token。

### 步骤 5 — 任务 token

解码器提示（decoder prompt）以控制 token 开头，告诉模型该做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型就是按这个约定训练的。你通过前缀控制任务。这是 2026 年指令微调（instruction-tuning）的语音等价物。

### 步骤 6 — 输出

束搜索（beam search，宽度为 5）配合对数概率阈值（log-prob threshold）。当不存在 `<|notimestamps|>` token 时，每 0.02 秒音频预测一次时间戳。

### Whisper 模型尺寸

| 模型 | 参数量 | 层数 | d_model | 注意力头数 | VRAM (fp16) |
|------|--------|------|---------|-----------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4 层解码器) |

Large-v3-turbo（2024）将解码器从 32 层削减到 4 层。解码速度提升 8 倍，WER（Word Error Rate，词错误率）仅下降不到 1 个百分点。正是这个解码速度突破，使得 Whisper-turbo 成为 2026 年实时语音助手的默认选择。

### Whisper 不做什么

- 不做说话人分离（diarization，判断谁在说话）。需要配合 pyannote 使用。
- 原生不支持实时流式处理 —— 30 秒窗口是固定的。现代封装工具（`faster-whisper`、`WhisperX`）通过 VAD（Voice Activity Detection，语音活动检测）+ 重叠窗口来实现流式。
- 不处理超过 30 秒的长文本上下文，除非外部做分块。但在实践中效果很好，因为人类语音的转录很少需要长距离上下文。

### 2026 年音频模型格局

| 任务 | 模型 | 说明 |
|------|------|------|
| 英语 ASR | Whisper-turbo, Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 ms 延迟目标 |
| TTS（Text-to-Speech，文本转语音） | Piper, XTTS-v2, Kokoro | 编码器-解码器模式，但呈 Whisper 形态 |
| 音频 + 语言 | AudioLM, SeamlessM4T | 一个 Transformer 中同时处理文本 token 和音频 token |

## 动手实现

参见 `code/main.py`。我们不训练 Whisper —— 我们构建 log-mel 频谱图流水线 + 任务 token 提示格式化器。这些才是你在生产环境中实际会接触到的部分。

### 步骤 1：合成音频

生成一个 440 Hz 的正弦波，持续 1 秒，采样率 16 kHz。共 16,000 个采样点。

### 步骤 2：log-mel 频谱图（简化版）

完整的 mel 频谱图需要 FFT（Fast Fourier Transform，快速傅里叶变换）。我们做一个简化的分帧 + 每帧能量版本，展示流水线而不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧长 = 25 ms，步长 = 10 ms。与 Whisper 的窗口设置一致。每帧能量在教学中替代 mel 频带。

### 步骤 3：填充至 30 秒

Whisper 总是处理 30 秒的音频块。将频谱图填充（或裁剪）到 3,000 帧。

### 步骤 4：构建提示 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制界面。一个 4-token 前缀。

## 使用方式

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、兼容 OpenAI 的方式：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 一个模型覆盖多语言 ASR。
- 对噪声大、来源多样的音频进行鲁棒转录。
- 研究 / 原型 ASR —— 最快的起点。

**何时选择其他方案：**

- 边缘设备上的超低延迟流式处理 —— Moonshine 在同等质量下击败 Whisper。
- 需要 <200 ms 延迟的实时对话 AI —— 专用流式 ASR。
- 说话人分离 —— Whisper 不做这个；需要配合 pyannote。

## 交付

参见 `outputs/skill-asr-configurator.md`。该 skill 会为新的语音应用选择 ASR 模型、解码参数和预处理流水线。

## 练习题

1. **简单。** 运行 `code/main.py`。确认 16 kHz 采样率、10 ms 步长下，1 秒信号的帧数约为 100 帧。30 秒约为 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的 log-mel 频谱图。验证 80 个 mel 频带与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差范围内一致。
3. **困难。** 实现流式推理：将音频分块为 10 秒窗口，重叠 2 秒，对每个块运行 Whisper，合并转录结果。在 5 分钟播客样本上测量与单遍处理的词错误率（WER）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Mel spectrogram（Mel 频谱图） | "音频图像" | 二维表示：一个轴是频率频带，另一个轴是时间帧；每个单元格是取对数后的能量。 |
| Log-mel | "Whisper 看到的东西" | Mel 频谱图经过对数变换；近似人类对响度的感知。 |
| Frame（帧） | "一个时间切片" | 25 ms 的采样窗口；以 10 ms 步长重叠。 |
| Task token（任务 token） | "语音的提示前缀" | 解码器提示中的特殊 token，如 `<\|transcribe\|>` / `<\|translate\|>`。 |
| VAD（Voice Activity Detection，语音活动检测） | "找到语音" | 在 ASR 之前去除静音的门控；大幅降低成本。 |
| CTC（Connectionist Temporal Classification） | "连接时序分类" | 经典的无对齐训练 ASR 损失函数；Whisper 不使用它。 |
| Whisper-turbo | "小解码器，完整编码器" | large-v3 编码器 + 4 层解码器；解码速度提升 8 倍。 |
| Faster-whisper | "生产环境封装" | CTranslate2 重新实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py` 可在约 400 行内看到 Conv1D 主干 + 编码器 + 解码器的完整实现。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步骤 5–6 中描述的束搜索 + 任务 token 逻辑在此；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 先驱工作；在某些场景下仍是 SOTA 特征提取器。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产环境封装，比参考实现快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年面向边缘设备的 ASR，呈 Whisper 形态但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 标准微调教程，包括 mel 频谱图预处理器和 token-时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（编码器、解码器、交叉注意力、生成），与本课的架构图一致。
