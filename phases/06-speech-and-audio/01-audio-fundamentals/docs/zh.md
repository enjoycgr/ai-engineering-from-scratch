# 音频基础 —— 波形、采样、傅里叶变换

> 波形是原始信号，频谱图是信号的表示形式，Mel 特征是适合机器学习的形式。每个现代 ASR 和 TTS 流程都沿着这个阶梯前行，而第一级就是理解采样和傅里叶变换。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 1 · 06（向量与矩阵），Phase 1 · 14（概率分布）
**时间：** 约 45 分钟

## 问题

麦克风产生的是压力随时间变化的信号。你的神经网络消费的是张量。在它们之间，堆叠着一系列约定，一旦违反，就会产生隐蔽的 bug：模型训练正常但 WER（词错误率）翻倍，或 TTS 输出嘶嘶声，或声音克隆系统记住了麦克风而不是说话人。

语音系统中的每个 bug 都可追溯到以下三个问题之一：

1. 数据的采样率是多少，模型期望的采样率又是多少？
2. 信号是否存在混叠（aliasing）？
3. 你操作的是原始采样点还是频率表示？

把这三点弄对，Phase 6 的其余内容就可迎刃而解。弄错了，即使是 Whisper-Large-v4 也会输出垃圾。

## 概念

![波形、采样、DFT 和频率分箱可视化](../assets/audio-fundamentals.svg)

**波形（Waveform）。** 一维浮点数数组，取值范围 `[-1.0, 1.0]`。以采样点编号为索引。要转换为秒，除以采样率：`t = n / sr`。一段 10 秒、16 kHz 的音频是一个包含 160,000 个浮点数的数组。

**采样率（Sample rate, sr）。** 每秒的采样点数。2026 年的常见采样率：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话、传统 VOIP。Nyquist 频率为 4 kHz，会丢失辅音。ASR 中应避免。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都要求 16 kHz。 |
| 22.05 kHz | 旧版 TTS 声码器训练。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农定理（Nyquist-Shannon）。** 采样率为 `sr` 时，可以无歧义地表示最高到 `sr/2` 的频率。`sr/2` 边界称为 *Nyquist 频率（奈奎斯特频率）*。高于 Nyquist 的能量会发生 *混叠（aliasing）*——被折叠到更低的频率中——从而污染信号。降采样前必须始终使用低通滤波器。

**位深（Bit depth）。** 16-bit PCM（有符号 int16，范围 ±32,767）是通用交换格式。24-bit 用于音乐，32-bit 浮点数用于内部 DSP。`soundfile` 等库读取 int16，但输出 float32 数组，范围 `[-1, 1]`。

**傅里叶变换（Fourier Transform）。** 任何有限信号都可以表示为不同频率正弦波的叠加。离散傅里叶变换（DFT, Discrete Fourier Transform）对 `N` 个采样点计算 `N` 个复系数——每个频率分箱（bin）一个。`分箱 k（bin k）` 对应的频率为 `k · sr / N` Hz。幅度表示该频率的振幅，角度表示相位。

**快速傅里叶变换（FFT, Fast Fourier Transform）。** 当 `N` 为 2 的幂时，DFT 的 `O(N log N)` 算法。每个音频库底层都使用 FFT。1024 点 FFT 在 16 kHz 下给出 512 个可用频率分箱，覆盖 0–8 kHz，分辨率 15.6 Hz。

**分帧 + 加窗（Framing + window）。** 我们不对整段音频做 FFT。而是将其切成重叠的 *帧（frames）*（通常 25 ms，hop 为 10 ms），每帧乘以一个窗函数（Hann、Hamming）以消除边缘不连续，然后对每帧做 FFT。这就是短时傅里叶变换（STFT, Short-Time Fourier Transform）。Lesson 02 从这里继续展开。

## 动手实现

### 步骤 1：读取音频片段并绘制波形

`code/main.py` 仅使用标准库 `wave` 模块以保持零依赖。生产环境中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤 2：从零开始合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz 正弦波（标准音 A）在 16 kHz 下持续 1 秒是 16,000 个浮点数。使用 `wave.open(..., "wb")` 以 16-bit PCM 编码写入。

### 步骤 3：手动计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`——对于 `N=256` 验证正确性没问题，实际音频中完全不可用。真实代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：找到主导频率

幅度峰值索引 `k_star` 对应的频率为 `k_star * sr / N`。在 440 Hz 正弦波上运行应返回峰值位于分箱 `440 * N / sr`。

### 步骤 5：演示混叠

以 10 kHz 采样 7 kHz 正弦波（Nyquist = 5 kHz）。7 kHz 音调高于 Nyquist，折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都配备砖墙式低通滤波器的原因。

## 实际应用

2026 年你实际会部署的栈：

| 任务 | 库 | 原因 |
|------|---------|-----|
| 读/写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、稳定，返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / Mel | `torchaudio` 或 `librosa` | 支持 GPU；PyTorch 生态。 |
| 实时流式处理 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 文件检查 | `ffprobe` 或 `soxi` | CLI，快速，可报告 sr/声道/编解码器。 |

决策规则：**先匹配采样率，再匹配其他任何东西**。Whisper 期望 16 kHz 单声道 float32。传入 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾输出。

## 交付产物

保存为 `outputs/skill-audio-loader.md`。该 skill 帮助你检查音频输入是否符合下游模型的期望，并在不匹配时正确重采样。

## 练习

1. **简单。** 在 16 kHz 下合成 1 秒的 220 Hz + 440 Hz + 880 Hz 混合信号。运行 DFT。确认三个峰值位于预期的分箱。
2. **中等。** 以 48 kHz 录制一段 3 秒的你的声音 WAV。使用 `torchaudio.transforms.Resample`（带抗混叠）降采样到 16 kHz，再用朴素抽取（每三个采样点取一个）降采样到 16 kHz。对两者做 FFT。混叠出现在哪里？
3. **困难。** 仅使用 `math` 和步骤 3 的 DFT，从零开始构建 STFT。帧大小 400，hop 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅度。这就是 Lesson 02 的频谱图。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Sample rate | 每秒多少个采样点 | ADC 测量信号的频率，单位为 Hz。 |
| Nyquist | 能表示的最大频率 | `sr/2`；高于它的能量会混叠回低频率。 |
| Bit depth | 每个采样点的分辨率 | `int16` = 65,536 级；`float32` = `[-1, 1]` 范围内的 24-bit 精度。 |
| DFT | 序列的傅里叶变换 | `N` 个采样点 → `N` 个复频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，要求 `N` 为 2 的幂。 |
| Bin | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层机制 | 分帧 + 加窗后随时间做 FFT。 |
| Aliasing | 奇怪频率幽灵 | 高于 Nyquist 的能量镜像到更低的分箱。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) —— 采样定理背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) —— 免费、权威的 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) —— 带代码的实用入门教程。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) —— 参考：为什么真实世界的音频不是干净的正弦波。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) —— 10 分钟理清频率分箱的直觉。
