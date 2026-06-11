# 频谱图、Mel 刻度与音频特征

> 神经网络不能很好地消费原始波形。它们消费频谱图。它们消费 mel 频谱图的效果更好。2026 年的每个 ASR、TTS 和音频分类器，成败都取决于这一个预处理选择。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 01（音频基础）
**时间：** 约 45 分钟

## 问题

取一段 10 秒、16 kHz 的音频片段。那是 160,000 个浮点数，都在 `[-1, 1]` 范围内，与标签"狗叫"或"单词 cat"几乎完全不相关。原始波形包含信息，但模型无法轻易提取。两个相同的音素相隔 100 毫秒发音，其原始采样点完全不同。

频谱图解决了这个问题。它在人耳感知忽略的地方压缩时间细节（微秒级抖动），在感知关注的地方保留结构（哪些频率有能量，在约 10–25 ms 的时间窗口内）。

Mel 频谱图更进一步。人类对音高的感知是对数性的：100 Hz 与 200 Hz 的"距离感"和 1000 Hz 与 2000 Hz 相同。mel 刻度将频率轴扭曲以匹配这种感知。mel 刻度频谱图是 2010 年到 2026 年间语音机器学习中最重要、最普遍的特征。

## 概念

![波形到 STFT 到 mel 频谱图到 MFCC 的阶梯](../assets/mel-features.svg)

**STFT（短时傅里叶变换，Short-Time Fourier Transform）。** 将波形切成重叠的帧（典型值：25 ms 窗口，10 ms hop，在 16 kHz 下对应 400 个采样点 / 160 个采样点）。每帧乘以一个窗函数（Hann 是默认值；Hamming 略有不同的权衡）。每帧做 FFT。将幅度谱堆叠成形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的频谱图。

**对数幅度（Log-magnitude）。** 原始幅度跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 来压缩动态范围。每个生产流程都使用对数幅度，而非原始幅度。

**Mel 刻度（Mel scale）。** 频率 `f`（Hz）映射到 mel `m` 的公式为 `m = 2595 * log10(1 + f / 700)`。该映射在 1 kHz 以下大致线性，在以上大致对数。80 个 mel 分箱覆盖 0–8 kHz 是 ASR 的标准输入。

**Mel 滤波器组（Mel filterbank）。** 一组在 mel 刻度上等距排列的三角滤波器。每个滤波器是相邻 FFT 分箱的加权和。将 STFT 幅度乘以滤波器组矩阵，一次矩阵乘法即可得到 mel 频谱图。

**对数 mel 频谱图（Log-mel spectrogram）。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年通用的音频前端。

**MFCC。** 取对数 mel 频谱图，应用 DCT（type II），保留前 13 个系数。对特征去相关并进一步压缩。在约 2015 年之前是主流特征，当时 CNN/Transformer 直接在原始对数 mel 上取得了更好的效果。仍在说话人识别中使用（x-vectors、ECAPA）。

**分辨率权衡（Resolution trade）。** 更大的 FFT = 更好的频率分辨率但更差的时间分辨率。25 ms / 10 ms 是音频 ML 的默认值；50 ms / 12.5 ms 用于音乐；5 ms / 2 ms 用于瞬态检测（鼓点击音、爆破音）。

## 动手实现

### 步骤 1：将波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒、16 kHz 的音频，使用 `frame_len=400, hop=160` 产生 998 帧。

### 步骤 2：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 前逐元素相乘。消除在非零端点截断导致的频谱泄漏。

### 步骤 3：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（基于 FFT，向量化）。这里的循环是教学用的；在 `code/main.py` 中可以在短片段上运行。

### 步骤 4：mel 滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 个 mel 覆盖 0–8 kHz，`n_fft=400` 时得到一个 `(80, 201)` 矩阵。将 `(n_frames, 201)` 的 STFT 幅度乘以其转置，得到 `(n_frames, 80)` 的 mel 频谱图。

### 步骤 5：对数 mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化 dB），`10 * log10(power + eps)`。Whisper 使用更复杂的截断 + 归一化流程（参见 Whisper 的 `log_mel_spectrogram`）。

### 步骤 6：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每帧对数 mel 应用 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码整体能量）。

## 实际应用

2026 年的技术栈：

| 任务 | 特征 |
|------|------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 log-mel，10 ms hop，25 ms 窗口 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 mel，5–12 ms hop 用于精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 log-mel，10 ms hop |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 log-mel 或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散 token（非 mel） |
| 关键词唤醒 | 40 MFCC 用于微型设备 |

经验法则：**如果你不做音乐，从 80 log-mel 开始。** 任何偏离都需承担举证责任。

## 2026 年仍会出现的陷阱

- **Mel 数量不匹配。** 训练用 80 mel，推理用 128 mel。静默失败。在两端都记录特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 计算的 mel 看起来与 16 kHz 不同。在特征提取*之前*先修复 SR。
- **dB 与 log 混淆。** Whisper 期望 log-mel，不是 dB-mel。一些 HF 流程会自动检测；你的自定义代码不会。
- **归一化漂移。** 训练时每条语句归一化，推理时全局归一化。生产中会使 WER 翻倍的 bug。
- **填充泄漏。** 对片段末尾补零会在尾部帧产生平坦频谱。对称填充或复制填充。

## 交付产物

保存为 `outputs/skill-feature-extractor.md`。该 skill 为给定模型目标选择特征类型、mel 数量、帧/hop 和归一化方式。

## 练习

1. **简单。** 运行 `code/main.py`。它合成一个 chirp（频率扫描 200 → 4000 Hz）并打印每帧的 argmax mel 分箱。绘制（可选）并确认它匹配扫描轨迹。
2. **中等。** 用 `n_mels` 取 `{40, 80, 128}` 和 `frame_len` 取 `{200, 400, 800}` 重新运行。测量时间轴上的尖峰带宽。哪种组合对 chirp 的分辨率最好？
3. **困难。** 实现 `power_to_db` 并比较一个微型 CNN 分类器在 AudioMNIST 上的 ASR 准确率，分别使用 (a) 原始 log-mel，(b) 以 `ref=max` 的 dB-mel，(c) MFCC-13 + delta + delta-delta。报告 top-1 准确率。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Frame | 一段切片 | 喂给一次 FFT 的 25 ms 波形块。 |
| Hop | 步长 | 连续帧之间的采样点数；10 ms 是 ASR 默认值。 |
| Window | Hann/Hamming 那个东西 | 逐点乘数，将帧边缘逐渐衰减到零。 |
| STFT | 频谱图生成器 | 分帧 + 加窗后做 FFT；产生时间 × 频率矩阵。 |
| Mel | 扭曲后的频率 | 对数感知刻度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 那个矩阵 | 将 STFT 投影到 mel 分箱的三角滤波器。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年已标准化。 |
| MFCC | 老派特征 | 对数 mel 的 DCT；13 个系数，去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) —— MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) —— 原始 mel 刻度。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) —— 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) —— `mfcc`、`melspectrogram` 和 hop/window 的参考。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) —— Parakeet + Canary 模型的生产级流程。
