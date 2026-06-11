# 音频分类 —— 从 MFCC 上的 k-NN 到 AST 和 BEATs

> 从"狗叫 vs 警笛"到"这是哪种语言"，都属于音频分类。特征是 mel。架构每十年更新一次。评估指标始终是 AUC、F1 和每类召回率。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图与 Mel），Phase 3 · 06（CNN），Phase 5 · 08（文本的 CNN 与 RNN）
**时间：** 约 75 分钟

## 问题

你拿到一段 10 秒的音频片段。你想知道："这是什么？"城市声音（警笛、电钻、狗叫）、语音命令（是/否/停止）、语言识别（英/西/阿）、说话人情绪（愤怒/中性）、或环境声音（室内/室外、嘈杂）。这些都属于*音频分类*，而在 2026 年，基线架构已经很成熟：log-mel → CNN 或 Transformer → softmax。

核心难点不是网络。是数据。音频数据集存在严重的类别不平衡、强烈的域偏移（干净 vs 噪声），以及标签噪声（谁决定了"城市嘈杂" vs "餐厅噪音"？）。80% 的问题在于数据整理、增强和评估，而不是把 CNN 换成 Transformer。

## 概念

![音频分类阶梯：从 MFCC 上的 k-NN 到 AST 到 BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990 年代基线）。** 将每段音频的 MFCC 展平，计算与标注样本库的余弦相似度，返回前 K 个的多数投票。在干净的小数据集（Speech Commands、ESC-50）上出奇地强。无需 GPU。

**Log-mel 上的 2D CNN（2015-2019）。** 将 `(T, n_mels)` 的 log-mel 当作图像处理。应用 ResNet-18 或 VGG 风格。对时间轴做全局均值池化。Softmax 分类。仍是 2026 年大多数 Kaggle 竞赛的基线。

**音频频谱图 Transformer，AST（2021-2024）。** 将 log-mel 分块（例如 16×16 的 patch），添加位置嵌入，输入 ViT。在 AudioSet（mAP 0.485）上达到监督学习的 SOTA。

**BEATs 和 WavLM-base（2024-2026）。** 在数百万小时数据上做自监督预训练。在你的任务上微调，只需 1-10% 的监督数据。2026 年这是非语音音频的默认起点。BEATs-iter3 在 AudioSet 上比 AST 高 1-2 mAP，同时计算量仅为 1/4。

**Whisper 编码器作为冻结主干（2024）。** 取 Whisper 的编码器，去掉解码器，接一个线性分类器。在语言识别和简单事件分类上接近 SOTA，无需任何音频增强。"免费午餐"基线。

### 类别不平衡才是真正的挑战

ESC-50：50 个类别，每类 40 段 —— 平衡，简单。UrbanSound8K：10 个类别，不平衡 10:1。AudioSet：632 个类别，长尾比例达 100,000:1。有效的技术：

- 训练时平衡采样（评估时不使用）。
- Mixup：将两段音频（及其标签）线性插值作为增强。
- SpecAugment：随机屏蔽时间和频率带。简单；关键。

### 评估

- 多类互斥（Speech Commands）：top-1 准确率，top-5 准确率。
- 多类多标签（AudioSet、UrbanSound 风格）：平均精度均值（mAP, mean average precision）。
- 严重不平衡：每类召回率 + macro F1。

2026 年你应该知道的数字：

| 基准测试 | 基线 | 2026 SOTA | 来源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs 论文 (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR 排行榜 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 结果 |

## 动手实现

### 步骤 1：特征提取

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤 2：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单但强大：跨时间取均值 + 方差，为 13 系数 MFCC 给出 26 维固定嵌入。瞬间运行。直到 2017 年仍在 ESC-50 上击败当时的 SOTA 神经网络基线。

### 步骤 3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤 4：升级到 log-mel 上的 CNN

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

300 万参数。在 ESC-50 上用单张 RTX 4090 约 10 分钟训练完成。准确率 80%+。

### 步骤 5：2026 年的默认方案 —— 微调 BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，通过 `beats` 库使用 `microsoft/BEATs-base`；transformers API 结构相同。

## 实际应用

2026 年的技术栈：

| 场景 | 起步方案 |
|-----------|-----------|
| 小型数据集 (<1000 段) | MFCC 均值的 k-NN（你的基线）+ 音频增强 |
| 中等数据集 (1K–100K) | BEATs 或 AST 微调 |
| 大型数据集 (>100K) | 从头训练或微调 Whisper 编码器 |
| 实时、边缘设备 | 40-MFCC CNN，量化为 int8（KWS 风格） |
| 多标签 (AudioSet) | BEATs-iter3 + BCE 损失 + mixup + SpecAugment |
| 语言识别 | MMS-LID，SpeechBrain VoxLingua107 基线 |

决策规则：**从冻结主干开始，而不是从零开始训练**。微调 BEATs 的头部可以在几小时内达到 SOTA 的 95%，而不是几周。

## 交付产物

保存为 `outputs/skill-classifier-designer.md`。为给定音频分类任务选择架构、增强、类别平衡策略和评估指标。

## 练习

1. **简单。** 运行 `code/main.py`。它在 4 类合成数据集（不同音高的纯音）上训练 k-NN MFCC 基线。报告混淆矩阵。
2. **中等。** 将 `summarize` 替换为 [mean, var, skew, kurtosis]。在相同合成数据集上，4 阶矩池化是否优于 mean+var？
3. **困难。** 使用 `torchaudio`，在 ESC-50 第 1 fold 上训练 2D CNN。报告 5-fold 交叉验证准确率。添加 SpecAugment（时间 mask = 20，频率 mask = 10）并报告提升。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| AudioSet | 音频界的 ImageNet | Google 的 200 万段、632 类弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准 | 50 类 × 40 段环境声音。 |
| AST | 音频频谱图 Transformer | ViT 作用于 log-mel patch；2021 年 SOTA。 |
| BEATs | 自监督音频 | Microsoft 模型，iter3 截至 2026 年领先 AudioSet。 |
| Mixup | 成对增强 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的增强 | 将频谱图的随机时间和频率带置零。 |
| mAP | 主要多标签指标 | 跨类别和阈值的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) —— 2021–2024 年的标杆架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) —— 2024 年之后的默认方案。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) —— 主导的音频增强方法。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) —— 延续至今的 50 类基准。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) —— 632 类 YouTube 分类体系；仍是黄金标准。
