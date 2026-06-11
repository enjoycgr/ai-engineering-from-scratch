---
name: feature-extractor
description: 为下游音频模型选择特征类型、mel 数量、帧/hop 和归一化方式。
version: 1.0.0
phase: 6
lesson: 02
tags: [audio, features, spectrogram, mel]
---

给定一个目标模型（ASR / TTS / 分类器 / 说话人 / 音乐）和输入音频（采样率、领域），输出：

1. 特征类型。Log-mel、mel、MFCC、原始波形，或离散编解码器（EnCodec、SoundStream）。一句话说明原因。
2. Mel 数量和频率范围。`n_mels`、`fmin`、`fmax`。原因关联到领域（语音 vs 音乐）和模型目标。
3. 帧和 hop。`frame_len`、`hop_len`、窗类型。原因关联到所需的时间分辨率。
4. 归一化。每条语句均值/方差、全局统计量、或固定参考的 dB；特征提取前或后。
5. 验证代码片段。在 1 秒参考片段上打印结果形状、最小/最大值、均值/标准差，并断言它们与训练时匹配的 Python 代码。

拒绝交付任何帧/hop/mel 数量与目标模型已发表训练配置不同的特征流程。标记任何为 Whisper 或 Parakeet 设计的基于 MFCC 的方案为错误——这些模型消费 log-mel。标记任何没有归一化断言的特征提取器。
