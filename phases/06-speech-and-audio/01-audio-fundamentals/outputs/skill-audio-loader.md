---
name: audio-loader
description: 根据目标模型的期望验证原始音频文件，并在需要时安全重采样。
version: 1.0.0
phase: 6
lesson: 01
tags: [audio, speech, preprocessing]
---

给定一个音频文件（路径、声道、采样率、位深、编解码器）和一个目标模型（具有所需采样率和声道数的 ASR / TTS / 分类器），输出：

1. 不匹配项。列出文件与目标在每个维度上的差异（sr、声道、时长下限、削波检查）。
2. 重采样计划。源 sr、目标 sr、重采样库（`torchaudio.transforms.Resample` 或 `librosa.resample`）、抗混叠滤波器类型。
3. 声道计划。单声道折叠策略（均值 vs 仅左声道），或当模型支持时多声道直通。
4. 归一化。峰值 vs RMS 归一化、目标 dBFS、削波保护。
5. 验证代码片段。加载文件、运行变换、并断言最终数组匹配 `(target_sr, dtype, channel_count, range)` 的 Python 代码。

拒绝不带抗混叠滤波器的降采样。拒绝超过 2 倍放大而不使用重建滤波器的升采样。标记任何峰值超过 ±0.999 或直流偏移超过 ±0.01 的输入文件。
