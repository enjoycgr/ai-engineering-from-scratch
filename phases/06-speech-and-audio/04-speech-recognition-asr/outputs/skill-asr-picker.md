---
name: asr-picker
description: 为给定部署目标选择 ASR 模型、解码策略、分块和 LM 融合。
version: 1.0.0
phase: 6
lesson: 04
tags: [audio, asr, speech-recognition]
---

给定一个部署目标（语言列表、领域、延迟预算、硬件、离线/流式、片段时长），输出：

1. 模型。Whisper-large-v3-turbo / Parakeet-TDT / Canary-Flash / wav2vec 2.0 / Moonshine。一句话原因。
2. 解码。Greedy / beam width / temperature fallback / LM 融合权重。原因与质量预算挂钩。
3. 分块和 VAD。Chunk 长度、stride、是否用 Silero-VAD 或 Whisper 自带的 VAD 把关。
4. 语言策略。强制语言 vs 自动 LID；如何处理跨语言帧。
5. 评估计划。领域测试集上的 WER、每说话人覆盖率、静音片段上的幻觉率。

拒绝任何没有 VAD 把关的长格式 Whisper 部署（静音上易产生幻觉）。拒绝没有文本归一化的 WER 报告（小写、去掉标点）。标记任何 beam width > 16 但没有 LM 的方案；原始 beam 在 blank 上没有帮助。
