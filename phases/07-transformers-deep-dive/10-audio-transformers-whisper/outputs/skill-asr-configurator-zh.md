---
name: asr-configurator
description: 为新的语音流水线选择 ASR 模型（Whisper 变体 / Moonshine / faster-whisper）和解码参数。
version: 1.0.0
phase: 7
lesson: 10
tags: [transformers, whisper, asr, speech]
---

给定一个语音任务（转录 / 翻译 / 流式处理 / 设备端推理）、语言、音频特征（噪声、口音、时长）以及延迟/质量目标，输出：

1. 模型选择。以下之一：faster-whisper large-v3-turbo（生产环境默认）、whisper large-v3（最高质量、多语言）、whisper medium（中端）、Moonshine base（边缘设备）、distil-whisper（英语场景快 2 倍）。附一句选择理由。
2. 量化（Quantization）。int8_float16（CPU 默认）、float16（GPU 默认）、fp32（研究用）。标注 VRAM 影响。
3. 解码（Decoding）。束宽度（beam width，典型值 5，流式处理为 1）、温度回退策略（temperature fallback schedule）、对数概率阈值（log-prob threshold）、无语音阈值（no-speech threshold）、VAD（Voice Activity Detection，语音活动检测）门控开关。
4. 分块（Chunking）。30 秒固定窗口 vs 流式分块（典型为 10 秒，重叠 2 秒）+ 基于 VAD 的分段。记录重叠区域的后合并策略。
5. 后处理（Post-processing）。时间戳对齐（WhisperX 强制对齐）、标点恢复、说话人分离（diarization，pyannote）。标注哪些是该任务必需的。

拒绝推荐原始 OpenAI Whisper（参考实现）用于生产环境 —— `faster-whisper` 速度快 4 倍且输出一致。拒绝在没有 VAD 的情况下交付流式 ASR，除非有文档说明原因。当输入可能是多说话人时，标注单说话人假设。
