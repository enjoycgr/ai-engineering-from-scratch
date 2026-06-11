---
name: whisper-tuner
description: 为给定语言、领域和延迟预算设计 Whisper 微调或推理流程。
version: 1.0.0
phase: 6
lesson: 05
tags: [audio, whisper, asr, fine-tuning, lora]
---

给定一个目标（语言集、领域、片段长度分布、延迟预算、硬件）和数据（可用小时数、质量），输出：

1. 变体。Tiny / Base / Small / Medium / Large-v3 / Turbo。原因。
2. 运行时。vanilla / faster-whisper / whisperx / whisper-streaming。原因。
3. 微调计划。全量微调 vs LoRA（r、target_modules）、冻结编码器策略、epoch 数。
4. 推理保护。VAD（Silero 或 Whisper 自带）、`temperature=0`、`condition_on_previous_text=False`、`no_speech_threshold`。
5. 评估。领域 WER 目标、文本归一化规则、静音片段上的幻觉率检查。

拒绝在没有 VAD 的情况下将 Whisper 部署在任意音频上。拒绝在多块任务中将 `condition_on_previous_text=True` 而不设失控保护。标记任何更换 Whisper tokenizer 或 mel 流程的微调。
