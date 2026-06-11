---
name: multilingual-picker
description: 为多语言 NLP 任务选择源语言、目标模型和评估计划。
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

给定需求（目标语言、任务类型、每种语言的可用标注数据），输出：

1. 微调 (fine-tuning) 的源语言 (source language)。默认 English；如果目标语言有类型学上接近的高资源语言，检查 LANGRANK 或 qWALS。
2. 基础模型 (base model)。XLM-R（分类）、mT5（生成）、NLLB（翻译）、Aya-23（生成式 LLM）。
3. Few-shot 预算。如果可用，从 100-500 条目标语言样本开始。仅在标注不可行时采用 zero-shot。
4. 评估计划。Per-language accuracy（不要聚合）、cross-lingual consistency、非 Latin 脚本上的 entity-level F1。

拒绝在没有 per-language evaluation 的情况下交付多语言模型 —— 聚合指标掩盖了长尾失败。标记 tokenization 覆盖率低下的脚本（Amharic、Tigrinya、许多非洲语言）为需要 byte-fallback 的模型（SentencePiece 的 byte_fallback=True，或 GPT-2 等 byte-level tokenizer）。
