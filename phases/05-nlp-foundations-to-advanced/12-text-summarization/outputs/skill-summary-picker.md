---
name: summary-picker
description: 选择抽取式或生成式摘要，指定库，添加事实性检查。
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach（方法）。Extractive or abstractive. Explain in one sentence why.
2. Starting model / library（起始模型/库）。Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan（评估计划）。ROUGE-1, ROUGE-2, ROUGE-L (use `rouge-score` with stemming). Plus factuality check if abstractive.
4. One failure mode to probe（一个需要探查的失效模式）。Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization, not just truncation.
