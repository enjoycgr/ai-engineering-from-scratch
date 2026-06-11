---
name: embedding-picker
description: 为给定语料库和部署环境选择 embedding 模型、维度和检索模式。
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

给定语料库（大小、语言、领域、平均长度）、部署目标（cloud / edge / on-prem）、latency budget（延迟预算）和 storage budget（存储预算），输出：

1. Model. 具名的 checkpoint 或 API。一句话说明理由。
2. Dimension. Full / Matryoshka-truncated / int8-quantized。理由关联 storage budget。
3. Mode. Dense / sparse / multi-vector / hybrid。理由。
4. Query prefix / template，如果模型卡要求。
5. Evaluation plan. 与领域相关的 MTEB 任务 + 使用 nDCG@10 的留出领域评估。

拒绝在未经验证的情况下将 Matryoshka 截断到 <64 维的建议。拒绝为 <10k 段落的语料库推荐 ColBERTv2（开销不合理）。标记将 >8k tokens 的长文档语料库路由到 512-token 窗口模型的做法。
