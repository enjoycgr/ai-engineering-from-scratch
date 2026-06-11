---
name: retrieval-picker
description: 为给定语料库和查询模式选择检索技术栈。
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

给定需求（语料库大小、查询模式、延迟预算、质量门槛、基础设施约束），输出：

1. 技术栈 (Stack)。仅 BM25、仅 dense、hybrid（BM25 + dense + RRF）、hybrid + cross-encoder rerank，或三路（BM25 + dense + learned-sparse）。
2. 稠密编码器 (Dense encoder)。命名具体模型。根据语言、领域和上下文长度匹配。
3. 重排器 (Reranker)。如果使用了，命名具体的 cross-encoder 模型。标记重排会在 top-30 上增加 30-100ms 延迟。
4. 评估方案 (Evaluation plan)。Recall@10 是检索器的主要指标。多答案场景使用 MRR。先建立基线，再针对基线测量增量改进。

对于包含命名实体、错误代码或产品 SKU 的语料库，除非用户有证据表明 dense 能处理精确匹配，否则拒绝推荐纯 dense 方案。对于高风险检索（法律、医疗），如果最终 top-5 决定用户答案，则拒绝跳过重排。
