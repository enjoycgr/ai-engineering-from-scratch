---
name: grammar-pipeline
description: 为下游 NLP 任务设计经典的 POS + 依存分析流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定一个下游任务（信息抽取、改写验证、查询分解、词形还原），输出：

1. 标签集（Tagset）。纯英语遗留流水线用 Penn Treebank，多语言或跨语言用 Universal Dependencies。
2. 库（Library）。大多数生产环境用 spaCy（`en_core_web_sm` / `_lg` / `_trf`），学术级多语言用 stanza，最高 UD 准确率用 trankit。
3. 集成片段（Integration snippet）。调用库并消费 `.pos_`、`.dep_`、`.head` 的 3-5 行代码。
4. 需要测试的失效模式（Failure mode to test）。名词-动词歧义（`saw`、`book`、`can`）和介词短语附着歧义（PP-attachment ambiguity）是经典陷阱。采样 20 条输出并人工检查。

拒绝推荐从零自建分析器。从零构建分析器是一个研究项目，不是应用任务。标记任何不处理小写/大写变体就消费 POS 标签的流水线为脆弱。
