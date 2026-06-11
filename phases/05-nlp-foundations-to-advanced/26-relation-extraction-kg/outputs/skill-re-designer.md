---
name: re-designer
description: 设计一个带来源 provenance 和规范化 (canonicalization) 的关系抽取管道。
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

给定一个语料库（领域、语言、数据量）和下游用途（KG-RAG、分析、合规），输出：

1. 抽取器 (Extractor)。基于模式 / 监督 / LLM / AEVS 混合。理由与精确率 vs 召回率 (precision vs recall) 目标挂钩。
2. 本体 (Ontology)。封闭属性列表（Wikidata / 领域）或带规范化检查的开放 IE。
3. 来源 (Provenance)。每个三元组携带源字符跨度 + 文档 id。对于审计是不可协商的。
4. 合并策略。规范实体 id + 关系 id + 时序限定符；去重策略。
5. 评估。在 200 个手工标注三元组上的精确率 / 召回率 + LLM 抽取样本上的幻觉率 (hallucination rate)。

拒绝任何没有跨度验证（来源 provenance）的基于 LLM 的 RE 管道。拒绝未经规范化就流入生产图谱的开放 IE 输出。标记对时间受限关系（雇主、配偶、职位）没有时序限定符的管道。
