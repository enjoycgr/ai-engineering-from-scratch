---
name: coref-picker
description: 选择指代消解方法、评估计划和集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定用例（single-doc / multi-doc、domain、language），输出：

1. 方法。Rule-based / neural span-based / LLM-prompted / hybrid。一句话理由。
2. 模型。如果是 neural，给出具体 checkpoint 名称。
3. 集成。操作顺序：tokenize → NER → coref → downstream task。
4. 评估。CoNLL F1（MUC + B³ + CEAF-φ4 平均值）在 held-out set 上 + 20 篇文档的 manual cluster review。

拒绝没有 sliding-window merge 的、对超过 2,000 token 文档使用纯 LLM coref 的方案。拒绝任何没有 mention-level precision-recall 报告的 pipeline。标记部署在 demographically diverse text 中的 gender-heuristic 系统。
