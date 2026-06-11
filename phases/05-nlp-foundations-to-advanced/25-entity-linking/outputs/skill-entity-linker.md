---
name: entity-linker
description: 设计一个实体链接管道 —— 知识库、候选生成器、消歧器、评估。
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

给定一个用例（领域 KB、语言、数据量、延迟预算），输出：

1. 知识库 (Knowledge base)。Wikidata / Wikipedia / 自定义 KB。版本日期。刷新周期。
2. 候选生成器 (Candidate generator)。别名索引、嵌入 (embedding) 或混合方案。目标 mention recall @ K。
3. 消歧器 (Disambiguator)。先验 + 上下文、基于嵌入、生成式或 LLM 提示。
4. NIL 策略。对最高分的阈值、分类器或显式 NIL 候选。
5. 评估。在 held-out 集上测量 mention recall @ 30、top-1 准确率、NIL 检测 F1。

拒绝任何没有 mention-recall 基线的 EL 管道（你无法在不知道候选生成是否 surfaced 正确实体的情况下评估消歧器）。拒绝任何使用 LLM 提示 EL 但没有将输出约束到有效 KB id 的管道。标记那些 popularity bias 影响少数实体（例如名称冲突）且没有领域微调 (domain fine-tuning) 的系统。
