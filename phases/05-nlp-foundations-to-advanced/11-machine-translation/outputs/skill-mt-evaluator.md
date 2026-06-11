---
name: mt-evaluator
description: 评估机器翻译输出，判断是否可上线。
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate（自动分数估计）。你预期的 BLEU 和 chrF 范围。说明是否有 reference（参考译文）可用。
2. Five-point human-verifiable checklist（五点人工可验证清单）：content preservation（内容保留，无幻觉）、correct target language（目标语言正确）、register / formality match（语域/正式程度匹配）、terminology consistency with glossary if provided（如提供词汇表则检查术语一致性）、no truncation or length explosion（无截断或长度爆炸）。
3. One domain-specific issue to probe（一个需要探查的领域特定问题）。Legal: named entities, statute citations. Medical: drug names, dosages. UI: placeholder variables like `{name}`.
4. Confidence flag（置信度标记）。"Ship" / "Ship with review" / "Do not ship"。与第 2 步发现的问题严重程度挂钩。

Refuse to ship without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
