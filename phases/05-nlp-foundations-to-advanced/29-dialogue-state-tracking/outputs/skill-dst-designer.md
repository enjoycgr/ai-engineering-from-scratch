---
name: dst-designer
description: 设计对话状态追踪器 —— schema、提取器、更新策略、评估。
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

给定一个用例（domain（领域）、语言、词汇开放度、合规需求），输出：

1. Schema。domain 列表、每个 domain 的 slot、每个 slot 的开放 vs 封闭词汇。
2. Extractor（提取器）。基于规则 / seq2seq / LLM-with-Pydantic。给出理由。
3. Update policy（更新策略）。全状态重新生成 / 增量更新；修正处理；否定处理。
4. Evaluation（评估）。在留出对话集上的 Joint Goal Accuracy、slot 级别 precision/recall、最难 slot 的混淆情况。
5. Confirmation flow（确认流程）。何时明确请求用户确认（破坏性操作、低置信度提取）。

拒绝无基于规则的二次检查的、用于合规敏感 slot 的纯 LLM DST。拒绝无法回滚用户修正的 slot 的任何 DST。标记无版本号的 schema。
