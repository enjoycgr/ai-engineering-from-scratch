---
name: structured-output-picker
description: 选择结构化输出方法、schema 设计和验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定用例（供应商、延迟预算、schema 复杂度、失败容忍度），输出：

1. Mechanism (机制). Native vendor structured output、Instructor retries、Outlines FSM 或 XGrammar CFG。一句话理由。
2. Schema design (Schema 设计). 字段顺序（reasoning 在前，answer 在后）、"unknown" 的可空字段、enum vs regex、必填字段。
3. Failure strategy (失败策略). 最大重试次数、回退模型、优雅的 `null` 处理、分布外拒绝。
4. Validation plan (验证计划). Schema compliance rate (目标 100%)、semantic validity (语义有效性, LLM-judge)、字段覆盖率、latency p50/p99。

拒绝任何将 `answer` 或 `decision` 放在 reasoning 字段之前的设计。拒绝在没有 schema 的情况下使用裸 JSON mode。标记仅支持 FSM 的库背后的递归 schema。
