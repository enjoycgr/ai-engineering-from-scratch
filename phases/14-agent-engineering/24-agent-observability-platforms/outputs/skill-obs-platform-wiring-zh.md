---
name: obs-platform-wiring
description: 选择一款可观测性平台（Langfuse、Phoenix、Opik、Datadog）并将 traces（追踪）+ evals（评估）+ prompt versions（提示词版本）接入现有 agent（智能体）。
version: 1.0.0
phase: 14
lesson: 24
tags: [observability, langfuse, phoenix, opik, datadog, tracing]
---

给定一个 agent runtime（智能体运行时）和产品需求，选择一款可观测性平台并搭建接入脚手架。

决策：

1. 需要 prompt management + session replay in one place（一处管理提示词 + 会话回放） -> **Langfuse**。
2. 需要 deep RAG relevancy + drift/anomaly detection（深度 RAG 相关性 + 漂移/异常检测） -> **Phoenix**。
3. 需要 automated prompt optimization + PII guardrails（自动提示词优化 + PII 护栏） -> **Opik**。
4. 已运行 Datadog -> **Datadog LLM Observability**（v1.37+ 原生映射 GenAI）。
5. 需要 ELv2-free license（无 ELv2 许可） -> **Langfuse** (MIT) 或 **Opik** (Apache 2.0)；避免 Phoenix 用于纯 OSS 分发。

生成：

1. OTel GenAI instrumentation（Lesson 23）—— 这是 common substrate（共同基底）。
2. Platform-specific SDK 或 OTel exporter configuration（平台专属 SDK 或 OTel 导出器配置）。
3. LLM-judge rubric（LLM 评委评分标准）for your domain（factual correctness 事实正确性、scope 范围、tone 语气、refusal quality 拒绝质量）。
4. Prompt versioning wired to traces（Langfuse）或 trace clustering config（Phoenix）或 experiment definitions（Opik）。
5. Guardrails on logged content（记录内容的护栏）：PII redaction（PII 脱敏）、secret scrubbing（机密擦除）。
6. Dashboards（仪表盘）：session health（会话健康）、failure taxonomy（失败分类）、latency distribution（延迟分布）、cost per session（每会话成本）。

Hard rejects（硬性拒绝）：

- 没有 evals 就发布。Tracing alone（仅追踪）只是 expensive logging（昂贵的日志）。
- 使用无 external verification（外部验证）的 self-written LLM-judge（自研 LLM 评委）。CRITIC pattern（Lesson 05）：judges 需要 external tools（外部工具）进行 factual grounding（事实依据）。
- 在 span bodies 中存储 PII。始终使用 external store + reference IDs。

Refusal rules（拒绝规则）：

- 如果用户要求 "one platform for everything（一个平台包办一切）"，拒绝并提供上面的决策。没有单一平台在所有三个维度上占主导。
- 如果产品对每个 agent task（智能体任务）没有 acceptance criteria（验收标准），拒绝发布 evals。LLM-judge 需要 rubric（评分标准）；rubric 需要产品决策。
- 如果用户想要 "no sampling, capture everything（无采样，捕获一切）"，拒绝。Trace volume（追踪量）随流量线性增长；scale 时需要 sampling（head-based 或 tail-based）。

输出：`instrumentation.py`、`judge.py`、`dashboards.md`、`README.md`，解释 platform choice（平台选择）、rubric（评分标准）、sampling strategy（采样策略）和 incident response（事件响应）。最后以 "what to read next" 指向 Lesson 30（eval-driven development 评估驱动开发）或 Lesson 26（failure-mode taxonomy 失败模式分类）。
