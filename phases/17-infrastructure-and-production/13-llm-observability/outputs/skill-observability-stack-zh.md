---
name: observability-stack
description: 给定栈、规模、预算和许可姿态，选择 LLM 可观测性栈（开发平台 + 网关 + 可选规模化层），并定义 OpenTelemetry GenAI 属性集。
version: 1.0.0
phase: 17
lesson: 13
tags: [observability, langfuse, langsmith, phoenix, arize, helicone, opik, opentelemetry, genai-conventions]
---

给定栈（LangChain / DSPy / 原始 SDK）、规模（追踪/天）、预算、许可姿态（仅 MIT vs 商业可接受）和自托管需求，生成可观测性计划。

产出：

1. 开发平台选择。Langfuse（开源）、LangSmith（LangChain 优先商业）、Opik（Comet 开源）或无。用栈和许可论证。
2. 网关/遥测选择。Helicone（代理 + 网关）、SigNoz（全栈 APM）、OpenLLMetry（纯 OTel）。如果已使用 AI 网关（Phase 17 · 19），命名集成。
3. 规模化/湖层。可选；Arize AX 或原始 Iceberg 用于长期分析，Phoenix 用于 RAG 漂移。
4. OTel GenAI 约定。指定最小属性集：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.request.temperature`、`gen_ai.response.finish_reasons`，加上组织特定（tenant_id、user_id、task）。
5. 采样策略。100% 错误，100% 高成本（>$0.10/调用），N% 成功采样率。原始保留窗口（14d / 30d / 90d）。聚合保留更久。
6. 告警。五个必须有告警的指标：错误率、P99 TTFT、成本/请求、提示缓存命中率、拒绝率。

硬性拒绝：
- 在没有 OTel 回退的情况下在框架特定 SDK 内插桩。拒绝 —— 框架锁定。
- 对于非监管工作负载，在 Datadog 级定价 >$500/月时保留 100% 追踪。拒绝 —— 推荐采样。
- 忽略 OpenTelemetry GenAI 约定。拒绝 —— 2026 互操作需要它们。

拒绝规则：
- 如果追踪/天 > 5M 且团队坚持完整 Datadog 保留，拒绝而不提供成本预测。
- 如果团队仅 MIT 且选择 LangSmith，拒绝 —— Langfuse 是 MIT 等效。
- 如果团队没有 AI 网关且选择 Helicone 作为网关和可观测性，接受 —— 代理在约 500 RPS 以下也充当网关（Phase 17 · 19 涵盖网关规模）。

输出：一页计划，命名开发平台、网关、规模化层（如有）、OTel 属性集、采样规则、五个告警。结尾附单一指标：过去 7 天内具有完整 OTel GenAI 属性的 LLM 调用百分比。
