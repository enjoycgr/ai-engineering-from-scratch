---
name: otel-genai-instrumentation
description: 为智能体代码库产出检测计划，以端到端发出 OTel GenAI span。
version: 1.0.0
phase: 13
lesson: 20
tags: [otel, observability, gen-ai, tracing]
---

给定智能体代码库（LLM 调用、工具分发、MCP 客户端、子智能体），产出 OTel GenAI 检测计划。

产出：

1. Span 层次结构。根 `agent.invoke_agent`（INTERNAL）和子项：`llm.chat`（CLIENT）、`tool.execute`（INTERNAL）、`mcp.call`（CLIENT）、`subagent.invoke`（INTERNAL）。
2. 每个 span 的属性清单。`gen_ai.operation.name`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`、`gen_ai.usage.*`、`gen_ai.tool.name`、`gen_ai.agent.name`。
3. 传播规则。在每次远程调用上注入 W3C traceparent；对于 MCP stdio 使用 `_meta.traceparent` 作为临时字段。
4. 内容捕获策略。默认关闭；记录启用它的环境变量；命名 PII 风险。
5. 导出器选择。Jaeger / Tempo / Langfuse / Phoenix / Datadog / Honeycomb；OTLP 作为线路。

硬拒绝项：
- 任何缺少跨 MCP 或子智能体边界的 trace 传播的计划。
- 任何默认开启内容捕获的计划。泄漏提示和 PII。
- 任何发出没有 `gen_ai.` 或显式供应商前缀的任意自定义属性的计划。

拒绝规则：
- 如果代码库使用内置 OTel 自动检测的框架（Pydantic AI、LangGraph、AgentOps），首先推荐框架钩子。
- 如果导出器后端是本地部署且团队没有 SRE 支持，推荐托管后端。
- 如果用户要求捕获内容以调试生产，拒绝而没有类型化同意策略和 PII 编辑流水线。

输出：一份一页计划，包含 span 层次结构、每个 span 的属性清单、传播规则、内容捕获策略和导出器选择。以要告警的首要指标结束（通常是 p95 `gen_ai.client.operation.duration`）。
