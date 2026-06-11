---
name: otel-genai
description: 使用 OpenTelemetry GenAI semantic conventions（语义约定）为 agent（智能体）添加仪器——invoke_agent、chat、tool_call spans，包含正确的 attributes（属性）和 opt-in content capture（选择加入内容捕获）。
version: 1.0.0
phase: 14
lesson: 23
tags: [opentelemetry, genai, observability, tracing, semantic-conventions]
---

给定一个 agent runtime（智能体运行时），接入 OTel GenAI semantic conventions（语义约定）。

生成：

1. `invoke_agent` span 每次 agent run（智能体运行）。Kind CLIENT 用于 remote agent services（远程智能体服务），INTERNAL 用于 in-process（进程内）。名称：`invoke_agent {gen_ai.agent.name}`。
2. `chat` span 每次 LLM 调用，包含 `gen_ai.operation.name=chat`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`。
3. `tool_call` span 每次 tool invocation（工具调用），包含 `gen_ai.tool.name` 以及适用时的 `gen_ai.data_source.id`（RAG corpus 语料库 / memory store 记忆存储）。
4. Opt-in content capture（选择加入内容捕获）：默认 OFF；当 ON 时，将 inputs/outputs 存储在外部并在 spans 上记录 `*.reference_id`。
5. Context propagation（上下文传播）：使用 W3C trace context headers，以便 multi-process runs（多进程运行）（Claude Agent SDK CLI subprocess）拼接为一个 trace（追踪）。

Hard rejects（硬性拒绝）：

- 默认 inline capture full prompts/outputs（内联捕获完整提示词/输出）。PII 和 secret leakage（机密泄漏）风险；也违反规范。
- 缺少 `gen_ai.provider.name`。Multi-provider dashboards（多提供商仪表盘）会损坏。
- Orphan tool spans（孤立工具跨度）。始终通过 active context 设置 parent-child relation（父子关系）。

Refusal rules（拒绝规则）：

- 如果 runtime 无法跨 process boundaries（进程边界）传播 context，拒绝。Multi-process trace stitching（多进程追踪拼接）对于 Claude Agent SDK + CLI 用户是必需的。
- 如果产品有监管约束（HIPAA、GDPR），拒绝 inline content capture（内联内容捕获）。仅允许 external store（外部存储）with access control（访问控制）。
- 如果后端未设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，警告：attribute names 可能在 collector upgrade 时改变。

输出：`tracer.py`、`attributes.py`、`content_store.py`、`README.md`，解释 span structure（跨度结构）、stability opt-in（稳定性选择加入）和 content-capture policy（内容捕获策略）。最后以 "what to read next" 指向 Lesson 24（backends 后端：Langfuse、Phoenix、Opik）或 Lesson 17 了解 Claude Agent SDK trace-context propagation（追踪上下文传播）。
