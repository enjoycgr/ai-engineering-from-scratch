# OpenTelemetry GenAI Semantic Conventions

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了 agent telemetry（智能体遥测）的标准 schema。Span names（跨度名称）、attributes（属性）和 content-capture rules（内容捕获规则）在 vendors（厂商）间趋同，因此 agent traces（智能体追踪）在 Datadog、Grafana、Jaeger 和 Honeycomb 中含义相同。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 分钟

## Learning Objectives

- 命名 GenAI span categories（跨度类别）：model/client、agent、tool。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL spans，以及各自适用的场景。
- 列出顶层 GenAI attributes（属性）：provider name（提供商名称）、request model（请求模型）、data-source ID（数据源 ID）。
- 解释 content-capture contract（内容捕获契约）：opt-in（选择加入）、`OTEL_SEMCONV_STABILITY_OPT_IN`、external-reference recommendation（外部引用推荐）。

## The Problem

每个 vendor（厂商）都发明自己的 span names（跨度名称）。Ops 团队最终为每个框架构建 dashboard（仪表盘）。OpenTelemetry 的 GenAI SIG 通过定义一个标准来修复这个问题，整个 ecosystem（生态系统）都围绕它对齐。

## The Concept

### Span categories（跨度类别）

1. **Model / client spans（模型/客户端跨度）。** 覆盖原始 LLM 调用。由 provider SDKs（Anthropic、OpenAI、Bedrock）和 framework model adapters（框架模型适配器）发射。
2. **Agent spans（智能体跨度）。** `create_agent`（agent 构建时）和 `invoke_agent`（agent 运行时）。
3. **Tool spans（工具跨度）。** 每次 tool invocation（工具调用）一个；通过 parent-child relation（父子关系）连接到 agent span。

### Agent span naming（智能体跨度命名）

- Span name：`invoke_agent {gen_ai.agent.name}`（如果有命名）；回退到 `invoke_agent`。
- Span kind（跨度类型）：
  - **CLIENT** —— 用于 remote agent services（远程智能体服务）（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** —— 用于 in-process agent frameworks（进程内智能体框架）（LangChain、CrewAI、本地 ReAct）。

### Key attributes（关键属性）

- `gen_ai.provider.name` —— `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` —— 模型 ID。
- `gen_ai.response.model` —— 解析后的模型（可能因 routing 而与请求不同）。
- `gen_ai.agent.name` —— agent identifier（智能体标识符）。
- `gen_ai.operation.name` —— `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` —— 用于 RAG：查询了哪个 corpus（语料库）或 store（存储）。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 存在技术特定的约定。

### Content capture（内容捕获）

默认规则：instrumentations（仪器）默认不应捕获 inputs/outputs（输入/输出）。捕获通过以下方式 opt-in（选择加入）：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在 spans 上记录 references（引用）（pointer IDs，而非文本）。这是 Lesson 27 的 content-poisoning defense（内容中毒防御）与 observability（可观测性）的结合。

### Stability（稳定性）

截至 2026 年 3 月，大多数约定仍处于 experimental（实验性）状态。通过以下方式 opt in（选择加入）stable preview（稳定预览）：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将其 LLM Observability schema 原生映射 GenAI attributes。其他后端（Grafana、Honeycomb、Jaeger）支持 raw attributes。

### 该模式在何处出错

- **在 spans 中捕获完整 prompts。** PII、secrets、客户数据进入 traces，ops 人员可以读取。存储在外部。
- **缺少 `gen_ai.provider.name`。** 缺少 attribution（归属）时，multi-provider dashboards（多提供商仪表盘）会损坏。
- **无 parent links 的 spans。** Orphaned tool spans（孤立的工具跨度）。始终传播 context（上下文）。
- **未设置 stability opt-in。** 你的 attributes 可能在后端升级时被重命名。

## Build It

`code/main.py` 实现一个符合 GenAI 约定的 stdlib span emitter（跨度发射器）：

- 带有 GenAI attribute schema（属性模式）的 `Span`。
- 带有 `start_span`、nested contexts（嵌套上下文）的 `Tracer`。
- 脚本化的 agent run（智能体运行），发射：`create_agent`、`invoke_agent` (INTERNAL)、per-tool spans（每工具跨度）、LLM 调用的 `chat` spans。
- 将 prompts 存储在外部并在 spans 上记录 IDs 的 content-capture mode（内容捕获模式）。

运行方式：

```
python3 code/main.py
```

输出：一个包含所有必需 GenAI attributes 的 span tree（跨度树），以及一个展示 opt-in content references（选择加入内容引用）的 "external store（外部存储）"。

## Use It

- **Datadog LLM Observability** (v1.37+) 原生映射 attributes。
- **Langfuse / Phoenix / Opik** (Lesson 24) —— auto-instrument the ecosystem（自动检测生态系统）。
- **Jaeger / Honeycomb / Grafana Tempo** —— raw OTel traces；从 GenAI attributes 构建 dashboards。
- **Self-hosted** —— 运行带 GenAI processor 的 OTel Collector。

## Ship It

`outputs/skill-otel-genai.md` 将 OTel GenAI spans 接入现有 agent，包含 content-capture defaults（内容捕获默认值）和 external-reference storage（外部引用存储）。

## Exercises

1. 用 `invoke_agent` (INTERNAL) + per-tool spans 为 Lesson 01 的 ReAct loop 添加仪器。发送到 Jaeger 实例。
2. 以 "references only" 模式添加 content capture（内容捕获）：prompts 存入 SQLite，span attributes 仅携带 row IDs。
3. 阅读 `gen_ai.data_source.id` 的 spec。将其接入 Lesson 09 的 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的 attributes 不会被 collector 重命名。
5. 构建一个 dashboard（仪表盘）：仅从 GenAI attributes 统计 "哪些 tool errors 与哪些 models 相关"。

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GenAI SIG | "OpenTelemetry GenAI 组" | 定义 schema 的 OTel working group（工作组） |
| invoke_agent | "Agent span（智能体跨度）" | 代表 agent run（智能体运行）的 span 名称 |
| CLIENT span | "Remote call（远程调用）" | 调用远程 agent service（智能体服务）的 span |
| INTERNAL span | "In-process（进程内）" | 进程内 agent run（智能体运行）的 span |
| gen_ai.provider.name | "Provider（提供商）" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source（RAG 来源）" | 检索命中的哪个 corpus（语料库）/store（存储） |
| Content capture | "Prompt logging（提示词日志）" | 消息的 opt-in（选择加入）捕获；生产中存储在外部 |
| Stability opt-in | "Preview mode（预览模式）" | 用于固定 experimental conventions（实验性约定）的环境变量 |

## Further Reading

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— 默认 GenAI spans
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 内置 OTel spans
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) —— W3C trace context propagation（追踪上下文传播）
