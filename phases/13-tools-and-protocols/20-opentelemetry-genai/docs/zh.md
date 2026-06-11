# OpenTelemetry GenAI —— 端到端追踪工具调用

> 智能体调用五个工具、三个 MCP 服务器和两个子智能体。你需要一个贯穿所有的 trace。OpenTelemetry GenAI 语义约定（v1.37 及以上中的稳定属性）是 2026 年标准，原生支持 Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps。本课命名所需属性、走过 span 层次结构（智能体 → LLM → 工具）并交付一个可以插入任何 OTel 导出器的 stdlib span 发出器。

**类型：** Build
**语言：** Python（stdlib，OTel span 发出器）
**前置要求：** Phase 13 · 07（MCP 服务器），Phase 13 · 08（MCP 客户端）
**时间：** ~75 分钟

## 学习目标

- 命名 LLM span 和工具执行 span 所需的 OTel GenAI 属性。
- 构建覆盖智能体循环、LLM 调用、工具调用和 MCP 客户端分发的 trace 层次结构。
- 决定捕获什么内容（选择加入）vs 编辑什么（默认）。
- 向本地收集器（Jaeger、Langfuse）发出 span 而无需重写工具代码。

## 问题

2026 年 2 月的调试：用户报告 "我的智能体有时需要 30 秒响应；其他时候 3 秒。" 无 trace。日志显示 LLM 调用，但不显示工具分发、不显示 MCP 服务器往返、不显示子智能体。你猜测。最终你发现：一个 MCP 服务器偶尔在冷启动时挂起。

没有端到端追踪，你无法找到这一点。OTel GenAI 修复了它。

约定在 2025-2026 年于 OpenTelemetry 语义约定组下稳定。它们定义稳定属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 span。一次检测；交付到任何后端。

## 概念

### Span 层次结构

```
agent.invoke_agent  (顶层，INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整个东西在一个 trace id 下嵌套。Span id 链接父子关系。

### 所需属性

每 2025-2026 semconv：

- `gen_ai.operation.name` —— `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` —— `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` —— 请求的模型字符串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` —— 实际服务的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` —— 用于关联的提供商响应 id。

对于工具 span：

- `gen_ai.tool.name` —— 工具标识符。
- `gen_ai.tool.call.id` —— 特定调用 id。
- `gen_ai.tool.description` —— 工具描述（可选）。

对于智能体 span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span kinds

- `SpanKind.CLIENT` 用于跨进程边界调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL` 用于智能体自己的循环步骤和工具执行。

### 选择加入内容捕获

默认情况下，span 携带指标和计时——不携带提示或补全。大载荷和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量以包含内容。在启用生产前仔细审查。

### Span 上的事件

Token 级事件可以作为 span 事件添加：

- `gen_ai.content.prompt` —— 输入消息。
- `gen_ai.content.completion` —— 输出消息。
- `gen_ai.content.tool_call` —— 记录的工具调用。

事件在 span 内按时间排序，用于详细重放。

### 导出器

OTel span 导出到：

- **Jaeger / Tempo。** OSS，本地部署。
- **Langfuse。** LLM 可观测性特定；可视化 token 使用。
- **Arize Phoenix。** 评估 + 追踪结合。
- **Datadog。** 商业；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列导向；查询友好。

都说 OTLP，线路格式。你的代码不关心。

### 跨 MCP 传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头部注入请求。Streamable HTTP 支持标准头部。Stdio 原生不携带 HTTP 头部；规范的 2026 年路线图讨论在每个 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在交付之前：在每个请求的 `_meta` 中手动包含 traceparent。服务器记录 trace id。

### 指标

与 span 一起，GenAI semconv 定义指标：

- `gen_ai.client.token.usage` —— 直方图。
- `gen_ai.client.operation.duration` —— 直方图。
- `gen_ai.tool.execution.duration` —— 直方图。

将这些用于不需要每次调用细节的仪表板。

### AgentOps 层

AgentOps（成立于 2024 年）专注于 GenAI 可观测性。它包装流行框架（LangGraph、Pydantic AI、CrewAI）以自动发出 OTel span。如果你的栈使用受支持的框架，这很有用；否则使用手动检测。

## 使用它

`code/main.py` 为一个调用 LLM、分发两个工具并执行一次 MCP 往返的智能体发出 OTel 形状的 span 到 stdout（以 OTLP-JSON 类似格式）。无真实导出器——本课专注于 span 形状和属性集。将输出粘贴到 OTLP 兼容查看器中或只是阅读它。

看点：

- Trace id 在所有 span 间共享。
- 父子链接通过 `parentSpanId` 编码。
- 所需 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量开启它。

## 交付它

本课产出 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该技能产出检测计划：在哪里添加 span、填充哪些属性以及目标哪些导出器。

## 练习

1. 运行 `code/main.py`。计数 span 并识别哪些是 CLIENT vs INTERNAL。

2. 开启内容捕获（环境变量）并确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意 PII 影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration` 并作为每次调用的直方图样本发出它。

4. 将 traceparent 从父智能体 span 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器会看到相同的 trace id。

5. 阅读 OTel GenAI semconv 规范。找出 semconv 中列出但本课代码未发出的一个属性。添加它。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | 追踪、指标、日志的开放标准 |
| GenAI semconv | "GenAI 语义约定" | LLM / 工具 / 智能体 span 的稳定属性名称 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "计时操作" | 具备开始、结束和属性的工作单元 |
| Trace | "跨 span 血统" | 共享 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 关于 span 方向的提示 |
| OTLP | "OpenTelemetry Line Protocol" | 导出器的线路格式 |
| 选择加入内容 | "提示 / 补全捕获" | 默认关闭；环境变量开启 |
| traceparent | "W3C 头部" | 跨服务传播 trace 上下文 |
| 导出器 | "后端特定发送器" | 将 span 发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— GenAI span、指标和事件的规范约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) —— LLM 和工具执行 span 属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) —— 智能体级 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) —— GitHub 托管的真实来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) —— 生产集成走过
