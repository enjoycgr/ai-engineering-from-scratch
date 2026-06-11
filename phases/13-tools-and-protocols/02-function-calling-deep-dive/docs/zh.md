# 函数调用深度解析 —— OpenAI、Anthropic、Gemini

> 三大前沿提供商在 2024 年汇聚于同一个工具调用循环，随后在其他方面分道扬镳。OpenAI 使用 `tools` 和 `tool_calls`。Anthropic 使用 `tool_use` 和 `tool_result` 块。Gemini 使用 `functionDeclarations` 和唯一 id 关联。本课将三者并排对比，使你在一个提供商上运行的代码在移植到另一个时不会中断。

**类型：** Build
**语言：** Python（stdlib，schema 转换器）
**前置要求：** Phase 13 · 01（工具接口）
**时间：** ~75 分钟

## 学习目标

- 陈述 OpenAI、Anthropic 和 Gemini 函数调用载荷（声明、调用、结果）之间的三个形状差异。
- 将一个工具声明翻译成所有三个提供商格式，并预测严格模式约束在何处会不同。
- 在每个提供商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个提供商的硬性限制（工具数量、schema 深度、参数长度）以及违反限制时各自发出的错误签名。

## 问题

函数调用请求的形状因提供商而异。来自 2026 年生产栈的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个你必须解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码强制执行 schema 合规性。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 形式返回。`input` 已经是解析好的（对象，而非字符串）。你用一个新的 `user` 消息回复，其中包含 `{type: "tool_result", tool_use_id, content}` 块。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 形式到达，其中 `id` 在 Gemini 3 及以上版本中唯一，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 回复。

相同的循环。不同的字段名、不同的嵌套、不同的字符串与对象约定、不同的关联机制。一个在 OpenAI 上编写天气代理的团队，仅仅为了管道代码的移植，就需要花两天时间移植到 Anthropic，再花一天移植到 Gemini。

本课构建一个转换器，将三种格式统一为一种规范化的工具声明，并在边缘路由。Phase 13 · 17 将同样的模式泛化为 LLM 网关。

## 概念

### 共同结构

每个提供商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入 schema。
2. **工具选择。** 强制特定工具、禁止工具，或让模型决定。
3. **调用发出。** 命名工具和参数的结构化输出。
4. **调用 id。** 将响应与正确的调用关联（对并行很重要）。
5. **结果注入。** 将结果绑定回调用的消息或块。

### 形状差异，逐字段

| 方面 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 声明信封 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | `content[]` 类型为 `tool_use` | `parts[]` 类型为 `functionCall` |
| 参数类型 | 字符串化 JSON | 解析后的对象 | 解析后的对象 |
| Id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | `user` 带 `tool_result`，`tool_use_id` | `functionResponse` 带匹配 `id` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格 schema | `strict: true` | schema 就是 schema（始终强制执行） | `responseSchema` 在请求级别 |

### 你实际会遇到的限制

- **OpenAI。** 每次请求 128 个工具。Schema 深度 5。参数字符串 <= 8192 字节。严格模式要求无 `$ref`，无重叠的 `oneOf`/`anyOf`/`allOf`，每个属性都列在 `required` 中。
- **Anthropic。** 每次请求 64 个工具。Schema 深度实际上无限制，但实际限制为 10。无严格模式标志；schema 是契约，模型倾向于遵守。
- **Gemini。** 每次请求 64 个函数。Schema 类型是 OpenAPI 3.0 子集（与 JSON Schema 2020-12 略有分歧）。Gemini 3 起并行调用带唯一 id。

### `tool_choice` 行为

每个人都支持的三种模式，名称不同。

- **Auto。** 模型选择工具或文本。默认。
- **Required / Any。** 模型必须至少调用一个工具。
- **None。** 模型不得调用工具。

加上每个提供商独有的一个模式：

- **OpenAI。** 按名称强制特定工具。
- **Anthropic。** 按名称强制特定工具；`disable_parallel_tool_use` 标志区分单调用与多调用。
- **Gemini。** `mode: "VALIDATED"` 将所有响应路由通过 schema 验证器，无论模型意图如何。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一个助手消息中发出多个调用。你全部运行它们，并用一个批处理工具角色消息回复，每个 `tool_call_id` 对应一个条目。Anthropic 历史上是单调用；`disable_parallel_tool_use: false`（Claude 3.5 起默认）启用多调用。Gemini 2 允许并行调用但不提供稳定 id；Gemini 3 添加 UUID 以便乱序响应能干净地关联。

### 流式传输

三者都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块到达。你累积直到 `finish_reason: "tool_calls"`。
- **Anthropic。** 块开始 / 块增量 / 块停止事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）发出带有 `functionCallId` 的块，因此多个并行调用可以交错。

Phase 13 · 03 深入讲解并行 + 流式重组。本课专注于声明和单调用形状。

### 错误与修复

无效参数错误看起来也不同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码期间发生；无效 JSON 不可能出现，但可能出现 `refusal`。
- **Anthropic。** `input` 可能包含意外字段；schema 是建议性的。在服务器端验证。
- **Gemini。** OpenAPI 3.0 怪癖：对象字段上的 `enum` 被静默忽略；自行验证。

### 转换器模式

代码中的规范化工具声明看起来像这样（形状由你选择）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三个提供商形状。`code/main.py` 中的 harness 正是这样做的，然后让每个提供商的响应形状往返于一个虚拟工具调用。不需要网络——本课教授形状，而非 HTTP。

生产团队将此转换器包装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。Phase 13 · 17 交付一个在三种提供商前面暴露 OpenAI 形状 API 的网关。

## 使用它

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个转换器，它们发出 OpenAI、Anthropic 和 Gemini 声明 JSON。然后它从每种提供商形状解析手工制作的响应到相同的规范调用对象，证明语义在表面之下是相同的。运行它并并排对比三个声明。

看点：

- 三个声明块仅在信封和字段名上不同。
- 三个响应块在调用所在位置不同（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从所有三种响应形状中提取 `{id, name, args}`。

## 交付它

本课产出 `outputs/skill-provider-portability-audit.md`。给定一个针对某一提供商的函数调用集成，该技能产出可移植性审计：它依赖哪些提供商限制，哪些字段需要重命名，以及移植到每个其他提供商时会破坏什么。

## 练习

1. 运行 `code/main.py` 并验证三个提供商声明 JSON 都序列化相同的底层 `Tool` 对象。修改规范工具以添加 enum 参数，并确认只有 Gemini 转换器需要处理 OpenAPI 怪癖。

2. 为每个提供商添加一个 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 没有原生的；注意这种不对称性。

3. 实现 `tool_choice` 转换：将规范 `ToolChoice(mode="force", tool_name="x")` 映射到所有三个提供商形状。然后映射 `mode="any"` 和 `mode="none"`。对照本课的差异表。

4. 选择三个提供商之一，从头到尾阅读其函数调用指南。在其 schema 规范中找出其他两个不支持的一个字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明 schema 的工具调用。通过每个提供商的验证器运行它（第 01 课中的 stdlib 验证器可以作为代理），并记录哪些错误触发。记录你会在生产中为了严格性使用哪个提供商。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| Function calling（函数调用） | "工具使用" | 用于结构化工具调用发出的提供商级 API |
| Tool declaration（工具声明） | "工具规范" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制/禁止" | 自动 / 必需 / 无 / 特定名称模式 |
| Strict mode（严格模式） | "Schema 强制执行" | OpenAI 标志，通过约束解码强制执行 schema |
| `tool_use` 块 | "Anthropic 的调用形状" | 内联内容块，带 id、name、input |
| `functionCall` 部分 | "Gemini 的调用形状" | `parts[]` 条目，包含 name、args 和 id |
| Arguments-as-string（字符串参数） | "字符串化 JSON" | OpenAI 将参数作为 JSON 字符串返回，而非对象 |
| Parallel tool calls（并行工具调用） | "一轮中的扇出" | 一个助手消息中的多个工具调用 |
| Refusal（拒绝） | "模型拒绝" | 严格模式特有的拒绝块，替代调用 |
| OpenAPI 3.0 子集 | "Gemini schema 怪癖" | Gemini 使用 JSON-Schema 类似的方言，有细微差异 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) —— 包括严格模式和并行调用的规范参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) —— `tool_use` 和 `tool_result` 块语义
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) —— 并行调用、唯一 id 和 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) —— Gemini 的企业级表面
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) —— 严格模式 schema 强制执行细节
