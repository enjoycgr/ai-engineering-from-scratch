# MCP 基础 —— 原语、生命周期、JSON-RPC 基础

> MCP 之前的每个集成都是一次性的。Model Context Protocol（模型上下文协议）由 Anthropic 于 2024 年 11 月首次推出，现由 Linux Foundation 的 Agentic AI Foundation 管理，它标准化了发现和调用，使任何客户端都能与任何服务器对话。2025-11-25 规范命名了六种原语（服务器三种，客户端三种）、一个三阶段生命周期和一个 JSON-RPC 2.0 线路格式。学会这些，本阶段 MCP 章节的其余部分就变成了阅读。

**类型：** Learn
**语言：** Python（stdlib，JSON-RPC 解析器）
**前置要求：** Phase 13 · 01 到 05（工具接口和函数调用）
**时间：** ~45 分钟

## 学习目标

- 命名所有六种 MCP 原语（服务器上的 tools、resources、prompts；客户端上的 roots、sampling、elicitation）并为每种给出一个用例。
- 走过三阶段生命周期（initialize、operation、shutdown）并说明每个阶段谁发送哪个消息。
- 解析和发出 JSON-RPC 2.0 请求、响应和通知信封。
- 解释 `initialize` 时的能力协商是什么，以及没有它会发生什么。

## 问题

MCP 之前，每个使用工具的 Agent 都有自己的协议。Cursor 有一个 MCP 形状但不兼容的工具系统。Claude Desktop 推出了另一个不同的。VS Code 的 Copilot 扩展有第三个。一个构建 "Postgres 查询" 工具的团队写了三次相同的工具，每次针对不同的宿主 API。复用它需要复制代码。

结果是一次性集成的寒武纪大爆发和生态系统速度的天花板。

MCP 通过标准化线路格式修复了这一点。单个 MCP 服务器在每个 MCP 客户端中工作：Claude Desktop、ChatGPT、Cursor、VS Code、Gemini、Goose、Zed、Windsurf，到 2026 年 4 月已有 300+ 客户端。每月 1.1 亿 SDK 下载量。1 万+ 公共服务器。Linux Foundation 于 2025 年 12 月在新的 Agentic AI Foundation 下接管了管理权。

本阶段使用的 spec 修订版是 **2025-11-25**。它添加了异步 Tasks（SEP-1686）、URL 模式 elicitation（SEP-1036）、带工具的 sampling（SEP-1577）、增量 scope consent（SEP-835）和 OAuth 2.1 资源指示器语义。Phase 13 · 09 到 16 涵盖这些扩展。本课止步于基础。

## 概念

### 三种服务器原语

1. **Tools（工具）。** 可调用动作。与 Phase 13 · 01 相同的四步循环。
2. **Resources（资源）。** 暴露的数据。通过 URI 可寻址的只读内容：`file:///path`、`db://query/...`、自定义 scheme。
3. **Prompts（提示）。** 可复用模板。宿主 UI 中的斜杠命令；服务器提供模板，客户端填充参数。

### 三种客户端原语

4. **Roots（根）。** 服务器允许触碰的 URI 集合。客户端声明它们；服务器遵守它们。
5. **Sampling（采样）。** 服务器请求客户端的模型执行补全。使服务器托管的 Agent 循环无需服务器端 API 密钥。
6. **Elicitation（引出）。** 服务器在飞行中向客户端用户请求结构化输入。表单或 URL（SEP-1036）。

MCP 中的每个能力恰好属于这六种之一。Phase 13 · 10 到 14 深入讲解每个。

### 线路格式：JSON-RPC 2.0

每条消息都是一个具有以下字段的 JSON 对象：

- 请求：`{jsonrpc: "2.0", id, method, params}`。
- 响应：`{jsonrpc: "2.0", id, result | error}`。
- 通知：`{jsonrpc: "2.0", method, params}` —— 无 `id`，不期望响应。

基础规范有 ~15 个方法，按原语分组。重要的有：

- `initialize` / `initialized`（握手）
- `tools/list`、`tools/call`
- `resources/list`、`resources/read`、`resources/subscribe`
- `prompts/list`、`prompts/get`
- `sampling/createMessage`（服务器到客户端）
- `notifications/tools/list_changed`、`notifications/resources/updated`、`notifications/progress`

### 三阶段生命周期

**第一阶段：initialize。**

客户端发送带有其 `capabilities` 和 `clientInfo` 的 `initialize`。服务器以其自己的 `capabilities`、`serverInfo` 和它所说的 spec 版本响应。客户端发送 `notifications/initialized` 表示已消化响应。从此，任何一方都可以按协商的能力发送请求。

**第二阶段：operation。**

双向。客户端调用 `tools/list` 来发现，然后调用 `tools/call` 来调用。如果服务器声明了该能力，它可能发送 `sampling/createMessage`。当工具集变更时，服务器可能发送 `notifications/tools/list_changed`。当用户更改根 scope 时，客户端可能发送 `notifications/roots/list_changed`。

**第三阶段：shutdown。**

任一方关闭传输。MCP 中没有结构化 shutdown 方法；传输（stdio 或 Streamable HTTP，Phase 13 · 09）携带连接结束信号。

### 能力协商

`initialize` 握手时交换的 `capabilities` 是契约。来自服务器的示例：

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

服务器声明它可以发出 `tools/list_changed` 通知并支持 `resources/subscribe`。客户端通过声明自己的能力来同意：

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

如果客户端未声明 `sampling`，服务器不得调用 `sampling/createMessage`。对称地：如果服务器未声明 `resources.subscribe`，客户端不得尝试订阅。

这就是防止生态系统漂移的原因。不支持 sampling 的客户端仍然是有效的 MCP 客户端；不调用 `sampling` 的服务器仍然是有效的 MCP 服务器。它们只是不一起使用该功能。

### 结构化内容和错误形状

`tools/call` 返回一个 `content` 数组，包含类型化块：`text`、`image`、`resource`。Phase 13 · 14 向该列表添加 MCP Apps（`ui://` 交互式 UI）。

错误使用 JSON-RPC 错误代码。规范定义的补充：`-32002` "Resource not found"、`-32603` "Internal error"，加上 MCP 特定的错误数据作为 `error.data`。

### 客户端能力 vs 工具调用细节

一个常见混淆：`capabilities.tools` 是客户端是否支持工具列表变更通知。客户端是否 WILL 调用特定工具是由其模型驱动的运行时选择，不是能力标志。能力标志是 spec 级别的契约。模型的选择是正交的。

### 为什么是 JSON-RPC 而非 REST？

JSON-RPC 2.0（2010）是一个轻量级双向协议。REST 是客户端发起的。MCP 需要服务器发起的消息（sampling、通知），因此 JSON-RPC 及其对称的请求/响应形状是自然的契合。JSON-RPC 也能干净地组合在 stdio 和 WebSocket/Streamable HTTP 之上，无需重新发明 HTTP 的请求形状。

## 使用它

`code/main.py` 交付了一个最小的 JSON-RPC 2.0 解析器和发出器，然后手工走过 `initialize` → `tools/list` → `tools/call` → `shutdown` 序列，打印每条消息。无真实传输；只有消息形状。与延伸阅读中链接的 spec 对比，以验证每个信封。

看点：

- `initialize` 双向声明能力；响应中有 `serverInfo` 和 `protocolVersion: "2025-11-25"`。
- `tools/list` 返回一个 `tools` 数组；每个条目有 `name`、`description`、`inputSchema`。
- `tools/call` 使用 `params.name` 和 `params.arguments`。
- 响应 `content` 是一个 `{type, text}` 块的数组。

## 交付它

本课产出 `outputs/skill-mcp-handshake-tracer.md`。给定一个 MCP 客户端-服务器交互的 pcap 风格转录，该技能为每条消息标注它属于哪个原语、哪个生命周期阶段以及它依赖哪个能力。

## 练习

1. 运行 `code/main.py`。确定能力协商发生的行，并描述如果服务器未声明 `tools.listChanged` 会发生什么变化。

2. 扩展解析器以处理 `notifications/progress`。消息形状：`{method: "notifications/progress", params: {progressToken, progress, total}}`。在长时运行的 `tools/call` 期间发出它，并确认客户端处理器会显示进度条。

3. 从头到尾阅读 MCP 2025-11-25 规范——整个文档约 80 页。找出大多数服务器不需要的一个能力标志。提示：它与资源订阅有关。

4. 在纸上草拟一个假设的 "cron job" 功能属于哪个原语。（提示：服务器希望客户端在计划时间调用它。六种原语都不适合。）MCP 的 2026 年路线图对此有一个草案 SEP。

5. 从 GitHub 上的一个开放 MCP 服务器解析一个会话日志。统计请求 vs 响应 vs 通知消息。计算生命周期流量占操作流量的比例。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| MCP | "Model Context Protocol" | 模型到工具发现和调用的开放协议 |
| Server primitive（服务器原语） | "服务器暴露什么" | tools（动作）、resources（数据）、prompts（模板） |
| Client primitive（客户端原语） | "客户端让服务器使用什么" | roots（scope）、sampling（LLM 回调）、elicitation（用户输入） |
| JSON-RPC 2.0 | "线路格式" | 对称的请求/响应/通知信封 |
| `initialize` 握手 | "能力协商" | 第一条消息对；服务器和客户端声明它们支持的功能 |
| `tools/list` | "发现" | 客户端向服务器询问其当前工具集 |
| `tools/call` | "调用" | 客户端请求服务器用参数执行工具 |
| `notifications/*_changed` | "变更事件" | 服务器告诉客户端其原语列表已更改 |
| Content block（内容块） | "类型化结果" | 工具结果中的 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "Spec Evolution Proposal" | 命名草案提案（例如 SEP-1686 表示异步 Tasks） |

## 延伸阅读

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— 规范文档
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) —— 六种原语的心智模型
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) —— 2024 年 11 月发布文章
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) —— 一周年回顾和 2025-11-25 规范变更
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) —— SEP-1686、1036、1577、835 和 1724 的摘要
