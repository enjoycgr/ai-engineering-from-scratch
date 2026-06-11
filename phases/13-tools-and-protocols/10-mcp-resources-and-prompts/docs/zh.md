# MCP 资源和提示 —— 超越工具的上下文暴露

> 工具获得了 90% 的 MCP 关注。另外两个服务器原语解决不同的问题。资源暴露用于读取的数据；提示将可复用模板作为斜杠命令暴露。许多服务器应该使用资源而非将读取包装在工具中，并使用提示而非在客户端提示中硬编码工作流。本课命名了决策规则并走过 `resources/*` 和 `prompts/*` 消息。

**类型：** Build
**语言：** Python（stdlib，资源 + 提示处理器）
**前置要求：** Phase 13 · 07（MCP 服务器）
**时间：** ~45 分钟

## 学习目标

- 决定对于给定域，将能力暴露为工具、资源还是提示。
- 实现 `resources/list`、`resources/read`、`resources/subscribe` 并处理 `notifications/resources/updated`。
- 实现带参数模板的 `prompts/list` 和 `prompts/get`。
- 识别宿主何时将提示作为斜杠命令 vs 自动注入上下文展示。

## 问题

一个笔记应用的朴素 MCP 服务器将所有内容暴露为工具：`notes_read`、`notes_list`、`notes_search`。这将每次数据访问都包装在模型驱动的工具调用中。后果：

- 模型必须决定是否为每个可能受益于上下文的查询调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧边面板。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的 "包含文件" 选择器）无法展示数据。

正确的拆分：将数据暴露为资源，将变更或计算动作暴露为工具，将可复用的多步骤工作流暴露为提示。每个原语都有其 UX  affordance（可用性）和访问模式。

## 概念

### 工具 vs 资源 vs 提示 —— 决策规则

| 能力 | 原语 |
|------------|-----------|
| 用户想要搜索、过滤或转换数据 | tool |
| 用户想要宿主将此数据作为上下文包含 | resource |
| 用户想要一个他们可以重新运行的模板化工作流 | prompt |

指导原则：如果模型会在每个相关查询上受益地调用它，它是工具。如果用户会受益地将其附加到对话，它是资源。如果用户想要复用的整个多步骤工作流是单元，它是提示。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的东西：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义 scheme）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 支持文本和二进制。二进制使用 `blob` 作为 base64 编码的字符串加上 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。服务器在资源变更时发送 `notifications/resources/updated`。客户端重新读取。

用例：一个资源是磁盘上文件的笔记服务器；文件监视器触发更新通知；Claude Desktop 在外部编辑时重新拉取文件到上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 让你暴露参数化 URI 模式：`notes://{id}` 且 `id` 作为补全目标。客户端可以在资源选择器中自动补全 id。

### 提示

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示是一个模板，填充为宿主喂给模型的消息列表。例如，一个 `code_review` 提示接受 `file_path` 参数并返回一个三消息序列：系统消息、带文件主体的用户消息和带推理模板的助手开场白。

### 宿主和提示

Claude Desktop、VS Code 和 Cursor 在聊天 UI 中将提示暴露为斜杠命令。用户输入 `/code_review` 并从表单中挑选参数。服务器的提示是 "用户快捷方式" 和 "发送给模型的完整提示" 之间的契约。

并非每个客户端都支持提示 —— 检查能力协商。声明了提示能力但客户端不支持提示的服务器将看不到斜杠命令。

### "list changed" 通知

资源和提示在集合变更时都发出 `notifications/list_changed`。一个刚导入 20 条新笔记的笔记服务器发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 以获取新增内容。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，加上 `blob` 字段。
MCP Apps（第 14 课）：`text/html;profile=mcp-app` 在 `ui://` URI 中。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须是稳定的。如果计算是一次性的，URI 应包含时间戳或 nonce，以免客户端缓存变 stale。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。不支持订阅的客户端或宿主通过重新读取来轮询。两者都规范合规。服务器的能力声明告诉客户端它支持哪种。

订阅的成本：服务器上的每会话状态（谁在订阅什么）。保持订阅集有界；断开连接的客户端应超时。

### 提示 vs 系统提示

MCP 中的提示不是系统提示。宿主的系统提示（它自己的操作指令）和 MCP 提示（用户调用的服务器提供模板）并存。一个行为良好的客户端永远不会让服务器提示覆盖其自己的系统提示；它将它们分层。

## 使用它

`code/main.py` 扩展了第 07 课的笔记服务器：

- 每个笔记的资源（`notes://note-1` 等）带 `resources/subscribe` 支持。
- 一个 `review_note` 提示，渲染为三消息模板。
- 一个文件监视器模拟，当笔记被修改时发出 `notifications/resources/updated`。
- 一个始终返回最新五条笔记的 `notes://recent` 动态资源。

运行演示以查看完整流程。

## 交付它

本课产出 `outputs/skill-primitive-splitter.md`。给定一个提议的 MCP 服务器，该技能将每个能力分类为工具 / 资源 / 提示并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发笔记编辑并验证 `notifications/resources/updated` 事件触发。

2. 添加一个 `resources/list_changed` 发出器：当新笔记创建时发送通知，以便客户端重新发现。

3. 为一个 GitHub MCP 服务器设计三个提示：`summarize_pr`、`triage_issue`、`release_notes`。每个都有参数 schema。提示主体应可运行而无需进一步编辑。

4. 取第 07 课服务器中的一个现有工具并分类它应该保持为工具还是拆分为资源加工具对。用一句话论证。

5. 阅读规范的 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中很少填充但规范支持的字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| Resource | "暴露的数据" | 宿主可以读取的 URI 可寻址内容 |
| Resource URI | "数据指针" | scheme 前缀标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视变更" | 客户端选择加入的特定 URI 的服务器推送更新 |
| `notifications/resources/updated` | "资源已变更" | 客户端订阅资源有新内容的信号 |
| Resource template | "参数化 URI" | 带宿主选择器补全提示的 URI 模式 |
| Prompt | "斜杠命令模板" | 带参数槽的命名多消息模板 |
| Prompt arguments | "模板输入" | 宿主在渲染前收集的类型化参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充的消息列表 |
| Content block | "类型化块" | `{type: text \| image \| resource \| ui_resource}` |
| 斜杠命令 UX | "用户快捷方式" | 宿主将提示作为以 `/` 开头的命令展示 |

## 延伸阅读

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) —— 资源 URI、订阅和模板
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) —— 提示模板和斜杠命令集成
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) —— 完整的 `resources/*` 消息参考
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) —— 完整的 `prompts/*` 消息参考
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) —— 官方文档的社区扩展指南
