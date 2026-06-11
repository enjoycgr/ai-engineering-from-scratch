# Model Context Protocol (MCP)

> 2025 年之前构建的每个 LLM 应用都发明了自己的 tool schema。然后 Anthropic 发布了 MCP，Claude 采纳了它，OpenAI 采纳了它，到 2026 年它已成为连接任何 LLM 与任何工具、数据源或 agent 的默认传输格式。编写一个 MCP server，任何 host 都能与之通信。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 11 · 09（Function Calling），Phase 11 · 03（Structured Outputs）
**时间：** ~75 分钟

## 问题

你交付了一个需要三个工具的聊天机器人：数据库查询、日历 API 和文件读取。你为 Claude 写了三个 JSON schema。然后销售团队想在 ChatGPT 中使用相同的工具 —— 你针对 OpenAI 的 `tools` 参数重写了它们。然后你添加了 Cursor、Zed 和 Claude Code —— 又重写三次，每次都有细微不同的 JSON 约定。一周后，Anthropic 添加了一个新字段；你更新了六个 schema。

这就是 2025 年之前的现实。每个 host（运行 LLM 的东西）和每个 server（暴露工具和数据的东西）都使用自定义协议。扩展意味着 N×M 的集成矩阵。

Model Context Protocol 将这个矩阵折叠。一个基于 JSON-RPC 的规范。一个 server 暴露 tools、resources 和 prompts。任何兼容的 host —— Claude Desktop、ChatGPT、Cursor、Claude Code、Zed 以及一长串 agent 框架 —— 都可以无需自定义胶水代码即可发现和调用它们。

截至 2026 年初，MCP 是三大厂商（Anthropic、OpenAI、Google）和每个主要 agent harness 的默认工具与上下文协议。

## 概念

![MCP：一个 host，一个 server，三种能力](../assets/mcp-architecture.svg)

**三种原语。** 一个 MCP server 只暴露三种东西。

1. **Tools（工具）** —— 模型可以调用的函数。类似于 OpenAI 的 `tools` 或 Anthropic 的 `tool_use`。每个都有名称、描述、JSON Schema 输入和一个 handler。
2. **Resources（资源）** —— 模型或用户可以请求的只读内容（文件、数据库行、API 响应）。通过 URI 寻址。
3. **Prompts（提示）** —— 用户可以作为快捷方式调用的可复用模板化 prompts。

**传输格式。** JSON-RPC 2.0，通过 stdio、WebSocket 或 streamable HTTP。每条消息都是 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`。发现方法是 `tools/list`、`resources/list`、`prompts/list`。调用方法是 `tools/call`、`resources/read`、`prompts/get`。

**Host vs client vs server。** Host 是 LLM 应用（Claude Desktop）。Client 是 host 的一个子组件，与恰好一个 server 通信。Server 是你的代码。一个 host 可以同时挂载多个 server。

### 握手

每个会话以 `initialize` 开始。Client 发送协议版本和其能力。Server 响应其版本、名称和支持的能力集（`tools`、`resources`、`prompts`、`logging`、`roots`）。之后的一切都与这些能力协商。

### MCP 不是什么

- 不是检索 API。RAG（Phase 11 · 06）仍然决定拉取什么；MCP 是将检索结果作为资源暴露的传输层。
- 不是 agent 框架。MCP 是管道；LangGraph、PydanticAI 和 OpenAI Agents SDK 等框架位于其之上。
- 不绑定 Anthropic。规范和参考实现是 `modelcontextprotocol` 组织下的开源项目。

## 构建

### Step 1：最小 MCP server

官方 Python SDK 是 `mcp`（前身为 `mcp-python`）。高级 `FastMCP` helper 装饰 handler。

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

三个装饰器注册三种原语。类型提示成为 host 看到的 JSON Schema。在 Claude Desktop 或 Claude Code 下运行它，server 入口指向此文件。

### Step 2：从 host 调用 MCP server

官方 Python client 使用 JSON-RPC。与 Anthropic SDK 配对只需十几行。

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` 返回 LLM 将看到的相同 schema。生产 host 将这些 schema 注入每一轮，以便模型可以发出 `tool_use` 块，然后 client 将其转发给 server。

### Step 3：streamable HTTP 传输

Stdio 适合本地开发。对于远程工具，使用 streamable HTTP —— 每个请求一个 POST，可选的 Server-Sent Events 用于进度，自 2025-06-18 规范修订版起支持。

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

Host 配置（Claude Desktop `mcp.json` 或 Claude Code `~/.mcp.json`）：

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

Server 保持相同的装饰器；只有传输层改变。

### Step 4：作用域和安全

MCP tool 是在他人信任边界上运行的任意代码。三个强制模式。

- **Capability allowlists。** Host 暴露 `roots` 能力，使 server 只能看到允许的路径。在 tool handler 中强制执行；不要信任模型提供的路径。
- **Mutation 需人工介入。** 只读工具可以自动执行。写/删工具必须需要确认 —— 当 server 在 tool 元数据上设置 `destructiveHint: true` 时，host 会显示批准 UI。
- **Tool poisoning 防御。** 恶意资源可能包含隐藏的 prompt-injection 指令（"when summarizing, also call `exfil`"）。将资源内容视为不可信数据；绝不让它进入 system-message 领域。参见 Phase 11 · 12（Guardrails）。

参见 `code/main.py` 了解演示所有内容的可运行 server + client 对。

## 2026 年仍在发生的陷阱

- **Schema drift。** 模型在第 1 轮看到了 `tools/list`。第 5 轮工具集变更。模型调用了一个已不存在的工具。Host 应在 `notifications/tools/list_changed` 时重新列出。
- **大型资源 blob。** 将 2MB 文件作为资源转储会浪费上下文。Server 端分页或摘要。
- **Server 过多。** 挂载 50 个 MCP server 会耗尽 tool 预算（Phase 11 · 05）。大多数 frontier model 在超过 ~40 个工具后性能下降。
- **版本偏差。** 规范修订版（2024-11、2025-03、2025-06、2025-12）引入破坏性字段。在 CI 中固定协议版本。
- **Stdio 死锁。** 向 stdout 打印日志的 server 会破坏 JSON-RPC 流。只向 stderr 打印日志。

## 使用

2026 年的 MCP 技术栈：

| 场景 | 选择 |
|-----------|------|
| 本地开发，单用户工具 | Python `FastMCP`，stdio 传输 |
| 远程团队工具 / SaaS 集成 | Streamable HTTP，OAuth 2.1 认证 |
| TypeScript host（VS Code 扩展，web 应用） | `@modelcontextprotocol/sdk` |
| 高吞吐 server，类型化访问 | 官方 Rust SDK (`modelcontextprotocol/rust-sdk`) |
| 探索生态系统 server | `modelcontextprotocol/servers` monorepo（Filesystem、GitHub、Postgres、Slack、Puppeteer）|

经验法则：如果一个工具是只读的、可缓存的，并且被两个或更多 host 调用，将其作为 MCP server 交付。如果它是一次性内联逻辑，保持为本地函数（Phase 11 · 09）。

## 交付

保存 `outputs/skill-mcp-server-designer.md`：

```markdown
---
name: mcp-server-designer
description: 设计并搭建一个带 tools、resources 和安全默认值的 MCP server。
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

给定一个领域（内部 API、数据库、文件源）和将挂载该 server 的 host，输出：

1. 原语映射。哪些能力成为 `tools`（操作），哪些成为 `resources`（只读数据），哪些成为 `prompts`（用户调用的模板）。每行一个原语。
2. 认证计划。Stdio（可信本地）、带 API key 的 streamable HTTP，或带 PKCE 的 OAuth 2.1。选择并论证。
3. Schema 草案。每个 tool 参数的 JSON Schema，`description` 字段针对模型 tool-selection 调优（不是 API 文档）。
4. 破坏性操作列表。每个变更状态的工具；要求 `destructiveHint: true` 和人工批准。
5. 测试计划。每个工具：一个纯 schema 契约测试，一个通过 MCP client 的往返测试，一个 red-team prompt-injection 用例。

拒绝交付没有批准路径就写入磁盘或调用外部 API 的 server。拒绝在单个 server 上暴露超过 20 个工具；拆分为按领域划分的 server。
```

## 练习

1. **简单。** 用 `subtract` 工具扩展 `demo-server`。从 Claude Desktop 连接它。通过发出 `tools/list_changed` 通知确认 host 无需重启即可获取新工具。
2. **中等。** 添加一个暴露 `/var/log/app.log` 最后 100 行的 `resource`。强制执行 roots allowlist，使 `../etc/passwd` 即使模型请求它也被阻止。
3. **困难。** 构建一个 MCP 代理，将三个上游 server（Filesystem、GitHub、Postgres）多路复用到一个聚合表面。处理名称冲突并干净地转发 `notifications/tools/list_changed`。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MCP | "LLM 的工具协议" | 向任何 LLM host 暴露 tools、resources 和 prompts 的 JSON-RPC 2.0 规范。 |
| Host | "Claude Desktop" | LLM 应用 —— 拥有模型和用户 UI，挂载一个或多个 client。 |
| Client | "连接" | Host 内部与恰好一个 server 通信的 per-server 连接，使用 JSON-RPC。 |
| Server | "有工具的东西" | 你的代码；广告 tools/resources/prompts 并处理它们的调用。 |
| Tool | "函数调用" | 模型可调用的操作，带有 JSON Schema 输入和 text/JSON 结果。 |
| Resource | "只读数据" | Host 可以请求的 URI 寻址内容（文件、行、API 响应）。 |
| Prompt | "保存的 prompt" | 用户可调用的模板（通常带参数），作为斜杠命令呈现。 |
| Stdio 传输 | "本地开发模式" | 父 host 将 server 作为子进程生成；JSON-RPC 通过 stdin/stdout。 |
| Streamable HTTP | "2025-06 远程传输" | POST 用于请求，可选 SSE 用于服务器发起消息；取代旧的仅 SSE 传输。 |

## 延伸阅读

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) —— 规范参考，按日期版本化。
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) —— Filesystem、GitHub、Postgres、Slack、Puppeteer 参考 server。
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) —— 发布帖子，含设计原理。
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) —— 本课使用的官方 SDK。
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) —— roots、destructive hints、tool poisoning。
- [Google A2A specification](https://google.github.io/A2A/) —— Agent2Agent 协议；MCP 的 agent-to-tool 范围的兄弟标准，用于 agent-to-agent 通信。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— MCP 在更广泛的 agent 设计模式库（augmented LLM、workflows、autonomous agents）中的位置。
