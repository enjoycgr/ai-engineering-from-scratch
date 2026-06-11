# Claude Agent SDK: Subagents and Session Store

> Claude Agent SDK 是 Claude Code harness (工具链) 的库形式。内置工具、用于上下文隔离的 subagent (子智能体)、hooks (钩子)、W3C trace propagation (追踪传播)、session store (会话存储) 对等。Claude Managed Agents 是托管替代方案，用于长时间运行的异步工作。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 分钟

## Learning Objectives

- 解释 Anthropic Client SDK（原始 API）与 Claude Agent SDK（harness (工具链) 形态）的区别。
- 描述 subagent (子智能体)——parallelization (并行化) 和 context isolation (上下文隔离)——以及何时使用它们。
- 说出 Python SDK 的 session store (会话存储) 接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的作用。
- 用标准库实现一个 harness (工具链)，包含内置工具、带隔离上下文的 subagent (子智能体) 生成、lifecycle hooks (生命周期钩子) 和 session store (会话存储)。

## The Problem

原始 LLM (Large Language Model，大语言模型) API 只能给你一次 round-trip (往返)。生产级 agent (智能体) 需要工具执行、MCP (Model Context Protocol，模型上下文协议) 服务器、lifecycle hooks (生命周期钩子)、subagent (子智能体) 生成、session persistence (会话持久化)、trace propagation (追踪传播)。Claude Agent SDK 将这个形态作为库交付——与 Claude Code 使用的 harness (工具链) 相同，但暴露给自定义 agent (智能体)。

## The Concept

### Client SDK vs Agent SDK

- **Client SDK (`anthropic`)。** 原始 Messages API。你自己管理 loop (循环)、工具、状态。
- **Agent SDK (`claude-agent-sdk`)。** 内置工具执行、MCP (模型上下文协议) 连接、hooks (钩子)、subagent (子智能体) 生成、session store (会话存储)。Claude Code loop (循环) 作为库。

### 内置工具

SDK 开箱即用提供 10+ 工具：文件读写、shell、grep、glob、web fetch 等。自定义工具通过标准 tool-schema (工具模式) 接口注册。

### Subagents (子智能体)

Anthropic 文档记载的两种用途：

1. **Parallelization (并行化)。** 并发运行独立工作。"为这 20 个模块各自找到测试文件"是 20 个并行的 subagent (子智能体) 任务。
2. **Context isolation (上下文隔离)。** Subagent (子智能体) 使用自己的 context window (上下文窗口)；只有结果返回给 orchestrator (编排器)。Orchestrator (编排器) 的预算得以保留。

Python SDK 最近新增：`list_subagents()`、`get_subagent_messages()` 用于读取 subagent (子智能体) 转录。

### Session store (会话存储)

与 TypeScript 的协议对等：

- `append(session_id, message)` —— 添加一轮对话。
- `load(session_id)` —— 恢复对话。
- `list_sessions()` —— 枚举。
- `delete(session_id)` —— 级联删除 subagent (子智能体) 会话。
- `list_subkeys(session_id)` —— 列出 subagent (子智能体) 键。

`--session-mirror`（CLI 标志）将 transcript (转录) 在流式传输时镜像到外部文件，用于调试。

### Hooks (钩子)

可注册的生命周期钩子：

- `PreToolUse`、`PostToolUse` —— 门控或审计工具调用。
- `SessionStart`、`SessionEnd` —— 设置和清理。
- `UserPromptSubmit` —— 在模型看到用户输入之前对其采取行动。
- `PreCompact` —— 在 context compaction (上下文压缩) 之前运行。
- `Stop` —— agent (智能体) 退出时清理。
- `Notification` —— 侧通道警报。

Hooks (钩子) 是 pro-workflow（Phase 14 课程参考）等系统添加 cross-cutting behavior (横切行为) 的方式。

### W3C trace context (W3C 追踪上下文)

调用方上活动的 OTel (OpenTelemetry) span (跨度) 通过 W3C trace context headers (追踪上下文头部) 传播到 CLI 子进程。整个多进程追踪在你的后端中显示为一个 trace (追踪)。

### Claude Managed Agents

托管替代方案（beta header `managed-agents-2026-04-01`）。长时间运行的异步工作、内置 prompt caching (提示词缓存)、内置 compaction (压缩)。以控制权换取托管基础设施。

### 这个模式何时会出错

- **Subagent over-spawn (子智能体过度生成)。** 为 100 个微小任务生成 100 个 subagent (子智能体)。开销占主导。改为批处理。
- **Hook creep (钩子蔓延)。** 每个团队都添加钩子；启动时间膨胀。每季度审查 hooks (钩子)。
- **Session bloat (会话膨胀)。** Session (会话) 累积；体积增长。使用 `list_sessions` + 过期策略。

## Build It

`code/main.py` 用标准库实现了 SDK 形态：

- `Tool`、`ToolRegistry`，内置 `read_file`、`write_file`、`list_dir`。
- `Subagent` —— 私有上下文、隔离运行、返回结果。
- `SessionStore` —— append、load、list、delete、list_subkeys。
- `Hooks` —— `pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 演示：主 agent (智能体) 并行生成 3 个 subagent (子智能体)（每个隔离），聚合结果，持久化 session (会话)。

运行方式：

```
python3 code/main.py
```

追踪显示 subagent (子智能体) 上下文隔离（orchestrator (编排器) 上下文大小保持有界）、hook (钩子) 执行和 session (会话) 持久化。

## Use It

- 对于需要 Claude Code harness (工具链) 形态的 Claude 优先产品，使用 **Claude Agent SDK**。
- 对于托管的长时间运行异步工作，使用 **Claude Managed Agents**。
- 对于 OpenAI 优先的对标产品，使用 **OpenAI Agents SDK**（Lesson 16）。
- 如果你想要 graph-shaped (图形态) 状态机而非 harness (工具链) 形态，使用 **LangGraph + 自定义工具**。

## Ship It

`outputs/skill-claude-agent-scaffold.md` 搭建了一个 Claude Agent SDK 应用，包含 subagent (子智能体)、hooks (钩子)、session store (会话存储)、MCP (模型上下文协议) 服务器连接和 W3C trace propagation (追踪传播)。

## Exercises

1. 添加一个 subagent (子智能体) 生成器，将 20 个任务分批为每组 5 个并行 subagent (子智能体)。测量 orchestrator (编排器) 上下文大小与 one-per-task 的对比。
2. 实现一个 `PreToolUse` hook (钩子)，对 `write_file` 调用进行速率限制（每 session (会话) 每分钟 5 次）。追踪该行为。
3. 将 `list_subkeys` 连接到渲染 subagent (子智能体) 树。深层嵌套看起来是什么样？
4. 将该玩具移植到真正的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。何时你会从自托管切换到托管？

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Agent SDK (智能体 SDK) | "Claude Code as a library" | Harness (工具链) 形态：工具、MCP (模型上下文协议)、hooks (钩子)、subagent (子智能体)、session store (会话存储) |
| Subagent (子智能体) | "Child agent (子智能体)" | 独立上下文，自己的预算；结果向上冒泡 |
| Session store (会话存储) | "Conversation DB (对话数据库)" | 持久化、加载、列出、删除轮次，并级联 subagent (子智能体) |
| Hook (钩子) | "Lifecycle callback (生命周期回调)" | 工具前/后、会话、提示词提交、压缩、停止 |
| W3C trace context (W3C 追踪上下文) | "Cross-process trace (跨进程追踪)" | Parent span (父跨度) 传播到 CLI 子进程 |
| Managed Agents (托管智能体) | "Hosted harness (托管工具链)" | Anthropic 托管的长时间运行异步工作 |
| `--session-mirror` | "Transcript mirror (转录镜像)" | 将 session (会话) 轮次在流式传输时写入外部文件 |
| MCP server (MCP 服务器) | "Tool surface (工具接口)" | 附加到 agent (智能体) 的外部工具/资源源 |

## Further Reading

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code 的库形式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 生产模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 对标产品
