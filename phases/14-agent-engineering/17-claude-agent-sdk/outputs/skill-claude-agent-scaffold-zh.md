---
name: claude-agent-scaffold
description: 搭建 Claude Agent SDK 应用，包含 subagent (子智能体)、lifecycle hooks (生命周期钩子)、session store (会话存储)、MCP (模型上下文协议) 服务器连接和 W3C trace propagation (追踪传播)。
version: 1.0.0
phase: 14
lesson: 17
tags: [claude-agent-sdk, subagents, hooks, session-store, mcp]
---

给定一个产品领域和一组 MCP (模型上下文协议) 服务器，搭建一个 Claude Agent SDK 应用。

产出：

1. 一个主 agent (智能体) 定义，包含 instructions (指令)、内置工具访问（read_file、write_file、shell、grep、glob、web fetch）和自定义函数工具。
2. Subagent (子智能体) 生成器用于 parallelization (并行化) 和 context isolation (上下文隔离)。当 orchestrator (编排器) 否则会耗尽上下文预算时使用。
3. 注册的生命周期 hooks (钩子)：PreToolUse + PostToolUse 用于审计，SessionStart 用于设置，SessionEnd 用于清理，UserPromptSubmit 用于规则执行（参见 pro-workflow 模式）。
4. Session store (会话存储)（默认 SQLite），`list_subkeys` 连接以渲染 subagent (子智能体) 树。
5. MCP (模型上下文协议) 服务器连接用于外部工具/资源接口。
6. W3C trace context propagation (追踪传播) 使调用方的 OTel span (跨度) 延续到 CLI。

Hard rejects：

- 为单个工具任务生成 subagent (子智能体)。Subagent (子智能体) 用于 parallelization (并行化) 或 context isolation (上下文隔离)；不是用于"一次 read_file 调用"。
- 带有同步昂贵工作的 Hooks (钩子)。Hooks (钩子) 应为微秒到毫秒级。长时间工作应放入 subagent (子智能体)。
- 没有 cascade-delete (级联删除) 策略的 Session store (会话存储)。孤立的 subagent (子智能体) 会话会膨胀存储。

Refusal rules：

- 如果产品需要长时间运行的异步工作（小时到天），拒绝自托管 SDK 并路由到 Claude Managed Agents (托管智能体)。
- 如果用户要求 `--session-mirror` 到共享位置，拒绝。Session transcript (会话转录) 携带 PII (个人身份信息)；镜像到按用户加密的存储。
- 如果 agent (智能体) 依赖原始 LLM streaming (流式传输) 作为 UX 而不使用工具，拒绝 Agent SDK 并直接推荐 Client SDK。

输出：`agent.py`、`tools.py`、`hooks.py`、`session.py`、`README.md`，解释 subagent (子智能体) 策略、hook (钩子) 注册表、session backend (会话后端)、MCP (模型上下文协议) 连接和 OTel 接线。结尾附 "what to read next"，指向 Lesson 22（语音 handoff (交接)）、Lesson 23（OTel span (跨度) 归属），或 Lesson 18（如果产品需要生产运行时形态）。
