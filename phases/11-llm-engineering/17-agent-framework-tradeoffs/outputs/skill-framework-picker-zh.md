---
name: framework-picker
description: 通过将抽象与问题形状匹配，为 agent 任务选择 LangGraph、CrewAI、AutoGen、Agno 或纯 Python。
version: 1.0.0
phase: 11
lesson: 17
tags: [langgraph, crewai, autogen, agno, agent-framework, orchestration, decision-matrix]
---

给定任务描述（问题形状、每次运行总 LLM 调用数、分支模式、持久化和恢复需求、human-in-the-loop checkpoints、并行扇出、session memory、预期日运行量），输出：

1. 形状匹配。一句话命名适合的抽象：graph（类型化状态、命名转换）、org chart（专家角色、manager 路由交接）、chat（agent 聊到完）、带工具的单 agent。如果你无法选择一个，任务还不是 agent 形状的；停止并分解。
2. 分支权威。谁选择下一步：developer（explicit edges）、manager LLM（CrewAI hierarchical）、conversational emergent（AutoGen GroupChat）、tool-call self-routed（Agno）。如果适用，引用 LLM-selected routing 的每轮 token 成本。
3. 状态预算。确认是否需要 resume-after-restart、time-travel 或 human interrupts。如果是，LangGraph 在 state-first 抽象上获胜；Agno 仅覆盖 session-scoped memory。
4. 框架选择。输出 langgraph、crewai、autogen、agno、plain_python 之一。包含将形状和状态答案映射到框架核心抽象的一句话理由。
5. 逃生舱。如果日运行量超过 10_000 或任务是两次或更少的 LLM 调用且无状态，推荐纯 Python 加 provider SDK。任务小时，没有框架是最快的框架。

拒绝为具有已知 DAG 的确定性 workflow 推荐 AutoGen；GroupChatManager 花费 tokens 选择 speaker，而 developer 本可以静态连接它们。CrewAI 确实通过 `output_pydantic` / `output_json` 支持结构化任务输出（参见 [docs.crewai.com/en/concepts/tasks](https://docs.crewai.com/en/concepts/tasks)），但其 `context` 通道仍通过下一个任务的 prompt string 流动。当 workflow 依赖原始 `context` 在没有这些 output schema 之一连接的情况下跨任务传递结构化状态时，反对 CrewAI。反对 LangGraph 用于 two-call summarizer；StateGraph 开销是纯税。反对 Agno 用于任务在超过 4 个并行 sub-workers 上扇出且需要 reducer 语义的情况；Agno 提供了一个 `Parallel` 块，其输出按步骤名称 join 成 dict（参见 [docs-v1.agno.com/workflows_2/overview](https://docs-v1.agno.com/workflows_2/overview) 和 [docs.agno.com/workflows/access-previous-steps](https://docs.agno.com/workflows/access-previous-steps)），但它没有暴露与 LangGraph 相当的 Send-style fanout-and-reduce API。

示例输入："Long-running research workflow: plan, fan out to three retrievers, synthesize, human approves brief, write report, cite sources. Must resume after crash. Production-bound to 50 runs per day."

示例输出：
- Shape: graph. Typed plan, three parallel retrievers, named transitions between synthesize and write.
- Branching: developer-decided via conditional edges. No per-turn manager LLM.
- State: requires resume and human interrupt. LangGraph mandatory.
- Framework: langgraph. State, Send fanout, interrupt_before, and PostgresSaver are all first-class.
- Escape hatch: not applicable. 50 runs per day is well below the plain-Python threshold and the workflow is too stateful to leave unframeworked.
