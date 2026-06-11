---
name: runtime-picker
description: 根据技术栈、延迟预算和运维形态选择生产级 agent runtime (智能体运行时)：Agno、Mastra、LangGraph 或 provider SDK (提供商 SDK)。
version: 1.0.0
phase: 14
lesson: 18
tags: [agno, mastra, langgraph, runtime, selection]
---

给定技术栈、延迟预算、所需原语和运维形态，选择一个 runtime (运行时)。

决策：

1. Python + FastAPI + 每秒数千个 short-lived agent (短生命周期智能体) -> **Agno**。
2. TypeScript + Next.js/Vercel + unified multi-provider (统一多提供商) -> **Mastra**。
3. Durable state (持久状态)、explicit graph (显式图)、resume-on-failure (失败恢复) -> **LangGraph**（Lesson 13）。
4. Claude-first (Claude 优先) 产品，想要 Claude Code harness (工具链) 形态 -> **Claude Agent SDK**（Lesson 17）。
5. OpenAI-first (OpenAI 优先) 产品，想要 handoff (交接) + guardrail (护栏) + tracing (追踪) -> **OpenAI Agents SDK**（Lesson 16）。
6. Multi-agent team (多智能体团队)、actor-model concurrency (参与者模型并发)、fault isolation (故障隔离) -> **AutoGen v0.4** / **Microsoft Agent Framework**（Lesson 14）。
7. Role-based collaboration (基于角色的协作) 或 event-driven deterministic workflows (事件驱动确定性工作流) -> **CrewAI** Crew 或 Flow（Lesson 15）。
8. 以上皆非 -> 直接 API 调用 + Lesson 01 的 stdlib loop (循环)。

产出：

- 一份简短的决策文档：技术栈、延迟目标、所需原语、观察到的 trade-offs (权衡)。
- 所选运行时的最小 scaffold (脚手架)。
- 如果今天正在使用另一个运行时的迁移计划。

Hard rejects：

- 当工作负载是每个请求一个慢调用时，纯粹因为"性能"选择 Agno 或 Mastra。性能很少是瓶颈。
- 在 Python monorepo (单一代码库) 中没有理由就选择 TypeScript 运行时。混合语言 agent (智能体) 代码是运维负担。
- 为无状态短任务选择 LangGraph。Checkpointer (检查点器) 增加了简单 workflow (工作流)（Lesson 12）可以避免的开销。

Refusal rules：

- 如果用户想要"所有五个运行时，用来对比"，拒绝。在你自己的工作负载上 benchmark (基准测试)；框架厂商的 benchmark (基准测试) 只是方向性的。
- 如果用户想要自托管 Mastra 的 `ee/` 功能，拒绝并指向许可证条款。
- 如果产品需要长时间运行的异步工作（小时到天），拒绝自托管并路由到 Claude Managed Agents (托管智能体) 或基于队列的架构（Lesson 29）。

输出：decision doc (决策文档) + scaffold (脚手架) + README。结尾附 "what to read next"，指向 Lesson 24（可观测性）和 Lesson 29（生产运行时），了解框架之上的运维层。
