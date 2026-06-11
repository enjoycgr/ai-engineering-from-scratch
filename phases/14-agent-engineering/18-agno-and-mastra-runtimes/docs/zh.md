# Agno and Mastra: Production Runtimes

> Agno（Python）和 Mastra（TypeScript）是 2026 年的生产运行时搭档。Agno 瞄准微秒级 agent (智能体) 实例化和无状态 FastAPI 后端。Mastra 提供 agent (智能体)、工具、workflow (工作流)、unified model routing (统一模型路由) 和 composite storage (复合存储)，构建于 Vercel AI SDK 之上。

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 分钟

## Learning Objectives

- 识别 Agno 的性能目标及其适用场景。
- 说出 Mastra 的三大原语——Agents (智能体)、Tools (工具)、Workflows (工作流)——以及支持的服务器适配器。
- 解释为什么无状态 session-scoped (会话作用域) FastAPI 后端是推荐的 Agno 生产路径。
- 为给定技术栈选择 Agno 或 Mastra（Python 优先 vs TypeScript 优先）。

## The Problem

LangGraph、AutoGen、CrewAI 都是重量级框架。想要"只需要 agent loop (智能体循环)，要快，能在我的运行时里跑"的团队会选择 Agno（Python）或 Mastra（TypeScript）。两者都牺牲了一些框架拥有的原语，换取了原始速度和对周围技术栈的更紧密适配。

## The Concept

### Agno

- Python 运行时，前身为 Phi-data。
- "No graphs, chains, or convoluted patterns — just pure python."（没有图、链或复杂的模式——只有纯 Python。）
- 来自其文档的性能目标：约 2μs agent (智能体) 实例化、约 3.75 KiB 内存 per agent (每个智能体)、约 23 个 model provider (模型提供商)。
- 生产路径：无状态 session-scoped (会话作用域) FastAPI 后端。每个请求启动一个全新的 agent (智能体)；session state (会话状态) 存放在 DB (数据库) 中。
- 原生 multimodal (多模态)（文本、图像、音频、视频、文件）和 agentic RAG (检索增强生成)。

速度目标在每秒有数千个 short-lived agent (短生命周期智能体)（聊天并发、评估流水线）时才有意义。当单个 agent (智能体) 运行 10 分钟时，它们意义不大。

### Mastra

- TypeScript，构建于 Vercel AI SDK 之上。
- 三大原语：**Agents (智能体)**、**Tools (工具)**（Zod-typed，Zod 类型化）、**Workflows (工作流)**。
- Unified Model Router (统一模型路由器)——94 个提供商的 3,300+ 模型（2026 年 3 月）。
- Composite storage (复合存储)：memory (记忆)、workflows (工作流)、observability (可观测性) 各自使用不同的后端；大规模可观测性推荐使用 ClickHouse。
- Apache 2.0 许可证，`ee/` 目录下的功能受 source-available (源码可用) 企业许可证限制。
- 服务器适配器：Express、Hono、Fastify、Koa；原生 Next.js 和 Astro 集成。
- 提供 Mastra Studio（localhost:4111）用于调试。
- 1.0 版本时 22k+ GitHub stars、30万+ 周 npm 下载量（2026 年 1 月）。

### 定位

两者都不想成为 LangGraph。它们在以下方面竞争：

- **Language fit (语言适配)。** Python 优先团队选 Agno；TypeScript 优先团队选 Mastra。
- **Runtime ergonomics (运行时人体工学)。** Agno = 接近零开销；Mastra = 与 Vercel 生态系统集成。
- **Observability (可观测性)。** 两者都集成 Langfuse/Phoenix/Opik（Lesson 24），但 Mastra Studio 是 first-party (第一方) 的。

### 何时选择哪个

- **Agno** —— Python 后端、大量 short-lived agent (短生命周期智能体)、强性能需求、FastAPI 团队。
- **Mastra** —— TypeScript 后端、Next.js / Vercel 部署、unified multi-provider model routing (统一多提供商模型路由)、Zod-typed tools (Zod 类型化工具)。
- **LangGraph**（Lesson 13）—— 当 durable state (持久状态) 和 explicit graph reasoning (显式图推理) 比原始速度更重要时。
- **OpenAI / Claude Agent SDK**（Lessons 16–17）—— 当你想要提供商的产品化形态时。

### 这个模式何时会出错

- **Perf-for-perf's-sake (为性能而性能)。** 当工作负载是每个请求一个缓慢的 agent (智能体) 调用时，因为"2μs"听起来不错就选 Agno。开销不是瓶颈。
- **Ecosystem lock-in (生态系统锁定)。** Mastra 的 Vercel 风格集成在 Vercel 上是加分项，在其他地方是减分项。
- **Enterprise license confusion (企业许可证混淆)。** Mastra 的 `ee/` 目录是 source-available (源码可用) 的，不是 Apache 2.0。如果你计划 fork (分叉)，请阅读许可证。

## Build It

本课主要是比较性的——没有单个代码产物能公正对待两个框架。参见 `code/main.py` 中并排的玩具：将"运行一个 agent (智能体)，流式输出，持久化 session (会话)"的最小流程实现两次（一次 Agno 形态，一次 Mastra 形态）。

运行方式：

```
python3 code/main.py
```

两条结构上不同但功能等价的 trace (追踪)。

## Use It

- **Agno** —— 需要速度和 FastAPI 形态的 Python 后端。
- **Mastra** —— 拥有大量提供商和 workflow (工作流) 原语的 TypeScript 后端。
- 两者都提供 first-party (第一方) 可观测性 hooks (钩子)。两者都集成 Langfuse。

## Ship It

`outputs/skill-runtime-picker.md` 根据技术栈、延迟预算和运维形态选择 Agno、Mastra、LangGraph 或 provider SDK (提供商 SDK)。

## Exercises

1. 阅读 Agno 文档。将 stdlib ReAct loop (Lesson 01) 移植到 Agno。什么消失了？什么留下了？
2. 阅读 Mastra 文档。将同一个 loop (循环) 移植到 Mastra。工具类型化有什么变化（Zod vs 无类型）？
3. Benchmark (基准测试)：在你的技术栈上测量 agent (智能体) 实例化延迟。Agno 的 2μs 对你的工作负载重要吗？
4. 设计一次迁移：如果你一直在 Python 中运行 CrewAI，迁移到 Agno 会破坏什么？
5. 阅读 Mastra 的 `ee/` 许可证条款。开源 fork (分叉) 会受哪些限制？

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Agno | "Fast Python agents (快速 Python 智能体)" | Stateless session-scoped agent runtime (无状态会话作用域智能体运行时) |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents (智能体) + Tools (工具) + Workflows (工作流) + Model Router (模型路由器) |
| Unified Model Router (统一模型路由器) | "Multi-provider access (多提供商访问)" | 94 个提供商的 3,300+ 模型的单一客户端 |
| Composite storage (复合存储) | "Multiple backends (多后端)" | Memory (记忆)/workflows (工作流)/observability (可观测性) 各自使用不同的存储 |
| Mastra Studio | "Local debugger (本地调试器)" | localhost:4111 的 agent (智能体) 内省 UI |
| Source-available (源码可用) | "Not OSS (非开源)" | 许可证允许阅读源码但限制商业使用 |

## Further Reading

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — 性能目标、FastAPI 集成
- [Mastra docs](https://mastra.ai/docs) — 原语、服务器适配器、Model Router (模型路由器)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — 有状态图替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 集成引用的可观测性对比
