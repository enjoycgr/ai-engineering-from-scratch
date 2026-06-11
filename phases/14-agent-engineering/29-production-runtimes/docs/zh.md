# Production Runtimes: Queue, Event, Cron（生产运行时：队列、事件、定时任务）

> 生产智能体以六种运行时形态运行：request-response（请求-响应）、streaming（流式）、durable execution（持久执行）、queue-based background（基于队列的后台）、event-driven（事件驱动）和 scheduled（定时调度）。在选择框架之前先选择形态。可观测性（Observability）在每种形态中都是承重结构。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 分钟

## Learning Objectives

- 说出六种生产运行时形态，并将每种匹配到一个框架 / 产品模式。
- 解释为什么 durable execution（LangGraph）对长程任务至关重要。
- 描述事件驱动运行时以及 Claude Managed Agents 的适用场景。
- 解释多步智能体的"可观测性即承重结构"主张。

## The Problem

生产智能体会以 Jupyter notebook 不会暴露的方式失败：第 37 步的网络超时、用户在中途挂断语音通话、cron 作业在机器重启时死亡、后台工作者内存耗尽。运行时形态决定了哪些故障是可生存的。

## The Concept

### Request-response（请求-响应）

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30 秒）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel spans。

### Streaming（流式）

- SSE 或 WebSocket 用于渐进式输出。
- LiveKit 将其扩展到 WebRTC 用于语音/视频（Lesson 22）。
- 技术栈：任何支持流式的框架 + 处理 SSE/WS 的前端。
- 可观测性：每块时间、首 token 延迟、尾延迟。

### Durable execution（持久执行）

- 每一步后状态检查点（checkpoint）；故障时自动恢复。
- AutoGen v0.4  actor 模型将故障隔离到单个智能体（Lesson 14）。
- LangGraph 的核心差异化（Lesson 13）。
- 当步数未知且恢复成本高昂时必不可少。

### Queue-based / background（基于队列 / 后台）

- 作业进入队列，工作者拾取，结果通过 webhooks 或 pub/sub 回流。
- 对长程智能体至关重要（每个任务数十到数百步，据 Anthropic 的 computer use 公告）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义。
- 可观测性：队列深度、每作业延迟分布、DLQ 大小。

### Event-driven（事件驱动）

- 智能体订阅触发器：新邮件、PR 打开、cron 触发。
- Claude Managed Agents 开箱即用地覆盖此场景（Lesson 17）。
- CrewAI Flows（Lesson 15）构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动延迟、智能体延迟。

### Scheduled（定时调度）

- Cron 形态的智能体定期运行。
- 与 durable execution 结合，使失败的夜间运行在下一次 tick 恢复。
- 技术栈：Kubernetes CronJob + durable 框架；托管（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** 用于事件驱动生产。
- **Agno** 无状态 FastAPI 用于 Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）用于嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用于托管语音（Lesson 22）。
- **Claude Managed Agents** 用于托管长程异步。

### 可观测性是承重结构

没有 OpenTelemetry GenAI spans（Lesson 23）加上 Langfuse/Phoenix/Opik 后端（Lesson 24），你无法调试在第 40 步失败的多步智能体。这在生产中不是可选的。它是"我们快速调试"和"我们用更多日志从头重放"之间的区别。

### 生产运行时的常见失效点

- **错误的形态选择。** 为 5 分钟任务选择 request-response。用户挂断；工作者堆积；重试加剧。
- **没有 DLQ。** 没有死信队列的队列工作者。失败的作业消失。
- **不透明的后台工作。** 没有导出追踪的后台智能体运行。故障 invisible 直到用户报告。
- **跳过 durable state。** 任何超过 30 秒且无法承受重启的运行都需要持久执行。

## Build It

`code/main.py` 是一个 stdlib 多形态演示：

- Request-response 端点（纯函数）。
- Streaming 处理器（生成器）。
- 带 DLQ 的基于队列的工作者。
- 事件触发注册表。
- Cron 形态调度器。

运行方式：

```bash
python3 code/main.py
```

输出：同一任务上五种形态的轨迹。相同的智能体逻辑，不同的外壳。Durable execution（第六种形态）有意在 Lesson 13 的 LangGraph checkpointing 中涵盖。

## Use It

- **Request-response** 用于聊天式 UX。
- **Streaming** 用于渐进式响应。
- **Durable** 用于长程任务。
- **Queue** 用于批处理 / 异步 / 长运行。
- **Event** 用于智能体反应性。
- **Cron** 用于内务（记忆整合、评估、成本报告）。

## Ship It

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并连接可观测性需求。

## Exercises

1. 将 Lesson 01 的 ReAct 循环移植到你技术栈的全部六种形态。哪种形态适合哪种产品表面？
2. 给基于队列的演示添加 DLQ。模拟 10% 作业失败；展示 DLQ 大小。
3. 编写一个 cron 触发的评估智能体，每晚针对当天的前 20 条轨迹运行。
4. 实现带背压的流式：如果客户端慢，暂停智能体。这与轮次预算如何交互？
5. 阅读 Claude Managed Agents 文档。何时你会将自托管的长程智能体迁移到托管？

## Key Terms

| 术语 | 行业说法 | 实际含义 |
|------|----------|----------|
| Request-response | "同步" | 用户等待；仅短任务 |
| Streaming | "SSE / WS" | 渐进式输出；更好的 UX；每块延迟可观测 |
| Durable execution | "故障恢复" | 检查点状态；在最后一步重启 |
| Queue-based | "后台作业" | 生产者 / 工作者池 / DLQ |
| Event-driven | "触发式" | 智能体响应外部事件 |
| DLQ | "死信队列" | 失败作业的停车场 |
| Claude Managed Agents | "托管载体" | Anthropic 托管的长程异步，带缓存 + 压缩 |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution 细节
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长程异步
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每个任务数十到数百步"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model 故障隔离
