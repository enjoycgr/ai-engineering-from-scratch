# AutoGen v0.4: Actor 模型与智能体框架

> AutoGen v0.4 (Microsoft Research，2025 年 1 月) 围绕 actor model (Actor 模型) 重新设计了 agent orchestration (智能体编排)。Async message exchange (异步消息交换)、event-driven agents (事件驱动智能体)、fault isolation (故障隔离)、natural concurrency (天然并发)。该框架目前处于维护模式，而 Microsoft Agent Framework (2025 年 10 月公开预览) 成为继任者。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 分钟

## 学习目标

- 描述 actor model (Actor 模型)：agents 作为 actors (角色)、messages (消息) 作为唯一的 IPC (进程间通信)、每个 actor 的 failure isolation (故障隔离)。
- 说出 AutoGen v0.4 的三个 API 层 —— Core、AgentChat、Extensions —— 以及各自的用途。
- 解释为什么将 message delivery (消息投递) 与 handling (处理) 解耦能带来 fault isolation 和 natural concurrency。
- 在 Python 标准库中实现一个 actor runtime (Actor 运行时)，并将一个 two-agent code-review flow (双智能体代码审查流程) 移植到其上。

## 问题背景

大多数 agent frameworks (智能体框架) 是 synchronous (同步的)：一个 agent 生产，一个 agent 消费，在同一个 call stack (调用栈) 中。失败会 crash (崩溃) 整个栈。Concurrency (并发) 是后加上的。Distribution (分布式) 需要重写。

AutoGen v0.4 的答案是：actor model。每个 agent 是一个带有 private inbox (私有收件箱) 的 actor。Messages 是唯一的交互方式。Runtime 将 delivery 与 handling 解耦。失败隔离到单个 actor。Concurrency 是原生的。Distribution 只是不同的 transport (传输层)。

## 核心概念

### Actors (角色)

一个 actor 拥有：

- Private state (私有状态)（永远不会被外部直接触碰）。
- Inbox (收件箱)（消息队列）。
- Handler (处理器)：`receive(message) -> effects`，其中 effects 可以是 "reply"、"send to other actor"、"spawn new actor"、"update state"、"stop self"。

两个 actors 不能共享 memory (内存)。它们只能发送 messages。

### AutoGen v0.4 的三个 API 层

1. **Core。** 底层 actor framework。`AgentRuntime`、`Agent`、`Message`、`Topic`。Async message exchange，event-driven。
2. **AgentChat。** Task-driven high-level API（v0.2 的 ConversableAgent 的替代品）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** Integrations (集成) —— OpenAI、Anthropic、Azure、tools、memory。

### 为什么解耦很重要

在 v0.2 模型中，同步调用 `agent_a.chat(agent_b)` 会阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的 inbox 并立即返回。Runtime 稍后投递。三个后果：

- **Fault isolation (故障隔离)。** Agent B 崩溃不会 crash Agent A —— runtime 在 B 的 handler 中捕获失败，并决定做什么（log、retry、dead-letter）。
- **Natural concurrency (天然并发)。** 多条消息同时在途；actors 并发处理它们的 inbox。
- **Distribution-ready (就绪于分布式)。** Inbox + transport 是同一个抽象，无论 actor 是 in-process (进程内) 还是另一个 host (主机) 上。

### 拓扑

- **RoundRobinGroupChat。** Agents 按固定轮换顺序轮流发言。
- **SelectorGroupChat。** 一个 selector agent 基于对话上下文选择下一个发言者。
- **Magentic-One。** 用于 web browsing、code execution、file handling 的参考 multi-agent team (多智能体团队)。基于 AgentChat 构建。

### 可观察性

内置 OpenTelemetry 支持。每条消息 emits a span (发出一个跨度)；tool calls 携带符合 2026 OTel GenAI semantic conventions (语义约定) 的 `gen_ai.*` 属性（Lesson 23）。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型设计是稳定的。Microsoft 已将活跃开发转向 Microsoft Agent Framework (2025 年 10 月 1 日公开预览；1.0 GA 目标为 2026 年第一季度末)。AutoGen 模式可以干净地移植 forward —— actor model 是持久的理念。

## 动手实现

`code/main.py` 实现了一个 stdlib actor runtime：

- `Message` —— 带有 `sender`、`recipient`、`topic`、`body` 的 typed payload (类型化载荷)。
- `Actor` —— 抽象类，带有 `receive(message, runtime)`。
- `Runtime` —— 带有 shared queue (共享队列)、delivery、failure isolation 的事件循环。
- 一个 two-actor demo：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行 checklist；它们交换消息直到达成共识。

运行方式：

```
python3 code/main.py
```

Trace 展示了消息投递、一个 actor 中模拟的失败不会 crash 另一个 actor，以及在一个 shared verdict (共享裁决) 上的 convergence (收敛)。

## 如何使用

- **AutoGen v0.4/v0.7** (维护中) —— 对研究、原型设计、多智能体模式稳定。
- **Microsoft Agent Framework** (公开预览) —— 前进路径；相同的 actor-model 理念， refreshed API。
- **LangGraph swarm topology** (Lesson 13) —— 通过 shared-tool handoffs 的类似模式。
- **Custom actor runtime** —— 当你需要特定 transport (NATS、RabbitMQ、gRPC) 时。

## 输出产物

`outputs/skill-actor-runtime.md` 生成一个最小的 actor runtime 加上一个 team template (团队模板)（RoundRobin 或 Selector），用于给定的 multi-agent task。

## 练习题

1. 添加 dead-letter queue (死信队列)：当 handler 抛出异常时，将失败的消息暂存供人工检查。在你的玩具中 DLQ 被命中的频率如何？
2. 实现 `SelectorGroupChat`：一个 selector actor 基于对话状态选择谁处理下一条消息。
3. 添加 distributed transport (分布式传输)：将进程内队列替换为 JSON-over-HTTP server，以便 actors 可以在单独的进程中运行。
4. 为每条消息接入 OTel span（或 no-op 占位符）。按照 Lesson 23 发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的玩具移植到真正的 `autogen_core` API。你跳过了哪些在生产中重要的部分？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| Actor | "Agent" | Private state + inbox + handler；不共享内存 |
| Message | "Event" | Typed payload；actors 交互的唯一方式 |
| Inbox | "Mailbox" | 每个 actor 的待处理消息队列 |
| Runtime | "Agent host" | 路由消息并隔离失败的事件循环 |
| Topic | "Channel" | Actors 之间的命名发布-订阅路由 |
| Fault isolation | "Let it crash" | 一个 actor 失败不会 crash 其他 actors |
| RoundRobinGroupChat | "固定轮换团队" | Agents 按顺序轮流发言 |
| SelectorGroupChat | "上下文路由团队" | Selector 选择下一个发言者 |
| Magentic-One | "参考团队" | 用于 web + code + files 的多智能体小队 |

## 延伸阅读

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— 重新设计文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 图形态替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— AutoGen 默认发出的 spans
