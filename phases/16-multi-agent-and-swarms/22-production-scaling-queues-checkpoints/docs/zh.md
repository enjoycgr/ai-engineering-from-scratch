# 生产级扩展 —— 队列、检查点、持久性

> 将多智能体（multi-agent）系统扩展到数千个并发运行需要**持久执行（durable execution）**。LangGraph 的运行时在每次 super-step（超级步骤）后写入一个以 `thread_id` 为键的 checkpoint（检查点）（默认使用 Postgres）；工作进程崩溃会释放租约，另一个工作进程恢复执行。智能体（agent）可以无限期休眠等待人工输入。**MegaAgent**（arXiv:2408.09955）为每个智能体运行一个生产者-消费者队列，具有三种状态（Idle / Processing / Response）和两层协调（组内聊天 + 组间管理员聊天）。**Fiber/async** 在 LLM 流式传输方面优于 thread-per-job（每任务一个线程）：线程 99% 的时间都在等待 token 时空闲，而 fibers 在 I/O 上协作式让出。反面观点：Ashpreet Bedi 的《Scaling Agentic Software》认为，在负载证明需要之前，**FastAPI + Postgres + 仅此而已** 就足够了——简单架构比预期走得更远。本课构建一个持久的 checkpoint log（检查点日志）、一个具有状态转换的 per-agent work queue（每智能体工作队列）、一个 async-vs-thread 演示，并落地务实的"先简单开始"规则。

**类型：** 学习 + 构建
**语言：** Python（标准库、`asyncio`、`sqlite3`）
**前置要求：** Phase 16 · 09（并行群体网络）、Phase 16 · 13（共享内存）
**时间：** 约 75 分钟

## 问题

一个原型多智能体系统在一台笔记本电脑上运行三个智能体的内存事件循环。你将其投入生产：

- 智能体有时运行数小时（长期研究、human-in-the-loop（人在回路）等待）。
- 工作进程崩溃。重启会丢失状态。
- 峰值负载是平均值的 10 倍；你需要 horizontal scaling（水平扩展）。
- 用户按 agent-run 付费；你需要 exactly-once semantics（恰好一次语义）来计费。

内存事件循环做不到这些。你需要一个底层的 durable execution（持久执行）层。2026 年的标准选项是：

1. 具有 checkpoint（检查点）的工作流引擎（Temporal、LangGraph runtime）。
2. 具有状态存储的消息队列（Postgres + SQS/RabbitMQ）。
3. Actor-model 框架（MegaAgent 的每个智能体生产者-消费者）。
4. 手写的 FastAPI + Postgres（Bedi 的论点）。

本课构建每个选项的微型版本。

## 概念

### 持久执行（durable execution），该模式

一个 durable-execution engine（持久执行引擎）在每次"步骤"（super-step，用 LangGraph 的术语）后持久化完整的程序状态。崩溃时：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

这能工作的要求：

- **可序列化状态（Serializable state）。** 所有智能体状态都必须可持久化。带有活动数据库连接的函数闭包无法存活。
- **确定性恢复（Deterministic resume）。** 给定相同的状态和相同的输入，智能体产生相同的动作（或将 LLM 调用推迟到外部确定性 oracle）。
- **幂等副作用（Idempotent side effects）。** 外部调用（工具调用、支付）必须是幂等的，或使用 deduplication key（去重键）。

LangGraph 在每次 super-step 后写入 checkpoint；Temporal 在每次 activity 后写入；Restate 使用 event-sourced journals（事件溯源日志）。三者都实现了相同的模式。

### LangGraph 的 runtime

每个智能体有一个 `thread_id`；状态是一个类型化的 dict；每个 super-step 向 checkpoints 表写入一行。恢复时，runtime 从最后一个 checkpoint 重放，而非从头开始。智能体可以 `interrupt()` 等待人工输入；runtime 持久化并释放工作进程。当输入到达时，任何工作进程都可以恢复。

这是 2026 年 4 月的参考生产设计。

### MegaAgent 的 per-agent queue

arXiv:2408.09955 描述了一个扩展实验：一个集群中数千个并发智能体。架构：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

两层协调让组内对话密集发生，而组间保持稀疏——这是用于在数千个智能体中保持成本线性的模式。

### Async vs thread-per-job

LLM 调用是 I/O 绑定的。一个等待下一个 token 的线程 99% 的时间都是空闲的。每个线程成本约 1MB RAM；在 10,000 个并发调用时，仅栈就需 10GB。

Fibers（Python `asyncio`、Go goroutines、Rust `tokio`）在 I/O 上协作式让出。同样的 10,000 个调用可以舒适地容纳在一个进程中。在 LLM-agent 规模下，async 不是优化——它是架构。

例外：CPU 绑定的后处理（embedding、tokenizer 技巧）仍然需要线程或进程。将 I/O 层与 CPU 层分开。

### Bedi 的反面观点

《Scaling Agentic Software》（Ashpreet Bedi，2026）认为大多数团队在测量负载之前就过度工程化了。务实的默认方案：

- FastAPI + Postgres。
- 每次 agent run 是一行；状态通过乐观并发就地更新。
- 通过 `pg_notify` 或简单的 Celery worker 进行后台作业。
- 应用代码中的 retry policy（重试策略）。

对于低于约 100 个并发 agent-run 的可管理任务负载，这通常就是你所需要的一切。当你测量到它失败时再升级。

规则：当你遇到简单架构无法解决的具体问题时，再采用 durable-execution（持久执行）框架。过早采用会浪费时间在无法带来回报的仪式上。

### Exactly-once semantics（恰好一次语义）

对于付费的 agent run，你需要"exactly-once effective"（至少一次交付 + 幂等消费者）。工程手段：

- **每次运行的去重键（Dedup key）。** 将其包含在每个副作用调用中。
- **发件箱模式（Outbox pattern）。** 副作用先写入表，然后单独进程执行它们。两个步骤都是幂等的。
- **补偿事务（Compensating transactions）。** 当副作用成功但其跟踪写入失败时，安排补偿。

这些是数据库工程模式，不是 LLM 特有的。LLM 的代价只是 LLM 调用很慢；其他一切都是标准分布式系统。

### 彩虹部署（Rainbow deployment）

Anthropic 的多智能体研究系统使用"rainbow deployments"：多个版本的 agent runtime 并发运行，因此长期运行的智能体不必在每次代码部署时被杀死。在新版本上对新流量进行 canary（金丝雀）；当旧版本的智能体完成时退役旧版本。

这对长期运行的有状态系统是标准的；2026 年的适配是智能体可以存活数小时，因此部署周期必须适应这一点。

### 标准生产检查清单

- 持久状态（Durable state）（checkpoints、snapshots 或 outbox + replayable log（可重放日志））。
- 幂等副作用（Idempotent side effects）。
- 用于 LLM 调用的 Async I/O layer（异步 I/O 层）。
- 带 dedup（去重）的 At-least-once delivery（至少一次交付）。
- 用于有状态工作负载的 Rainbow/canary deployment（彩虹/金丝雀部署）。
- 可观测性（Observability）：per-agent traces（每智能体链路追踪）、super-step audit（超级步骤审计）、retry counter（重试计数器）。

## 构建

`code/main.py` 实现了：

- `CheckpointStore` —— 基于 SQLite 的 checkpoint log（检查点日志），使用 thread-id 键。每个 super-step 追加一行。
- `run_with_checkpoint(agent, thread_id)` —— 模拟运行中崩溃；第二个工作进程从最后一个 checkpoint 恢复。
- `AgentQueue` —— 具有 Idle / Processing / Response 状态机的 per-agent（每智能体）小型工作队列。
- `demo_async_vs_threads()` —— 通过 asyncio 和线程运行 500 个并发模拟"LLM 调用"；报告 wall-clock（挂钟时间）和峰值内存（近似值）。

运行：

```
python3 code/main.py
```

预期输出：在模拟崩溃后 checkpoint resume 成功；async 版本在 < 1s 内处理 500 个并发调用；thread 版本需要数秒，且每个并发单元使用的内存数量级更高。

## 应用

`outputs/skill-scaling-advisor.md` 就 durable-execution（持久执行）选择提供建议：FastAPI + Postgres、LangGraph runtime、Temporal 或自定义。根据负载、状态保留需求和部署频率进行校准。

## 交付

标准生产加固：

- **先简单开始（Bedi 规则）。** FastAPI + Postgres，直到你测量到它失败。
- **在优化前对所有内容进行检测。** 每次运行的 latency histogram（延迟直方图）、每步时间、重试计数、失败分类。
- **副作用使用发件箱模式（Outbox pattern）。** 特别是支付和外部 API 调用。
- **彩虹部署（Rainbow deploys）。** 绝不在部署期间杀死运行中的 agent run。
- **在以下情况采用持久执行引擎（durable-execution engines）（Temporal / LangGraph / Restate）：** 你遇到具体问题：数小时的 human-in-the-loop（人在回路）等待、跨区域协调、复杂的重试/补偿策略。
- **I/O 层使用 Async。** 仅对 CPU 绑定的后处理使用线程。

## 练习

1. 运行 `code/main.py`。确认 checkpoint resume 有效；测量 async 与 thread 并发差异。
2. 实现一个 **outbox** 表：每个工具调用先写入 outbox，然后单独的 goroutine/task 执行。通过运行两次工具调用来验证 idempotency（幂等性）。
3. 模拟一个 **rainbow deploy**：两个并发 runtime 版本；将一半新 thread_id 路由到每个版本；确认旧版本上的运行中线程不会被中断。
4. 阅读 LangGraph 的 runtime 文档（见下方链接）。识别 runtime 的哪些功能在手写的 FastAPI + Postgres 版本中复制耗时最长。这是采用它的理由，还是可以推迟？
5. 阅读 MegaAgent（arXiv:2408.09955）第 3 节。两层协调（intra-group + inter-group admin chat）是显式的。草图如何将其映射到具有两个 queue families（队列族）的消息队列。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Durable execution | "Persist the program state" | 引擎在每次 super-step 后写入状态；崩溃恢复是确定性的。 |
| Super-step | "Transactional boundary" | 检查点之间的工作单元。LangGraph 术语。 |
| thread_id | "Agent run identifier" | 绑定检查点和恢复逻辑的键。 |
| Idempotency | "Safe to retry" | 重复副作用产生的结果与单次尝试相同。 |
| Outbox pattern | "Decouple side effects" | 将意图写入一张表；独立执行器执行并标记完成。 |
| At-least-once delivery | "Possible duplicates" | 消息队列语义；去重键使消费者达到 effective-once。 |
| Rainbow deploy | "Overlapping versions" | 长期运行工作负载期间多个运行时版本并发。 |
| Async fiber | "Cooperative yielding" | 用户态并发；相比线程，对 I/O 绑定负载更轻量。 |
| Checkpoint | "State snapshot" | super-step 边界处的序列化状态；恢复的关键。 |

## 延伸阅读

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) —— LangGraph 运行时设计
- [MegaAgent](https://arxiv.org/abs/2408.09955) —— 每个智能体的生产者-消费者队列；数千并发智能体的两层协调
- [Matrix](https://arxiv.org/abs/2511.21686) —— 以消息队列为协调基础的分布式框架
- [Temporal docs](https://docs.temporal.io/) —— 持久执行的参考工作流引擎
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) —— 生产经验，包括彩虹部署
