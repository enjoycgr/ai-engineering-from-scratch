---
name: actor-runtime
description: 构建一个 AutoGen v0.4 风格的 actor runtime，带有 private state (私有状态)、每个 actor 的 inbox (收件箱)、仅通过消息 IPC、fault isolation (故障隔离) 和 dead-letter queue (死信队列)。
version: 1.0.0
phase: 14
lesson: 14
tags: [autogen, actor-model, messaging, fault-isolation, dead-letter]
---

给定一个 multi-agent task (多智能体任务)，生成一个 actor runtime 和所需的 agent actors。

生成：

1. 一个 `Message` 类型，带有 `sender`、`recipient`、`topic`、`body`、`mid`。
2. 一个 `Actor` 基类，带有 `receive(message, runtime)`。Actor state 是私有的。
3. 一个 `Runtime`，带有 shared queue (共享队列)、`send()`、`run_until_idle()` 和 dead-letter queue。Handler 中的异常进入 DLQ；不要 propagate (传播)。
4. 一个 topology helper (拓扑辅助器)：RoundRobin（固定轮换）、Selector（LLM 选择下一个）或 custom broadcast (自定义广播)。
5. 每条消息的可观测性钩子：按照 Lesson 23 发出带有 `gen_ai.agent.name` 和 `gen_ai.operation.name` 的 OTel spans。

硬性拒绝：

- 阻塞发送方直到收件方返回的同步消息传递。这是 v0.2 模型；它破坏了 fault isolation。
- Actors 之间的共享可变状态。Actors 通过消息读取 state，或者根本不读取。
- 传播 handler 异常的 runtime。失败属于 DLQ；让其他 actors 继续运行。

拒绝规则：

- 如果任务只有两个 actors 且是固定的 back-and-forth (来回对话)，拒绝 actor 框架并建议 prompt chain (Lesson 12)。Actors 在有 >=3 个 actors 或 async concurrency 时才值得其成本。
- 如果用户想要 "同步模式" 以 "更容易调试"，拒绝。建议 logging + tracing (Lesson 23) 代替。
- 如果领域是严格的 request/response 且只有一个 specialist，建议 routing (Lesson 12) 而不是 actor 团队。

输出：`message.py`、`actor.py`、`runtime.py`、`teams.py`、`README.md`，解释 DLQ 策略、topology 选择以及 OTel spans 的接入方式。以 "what to read next" 结尾，指向 Lesson 25（如果 actors 需要协商），Lesson 23（如果需要 tracing），或 Microsoft Agent Framework（如果你想要前瞻性的 runtime）。
