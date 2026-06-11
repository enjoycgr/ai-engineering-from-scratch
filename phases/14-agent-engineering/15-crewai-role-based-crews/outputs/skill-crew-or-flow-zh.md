---
name: crew-or-flow
description: 为给定任务选择 CrewAI Crew 或 Flow，并搭建最小实现。
version: 1.0.0
phase: 14
lesson: 15
tags: [crewai, crews, flows, multi-agent, role-based]
---

给定一个任务描述，选择 Crew（自主）或 Flow（确定性），然后搭建。

决策：

1. 任务是否有 SLA、合规或确定性重放要求？ -> Flow。
2. 任务是否是探索性的（研究、初稿、头脑风暴）？ -> Crew。
3. 任务是否有 4+ 个 specialists 且顺序由 LLM 挑选？ -> Hierarchical Crew。
4. 任务是否有 <=3 个 specialists 且顺序固定？ -> Sequential Crew 或 Flow —— 优先 Flow。

对于 Crews，生成：

1. Agent 定义：role、goal、backstory（精简，<=200 词）、tools。
2. Task 定义：description、expected_output、agent。
3. 带有正确 Process（Sequential | Hierarchical）的 Crew。
4. 一个 test harness (测试框架)，在样本输入上运行 Crew 并检查 expected_outputs 是否被生成。

对于 Flows，生成：

1. `@start` 入口函数。
2. 形成 DAG 的 `@listen(topic)` 步骤。
3. 显式事件 topics；没有 magical broadcast (魔法广播)。
4. 一个 replay harness (重放框架)：给定 kickoff payload，确定性地重新运行。

硬性拒绝：

- 没有 backstories 的 Crews。Backstories 是承重性的。
- 没有 explicit topic names 的 Flows。"Implicit chaining" 破坏了审计目的。
- 只有 2 个 specialists 的 Hierarchical Crews。Manager overhead 不值得其成本。

拒绝规则：

- 如果用户在纯生产合规任务上要求 Crew，拒绝并迁移到 Flow。
- 如果用户在开放式研究任务上要求 Flow，拒绝并迁移到 Crew。
- 如果 backstory 超过 200 词，拒绝并要求删减。Context budget 是有限的。

输出：`agents.py`、`tasks.py`、`crew.py` 或 `flow.py`，以及带有决策理由的 `README.md`。以 "what to read next" 结尾，指向 Lesson 24（Langfuse/AgentOps，用于可观测性）或 Lesson 13（如果 Flow 需要 durable resume 语义）。
