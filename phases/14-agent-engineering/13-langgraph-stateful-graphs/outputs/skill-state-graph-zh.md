---
name: state-graph
description: 构建一个 LangGraph 风格的状态机，带有 typed state (类型化状态)、conditional edges (条件边)、per-node checkpointing (每节点检查点) 和 durable resume (持久化恢复)。
version: 1.0.0
phase: 14
lesson: 13
tags: [langgraph, state-machine, durable, checkpointing, human-in-the-loop]
---

给定一个目标 runtime、一个 state shape、一组节点函数和一个 checkpointer backend，生成一个有状态的 agent graph。

生成：

1. 一个 typed `State`（dict 或 Pydantic）。记录每个字段。Nodes 读取 state；它们返回 updates。
2. 一个 `StateGraph`，带有 `add_node`、`add_edge`、`add_conditional_edges`、`set_entry`，以及 `START`/`END` sentinels。
3. 一个 `Checkpointer` 接口，带有 `save(session_id, node, state)` 和 `load_latest(session_id)`。默认 SQLite；允许 Postgres/Redis/自定义。
4. 一个 `Runner`，逐步遍历图，每个节点后序列化 state，捕获 `PausedAtNode` 用于 human-in-the-loop，并支持带可选 `state_override` 的 `resume_from`。
5. 三个 topology helpers (拓扑辅助器)：supervisor (中央路由器)、swarm (共享工具交接)、hierarchical (子图)。

硬性拒绝：

- 没有显式捕获 random-seed 或 wall-clock 的非确定性节点。Resume 假设给定输入 state 的节点输出是可复现的。
- 只保存 "summary" state 的 checkpointer。序列化完整的 state，否则 resume 会失败。
- 每条边都是条件的图。优先选择带有偶尔分支的线性链。

拒绝规则：

- 如果用户要求没有持久化的状态图，拒绝。重点是 durable resume；如果你不需要 resume，使用 Lesson 12 的工作流模式。
- 如果用户要求 "只在成功时 checkpoint"，拒绝。失败也需要 state —— 那是调试开始的地方。
- 如果图有超过约 30 个节点，拒绝扁平布局并要求 nested subgraphs。扁平的 30 节点图无法审查。

输出：`state.py`、`graph.py`、`checkpointer.py`、`runner.py`、`README.md`，解释 state schema、checkpointer 选择和 resume 语义。以 "what to read next" 结尾，指向 Lesson 14（Actor 模型替代方案）、Lesson 16（handoffs/guardrails 层）或 Lesson 23（图步骤上的 OTel spans）。
