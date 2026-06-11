# LangGraph: 有状态图与持久化执行

> LangGraph 是 2026 年 low-level stateful orchestration (低层有状态编排) 的参考实现。Agent (智能体) 是一个 state machine (状态机)；nodes (节点) 是函数；edges (边) 是 transitions (转移)；state (状态) 是不可变的，每步之后都会被 checkpointed (检查点化)。从任何失败点精确恢复。

**Type:** Learn + Build
**Languages:** Python (stdlib), TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 分钟

## 学习目标

- 描述 LangGraph 的核心模型：带有 immutable state (不可变状态)、function nodes (函数节点)、conditional edges (条件边) 和 post-step checkpoints (步骤后检查点) 的状态机。
- 说出文档强调的四种能力：durable execution (持久化执行)、streaming (流式传输)、human-in-the-loop (人在回路)、comprehensive memory (综合记忆)。
- 解释 LangGraph 支持的三种 orchestration topologies (编排拓扑)：supervisor (监督者)、peer-to-peer (对等，swarm)、hierarchical (层级，嵌套子图)。
- 用标准库实现一个 state graph (状态图)，带有 immutable state、conditional edges 和 checkpoint/resume cycle (检查点/恢复周期)。

## 问题背景

Agents 和 workflows 共享一个问题：当 40-step 运行在第 38 步失败时，你希望从第 38 步恢复，而不是从头开始。二流的状态模型让运维人员不得不在假设 fresh runs (全新运行) 的库周围 hack retries (hack 重试)。

LangGraph 的设计答案是：state 是一等 typed object (类型化对象)，mutations (变更) 是显式的，checkpoints 在每个节点后持久化。Resume 是一个 `load_state(session_id)` 调用。

## 核心概念

### 图

图由以下要素定义：

- **State type (状态类型)。** 一个 typed dict（或 Pydantic 模型），每个节点读取并变更它。
- **Nodes (节点)。** Pure functions `(state) -> state_update`。Updates 在返回后合并到 state 中。
- **Edges (边)。** 节点之间的 conditional (条件) 或 direct (直接) transitions。
- **Entry and exit (入口和出口)。** `START` 和 `END` sentinel nodes (哨兵节点) 标记边界。

示例：一个带有 `classify`、`refund`、`bug`、`sales`、`done` 节点的 agent —— 一个作为图的 routing workflow (路由工作流)。

### Durable execution (持久化执行)

每个节点返回后，runtime 将 state 序列化并写入 checkpointer (检查点器)（SQLite、Postgres、Redis、自定义）。如果在第 N 步失败，runtime 可以 `resume(session_id)` 并从第 N+1 步开始，携带精确的 state。

LangGraph 文档明确强调了这方面重要的生产用户：Klarna、Uber、J.P. Morgan。声明的重点不是图的形状；而是图的形状加上 checkpointing 使恢复变得廉价。

### Streaming (流式传输)

每个节点都可以 yield (产出) 部分输出。图将 per-node-delta events (每个节点的增量事件) 流式传输给调用者，因此 UI 随图运行而更新。

### Human-in-the-loop (人在回路)

在节点之间检查和修改 state。实现方式：在关键节点前暂停，将 state 呈现给人类，接受修改，恢复。Checkpointer 使这变得容易，因为 state 已经被序列化。

### Memory (记忆)

Short-term (短期，单次运行内 —— state 中的对话历史) 和 long-term (长期，跨运行 —— 通过 checkpointer 加单独的长期存储持久化)。LangGraph 通过 tools 与外部 memory 系统（Mem0、自定义）集成。

### 三种拓扑

1. **Supervisor (监督者)。** 中央 router LLM (路由大语言模型) 分发给 specialist subagents (专家子智能体)。`langgraph-supervisor` 中的 `create_supervisor()`（尽管 LangChain 团队在 2026 年建议通过 tool calls 直接实现以获得更好的 context control (上下文控制)）。
2. **Swarm / peer-to-peer (蜂群/对等)。** Agents 通过共享工具表面直接 hand off (交接)。没有中央路由器。
3. **Hierarchical (层级)。** 管理子监督者的监督者，实现为 nested subgraphs (嵌套子图)。

### 这种模式的失效场景

- **Checkpoints 太小。** 只 checkpointing conversation turns (对话轮次) 会留下 tool state 和 memory writes 无法恢复。必须序列化完整的 state。
- **Non-deterministic nodes (非确定性节点)。** Resume 假设节点输入产生相同的 state update。Random seeds、wall-clock、external APIs 必须被捕获。
- **过度使用 conditional edges。** 每个边都是条件的图是一个无法被推理的 state machine。优先选择带有偶尔分支的 linear chains (线性链)。

## 动手实现

`code/main.py` 实现了一个 stdlib stateful graph：

- `State` —— 一个带有 `messages`、`step`、`route`、`output`、`human_approval` 的 typed dict。
- `Node` —— 接收 state 并返回 update dict 的 callable。
- `StateGraph` —— nodes + edges + conditional edges + run + resume。
- `SQLiteCheckpointer` (内存中的 fake) —— 每个节点后序列化 state；`load(session_id)` 恢复。
- 一个 demo graph：classify -> branch(refund / bug / sales) -> human gate -> send。

运行方式：

```
python3 code/main.py
```

Trace 展示了首次运行在 human gate 失败、持久化、然后 resume 产生最终输出的过程。

## 如何使用

- **LangGraph** —— 参考实现，生产就绪。使用 `create_react_agent`、`create_supervisor`，或构建自己的图。
- **AutoGen v0.4** (Lesson 14) —— 高并发场景的 actor model (Actor 模型) 替代方案。
- **Claude Agent SDK** (Lesson 17) —— 带有内置 session store 的托管 harness。
- **Custom** —— 当你需要精确控制 state shape 或 checkpointer backend 时。

## 输出产物

`outputs/skill-state-graph.md` 在任何目标 runtime 中生成 LangGraph 风格的状态图，内置 checkpointing 和 resume。

## 练习题

1. 从 `classify` 到 `end` 添加一条 conditional edge：当 classification confidence 低于阈值时触发。在人类手动设置 `route` 后恢复运行。
2. 将类似 SQLite 的 fake 替换为真正的 SQLite checkpointer。测量每步序列化开销。
3. 实现 parallel edges (并行边)：两个节点并发运行，通过 custom reducer (自定义归约器) 合并。Immutable state 在这里带来了什么？
4. 阅读 `langgraph-supervisor` 参考。将玩具示例移植到 `create_supervisor`。比较 trace 形状。
5. 添加 streaming：每个节点在运行时 yield 部分 state。打印到达的 deltas。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| State graph | "智能体作为状态机" | Typed state + nodes + edges + reducers |
| Checkpointer | "持久化后端" | 每个节点后序列化 state；支持 resume |
| Reducer | "状态合并器" | 将当前 state 与节点的 update 合并的函数 |
| Conditional edge | "分支" | 由 state 的函数选择的边 |
| Subgraph | "嵌套图" | 在另一个图中作为节点使用的图 |
| Durable execution | "从失败恢复" | 从最后一个成功节点重启，携带精确 state |
| Supervisor | "路由器 LLM" | 专家子智能体的中央调度器 |
| Swarm | "P2P 智能体" | 通过共享工具交接的 Agents；没有中央路由器 |

## 延伸阅读

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 参考文档
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) —— supervisor 模式 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) —— Actor 模型替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— session store 和 subagents
