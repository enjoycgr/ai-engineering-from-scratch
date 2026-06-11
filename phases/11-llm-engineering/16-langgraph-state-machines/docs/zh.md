# LangGraph —— Agent 的状态机

> 手写的 ReAct 循环是一个 `while True`。用 LangGraph 写的 ReAct 循环是一个你可以 checkpoint、中断、分支和 time-travel 的图。Agent 本身没有变。围绕它的 harness 变了。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 11 · 09（Function Calling），Phase 11 · 14（Model Context Protocol）
**时间：** ~75 分钟

## 问题

你交付了一个 function-calling agent。它运行三轮，然后出了问题：模型尝试了一个返回 500 的工具，用户在任务中途改变了主意，或者 agent 决定在没有人工签字的情况下退款订单。`while True:` 循环没有钩子。你无法暂停它，无法回退它，也无法分支到 "如果模型选择了另一个工具会怎样"。一旦你把这个 demo 交付出去，agent 就变成了一个黑盒，要么成功了，要么失败了。

下一步一旦你看到它就很明显。Agent 已经是一个状态机 —— system prompt 加上 message history 加上 pending tool calls 加上下一个动作。让状态机显式化：节点代表 "模型思考"、"工具运行"、"人工批准"，边代表它们之间的条件转换。一旦图是显式的，harness 就免费获得四样东西：checkpointing（在步骤之间保存状态）、interrupts（暂停等待人工）、streaming（流式传输 token 和中间事件）和 time-travel（回退到先前状态并尝试不同分支）。

LangGraph 是实现这个抽象的库。它不是 LangChain 意义上的 agent 框架（"给你一个 AgentExecutor，祝你好运"）。它是一个图运行时，具有一等状态、一等持久化和一等中断。Agent 循环是你画出来的，不是手写的。

## 概念

![LangGraph StateGraph：节点、边和 checkpointer](../assets/langgraph-stategraph.svg)

一个 `StateGraph` 有三样东西。

1. **State（状态）。** 一个类型化的 dict（TypedDict 或 Pydantic model），流经整个图。每个节点接收完整状态并返回部分更新，LangGraph 使用每个字段的 *reducer* 进行合并 —— 对于应该累积的列表使用 `operator.add`，默认覆盖。
2. **Nodes（节点）。** Python 函数 `state -> partial_state`。每个都是一个离散步："调用模型"、"运行工具"、"摘要"。
3. **Edges（边）。** 节点之间的转换。Static edges 指向一个地方。Conditional edges 接受一个 router 函数 `state -> next_node_name`，使图可以基于模型输出分支。

你编译图。Compile 绑定拓扑，附加 checkpointer（可选但对生产环境至关重要），并返回一个 runnable。你用初始状态和 `thread_id` 调用它。执行的每一步都将一个 checkpoint 持久化到以 `(thread_id, checkpoint_id)` 为 key 的存储中。

### 四种超能力

**Checkpointing。** 每个节点转换将新状态写入存储（测试用内存，生产用 Postgres/Redis/SQLite）。通过使用相同的 `thread_id` 再次调用图来恢复。图从暂停处继续。

**Interrupts。** 用 `interrupt_before=["human_review"]` 标记一个节点，执行在该节点运行前停止。状态持久化。你的 API 向用户响应 "awaiting approval"。稍后使用 `Command(resume=...)` 向同一 `thread_id` 发送请求即可恢复执行。

**Streaming。** `graph.stream(state, mode="updates")` 在发生时产生状态增量。`mode="messages"` 在 model 节点内部流式传输 LLM tokens。`mode="values"` 产生完整快照。你选择要在 UI 中展示什么。

**Time-travel。** `graph.get_state_history(thread_id)` 返回完整的 checkpoint 日志。将任何先前的 `checkpoint_id` 传递给 `graph.invoke`，你就从该点分叉。非常适合调试（"如果模型选择了工具 B 会怎样？"）和用于 replay 生产 trace 的回归测试。

### Reducers 是关键

每个状态字段都有一个 reducer。大多数默认值没问题 —— 新值覆盖旧值。但 message lists 需要 `operator.add`，以便新消息追加而不是替换。Parallel edges 通过 reducer 合并它们的更新。如果两个节点都更新 `messages` 而你忘记了 `Annotated[list, add_messages]`，第二个会静默获胜，你会丢失半轮对话。Reducer 是库中唯一微妙的东西；做对了，其余部分就会组合起来。

### 四节点 ReAct 图

一个生产级 ReAct agent 是四个节点和两条边：

1. `agent` —— 用当前 message history 调用 LLM。返回 assistant message（可能包含 tool_calls）。
2. `tools` —— 执行最后一条 assistant message 中的任何 tool_calls，将工具结果作为 tool messages 追加。
3. 从 `agent` 出发的 conditional edge，如果最后一条消息有 tool_calls 则路由到 `tools`，否则到 `END`。
4. 从 `tools` 回到 `agent` 的 static edge。

就是这样。你获得了完整的 ReAct 循环（Thought → Action → Observation → Thought → …），带 checkpointing、interrupts 和 streaming，大约 40 行代码。

### StateGraph vs Send（扇出）

`Send(node_name, state)` 让一个节点可以 dispatch 并行子图。示例：agent 决定同时查询三个检索器。每个 `Send` 生成目标节点的一个并行执行；它们的输出通过 state reducer 合并。这就是 LangGraph 表达 orchestrator-workers 模式的方式，无需线程原语。

### Subgraphs

一个编译后的图可以成为另一个图的节点。外层图看到一个单一节点；内层图有自己的状态和自己的 checkpoints。这就是团队构建 supervisor-worker agent 的方式：supervisor 图将用户意图路由到 per-domain worker subgraph。

## 构建

### Step 1：状态和节点

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是让 message list 累积而不是覆盖的 reducer。忘记它是最常见的 LangGraph bug。

### Step 2：用 thread 运行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每个更新都是一个 dict `{node_name: state_delta}`。你的前端可以将这些流式传输到 UI，让用户看到 "agent 正在思考… 调用 search_web… 获得结果… 回答中。

### Step 3：添加 human-in-the-loop 中断

标记一个节点，使其在运行前暂停执行。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

状态、checkpoint 和 thread 都在中断期间持久化。执行期间只有内存中有数据。

### Step 4：用于调试的 time-travel

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

传递 `None` 作为输入会从给定 checkpoint replay；传递一个值会将其作为更新追加到该 checkpoint 的状态后再恢复。这就是你无需重新运行整个对话即可复现 bad agent run 的方式。

### Step 5：为生产环境更换 checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 和 Postgres 都已提供。`MemorySaver` 用于测试。任何需要跨重启持久化的东西都需要真正的存储。

## 技能

> 你将 agent 构建为图，而不是 `while True` 循环。

在你使用 LangGraph 之前，做一个 60 秒的设计：

1. **命名节点。** 每个离散决策或副作用操作都是一个节点。"Agent 思考"、"工具运行"、"reviewer 批准"、"响应流式传输"。如果你列不出来，任务还不是 agent 形状的。
2. **声明状态。** 最小 TypedDict，每个 list 字段都有 reducer。不要把所有东西都塞进 `messages`；将任务特定字段（一个工作中的 `plan`、一个 `budget` 计数器、一个 `retrieved_docs` 列表）提升到顶层。
3. **画出边。** 除非下一步依赖模型输出，否则都是 static。每个 conditional edge 都需要一个带命名分支的 router 函数。
4. ** upfront 选择 checkpointer。** 测试用 `MemorySaver`，其他情况用 Postgres/Redis/SQLite。不要没有 checkpointer 就交付 —— 没有 checkpointer 意味着没有 resume、没有 interrupt、没有 time-travel。
5. **在工具运行前决定中断，而不是之后。** 批准放在 side-effecting 节点的入边上，这样你可以在伤害发生前取消；验证放在 model 的出边上，这样你可以廉价地拒绝 bad calls。
6. **默认流式传输。** UI 用 `mode="updates"`，model 节点内部的 token-level streaming 用 `mode="messages"`，eval 期间的完整快照用 `mode="values"`。

拒绝交付没有 checkpointer 的 LangGraph agent。拒绝在副作用*之后*中断的那个。拒绝 `messages` 字段没有 `add_messages` 作为其 reducer 的那个。

## 练习

1. **简单。** 用 calculator tool 和 web-search tool 实现上面的四节点 ReAct 图。验证 `list(app.get_state_history(config))` 在两轮对话中至少返回四个 checkpoints。
2. **中等。** 在 `agent` 之前添加一个 `planner` 节点，将结构化 `plan: list[str]` 写入状态。让 `agent` 标记 plan 步骤为完成。如果 `plan` 在 checkpoint resume 中丢失（错误的 reducer），则测试失败。
3. **困难。** 构建一个 supervisor 图，使用 `Send` 在三个 subgraph（`researcher`、`writer`、`reviewer`）之间路由。每个 subgraph 有自己的状态和 checkpointer。在外层图上添加 `interrupt_before=["writer"]`，以便人工可以批准 research brief。确认从先前 checkpoint 的 time-travel 只重新运行分叉的分支。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| StateGraph | "The LangGraph graph" | 你在 compile 之前添加节点和边的 builder 对象。 |
| Reducer | "How the field merges" | 当节点返回该字段的更新时应用的函数 `(old, new) -> merged`；默认是覆盖，`add_messages` 是追加。 |
| Thread | "A conversation ID" | 一个 `thread_id` 字符串，为一次会话的所有 checkpoints 划定范围。 |
| Checkpoint | "A paused state" | 节点转换后完整图状态的持久化快照，以 `(thread_id, checkpoint_id)` 为 key。 |
| Interrupt | "Pause for a human" | `interrupt_before` / `interrupt_after` 在节点边界停止执行；用 `Command(resume=...)` 恢复。 |
| Time-travel | "Fork from a prior step" | `graph.invoke(None, config_with_old_checkpoint_id)` 从该 checkpoint 向前 replay。 |
| Send | "Parallel subgraph dispatch" | 一个节点可以返回的构造函数，用于生成目标节点的 N 个并行执行。 |
| Subgraph | "A compiled graph as a node" | 用作另一个图中节点的编译后 StateGraph；保留自己的状态作用域。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) —— StateGraph、reducers、checkpointers 和 interrupts 的规范参考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) —— 本课使用的心智模型，来自官方来源。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) —— Postgres/SQLite/Redis 存储、checkpoint 命名空间和 thread ID 的细节。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) —— `interrupt_before`、`interrupt_after`、`Command(resume=...)` 和 edit-state 模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个 LangGraph agent 实现的模式；阅读它以了解 reasoning trace 的原理。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 应优先选择哪些图形状（chain、router、orchestrator-workers、evaluator-optimizer）以及何时选择。
- Phase 11 · 09（Function Calling）—— 每个 LangGraph agent 节点复用的 tool-call 原语。
- Phase 11 · 14（Model Context Protocol）—— 通过 MCP adapter 插入 LangGraph `ToolNode` 的外部工具发现。
- Phase 11 · 17（Agent framework tradeoffs）—— 何时选择 LangGraph 而非 CrewAI、AutoGen 或 Agno。
