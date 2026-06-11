# Handoffs and Routines — 无状态编排

> OpenAI 的 Swarm（2024 年 10 月）将 multi-agent (多智能体) 编排提炼为两个原语：**routine (例程)**（指令 + 工具，作为 system prompt (系统提示词)）和 **handoff (交接)**（一个返回另一个 Agent 的工具）。没有状态机，没有分支 DSL——LLM 通过调用正确的 handoff 工具来路由。OpenAI Agents SDK（2025 年 3 月）是其生产级继任者。Swarm 本身仍然是最简洁的概念参考——它的全部源码只有几百行。这个模式之所以迅速传播，是因为 API 表面大致就是 "agent = prompt + tools; handoff = function returning agent"。局限：无状态，因此记忆由调用方负责。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 分钟

## Problem (问题)

每个 multi-agent 框架都想让你学习它的 DSL：LangGraph 的 node (节点) 和 edge (边)，CrewAI 的 crew 和 task (任务)，AutoGen 的 GroupChat 和 manager (管理者)。这些 DSL 确实是真实的抽象，但它们让事情感觉比实际需要的更重。

Swarm 走向了相反的方向：使用模型已经具备的 tool-calling (工具调用) 能力。Handoffs 变成了 tool calls (工具调用)。编排器是任何当前持有对话的 agent (智能体)。状态机隐含在 agent 的 system prompt 中。

## Concept (概念)

### Two primitives (两个原语)

**Routine (例程)。** 定义 agent 角色和可用工具的 system prompt。把它想象成一组有作用域的指令："你是一个 triage agent (分流智能体)；如果用户询问退款，hand off (交接) 给 refund agent (退款智能体)。"

**Handoff (交接)。** Agent 可以调用的一个工具，返回一个新的 Agent 对象。Swarm 运行时检测到 Agent 返回值，并在下一轮切换 active agent (活跃智能体)。

这就是整个抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm 看到 Agent 返回值 → 切换活跃智能体

triage_agent = Agent(
    name="triage",
    instructions="将用户路由到正确的专家。",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

Triage agent 的 system prompt 让它根据用户消息选择合适的 handoff。LLM 的 tool-calling 完成了路由。

### Why it is viral (为什么它迅速传播)

- **Small API (小型 API)。** 两个概念即可上手。
- **Uses what the model already does (使用模型已有的能力)。** Tool calling 在各提供商那里已经是生产级的能力。
- **No state-machine burden (无状态机负担)。** 你不需要描述图；agent 的 prompt 描述了它们要 hand off 给谁。

### The stateless trade (无状态的取舍)

Swarm 明确在运行之间是无状态的。框架在一次运行期间保留 message history (消息历史)，但不持久化任何东西。记忆、连续性、长时间运行的任务——都是调用方的问题。

在生产环境中（OpenAI Agents SDK，2025 年 3 月），这是主要改变之一：SDK 添加了内置的 session management (会话管理)、guardrails (护栏) 和 tracing (追踪)，同时保留了 handoff 原语。

### When Swarm/handoffs fit (何时适合 Swarm/handoffs)

- **Triage patterns (分流模式)。** 一线 agent 将用户路由到专家。
- **Skill-based handoffs (基于技能的交接)。** "如果任务需要代码，调用 coder；如果需要研究，调用 researcher。"
- **Short, bounded conversations (短而有界的对话)。** 客户支持、FAQ 转工单、简单工作流。

### When Swarm struggles (何时 Swarm 力不从心)

- **Long sessions with shared memory (带共享记忆的长时间会话)。** Handoffs 将对话状态重置为新 agent 的 prompt 加历史。没有调用方管理的记忆，就无法跨 agent 持久化状态。
- **Parallel execution (并行执行)。** Handoff 是单次的——active agent 切换。并行需要调用方编排多个 Swarm 运行。
- **Audit and replay (审计与重放)。** 无状态运行很难精确重放；LLM 的 handoff 选择不是确定性的。

### OpenAI Agents SDK (March 2025)

生产级继任者添加了：

- **Session state (会话状态)。** 跨运行的持久化 thread (线程)。
- **Guardrails (护栏)。** 输入/输出验证钩子。
- **Tracing (追踪)。** 每个 tool call 和 handoff 都被记录。
- **Handoff filters (交接过滤器)。** 控制 handoff 时传输什么 context (上下文)。

Handoff 原语保留了下来；生产级的人机工程学围绕它添加。

### Swarm vs GroupChat

两者都使用 LLM 驱动的路由，但在**谁选择下一个**上不同：

- GroupChat：selector (选择器)（函数或 LLM）从外部选择下一个发言者。
- Swarm：当前 agent 通过调用 handoff 工具选择其继任者。

Swarm 是 "agent 决定接下来是什么"；GroupChat 是 "manager 决定接下来是什么"。Swarm 的决策存在于 active agent 的 tool call 中；GroupChat 的决策存在于 `GroupChatManager` 中。

## Build It (动手实现)

`code/main.py` 从头实现了 Swarm：一个 Agent dataclass (数据类)、一个 handoff 机制（工具返回 Agent），以及一个检测 agent 切换的运行循环。

演示：一个 triage agent 路由到 refund、sales 或 support 专家。每个专家有自己的工具。运行循环打印每次 handoff。

运行：

```
python3 code/main.py
```

## Use It (使用它)

`outputs/skill-handoff-designer.md` 为给定任务设计一个 handoff topology (交接拓扑)：哪些 agent 存在、它们可以调用哪些 handoffs、什么 context 被传输。

## Ship It (交付上线)

Checklist (检查清单)：

- **Handoff logging (交接日志)。** 每次 handoff 写入一个 trace event (追踪事件)，包含 from-agent、to-agent、context snapshot (上下文快照)。
- **Context transfer rules (上下文传输规则)。** 决定 handoff 时移动什么：完整历史（昂贵）、最后 N 条消息，或摘要。
- **Guardrail on handoff (交接护栏)。** 向具有不同工具权限的专家 handoff 必须经过认证——否则 prompt injection (提示注入) 可以强制不想要的 handoffs。
- **Loop detection (循环检测)。** 两个 agent 来回交接是常见故障；用简单的 last-K ring check (最近 K 次环形检查) 检测。
- **Fallback agent (回退智能体)。** 如果 handoff 目标不存在，回退到安全的默认值。

## Exercises (练习)

1. 运行 `code/main.py`，triage 到 refund agent。确认第二轮的 active agent 是 refund。
2. 添加一个 loop-detection (循环检测) 规则：如果相同的两个 agent 连续交接 3 次，强制退出。设计 fallback (回退)。
3. 阅读 OpenAI Agents SDK 文档中的 handoff filters。实现一个 "summarize-on-handoff" 版本：outgoing agent ( outgoing 智能体) 在 incoming agent ( incoming 智能体) 接管前将 context 压缩为 bullet summary (要点摘要)。
4. 比较 Swarm handoff 和 GroupChatManager selector。哪种模式让 prompt injection 更糟，为什么？
5. 阅读 Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。找出一个 Swarm 明确做出的设计决策，OpenAI Agents SDK 改变或保留了它。

## Key Terms (关键术语)

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Routine (例程) | "The agent prompt" | System prompt + 工具列表。定义角色和可用的 handoffs。 |
| Handoff (交接) | "Transfer to another agent" | Active agent 可以调用的一个工具，返回一个新的 Agent。运行时切换 active agent。 |
| Stateless (无状态) | "No memory between runs" | Swarm 不持久化任何东西；记忆是调用方的责任。 |
| Active agent (活跃智能体) | "Who's speaking now" | 当前持有对话的 agent。Handoff 改变这个。 |
| Context transfer (上下文传输) | "What moves on handoff" | Incoming agent 看到什么历史的策略：完整、最后 N 条，或摘要。 |
| Handoff loop (交接循环) | "Agents ping-pong" | 两个 agent 不断互相 hand back (交回) 的故障模式。 |
| OpenAI Agents SDK | "Production Swarm" | 2025 年 3 月的继任者；在 handoff 原语之上添加了 sessions、guardrails、tracing。 |
| Handoff filter (交接过滤器) | "Gate on transfer" | SDK 功能，在 handoff 边界检查和修改 context。 |

## Further Reading (延伸阅读)

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 参考性阐述
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始实现，作为概念参考保留
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 带 sessions 和 tracing 的生产级继任者
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code subagents 如何通过 `Task` 使用类似 handoff 的模式
