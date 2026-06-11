# Agent 框架权衡 —— LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都卖同一个 demo（research agent 生成报告）并隐藏同一个 bug（state schema 与 orchestration 层打架）。选择抽象与你问题形状匹配的框架；其余都是你写两遍的胶水代码。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 11 · 09（Function Calling），Phase 11 · 16（LangGraph）
**时间：** ~45 分钟

## 问题

你有一个需要多次 LLM 调用的任务。也许是一个 research workflow（plan、search、summarize、cite）。也许是一个 code-review pipeline（parse diff、critique、patch、validate）。也许是一个多轮助手，预订航班、写邮件和提交报销单。你选择一个框架。

三天后，你发现框架的抽象在泄漏。CrewAI 给你角色，但当 "researcher" 需要向 "writer" 传递结构化 plan 时，它跟你作对。AutoGen 给你 agent 之间的聊天，但没有一等状态，所以你的 checkpoint 是一个 conversation log 的 pickle。LangGraph 给你状态图，但迫使你在知道 agent 会做什么之前就命名每个转换。Agno 给你单 agent 抽象，当你尝试扇出到三个并发 worker 时它会尖叫。

解决方案不是"选择最佳框架"。而是将框架的核心抽象与你问题的形状匹配。本课绘制了那张地图。

## 概念

![Agent 框架矩阵：核心抽象 vs 问题形状](../assets/framework-matrix.svg)

四个框架主导 2026 年的格局。它们的核心抽象并不相同。

| 框架 | 核心抽象 | 最佳匹配 | 最差匹配 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` —— 类型化状态、节点、conditional edges、checkpointer。 | 具有显式状态和 human-in-the-loop 中断的 workflow；需要 time-travel 调试的生产 agent。 | 松散的、角色驱动的头脑风暴，拓扑未知。 |
| **CrewAI** | `Crew` —— 角色（goal、backstory）、任务、process（sequential 或 hierarchical）。 | 具有短线性/层次化 plan 的角色扮演或 persona-driven workflow。 | 任何超出 crew 轮历史的 stateful 东西；复杂分支。 |
| **AutoGen** | `ConversableAgent` 对 —— 两个或更多 agent 轮流发言直到退出条件。 | 多 agent *对话*（teacher-student、proposer-critic、actor-reviewer），思考从聊天中涌现。 | 具有已知 DAG 的确定性 workflow；任何需要跨重启持久状态的东西。 |
| **Agno** | `Agent` —— 单个 LLM + 工具 + memory，可组合成团队。 | 快速构建的单 agent 和轻量级团队；强多模态和内置存储驱动。 | 具有自定义 reducers 的深度、显式分支图。 |

### "抽象"实际意味着什么

框架的核心抽象是你在白板上推销架构时画的东西。

- **LangGraph** → 你画一个图。节点是步骤，边是转换，每一点的状态对象都是类型化的。心智模型是状态机。
- **CrewAI** → 你画一个组织架构图。每个角色有工作描述，manager 路由任务。心智模型是一个小型专家团队。
- **AutoGen** → 你画一个 Slack DM。两个 agent 互相发消息；如果你需要 moderator，第三个加入。心智模型是聊天。
- **Agno** → 你画一个带工具的单一盒子。把盒子并排放置就组成团队。心智模型是"带电池包含的 agent"。

### 状态问题

状态是大多数框架选择在生产环境中崩溃的地方。

- **LangGraph。** 类型化状态（`TypedDict` 或 Pydantic model），per-field reducers，一等 checkpointer（SQLite/Postgres/Redis）。Resume、interrupt 和 time-travel 是免费的。*(参见 Phase 11 · 16。)*
- **CrewAI。** 状态通过 `context` 字段作为字符串在任务之间流动，或通过 `output_pydantic` 结构化。没有开箱即用的 durable per-crew store；如果 crew 必须 survive restart，你需要自己附加。
- **AutoGen。** 状态是 chat history 和任何用户定义的 `context`。Conversation transcripts 持久化；任意 workflow 状态不会，除非你写 adapter。
- **Agno。** 内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）通过 `storage=` 附加到 `Agent` —— conversation sessions 和 user memories 自动持久化。不是完整的 graph checkpointer；是一个 session store。

### 分支问题

每个非平凡 agent 都会分支。谁决定分支很重要。

- **LangGraph** —— 你决定，通过 conditional edges。Routing 是一个带命名分支的 Python 函数。分支在编译后的图中是一等的；checkpointer 记录了哪个分支被选中。
- **CrewAI** —— manager 在 hierarchical 模式下决定；sequential 模式下你在构建时决定。Routing 隐式在任务列表中；manager 的 prompt 之外没有一等 "if"。
- **AutoGen** —— agent 通过聊天决定。Branching 从谁接下来发言中涌现。`GroupChatManager` 选择下一个发言者；你可以手写 `speaker_selection_method`，但默认是 LLM-driven。
- **Agno** —— agent 通过接下来调用哪个工具来决定。Teams 有 coordinator/router/collaborator 模式；超出该范围的分支是开发者的责任。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任何 OTel exporter 的 OpenTelemetry。每个节点转换都是一个 trace span；checkpoints 兼作可 replay 的 traces。LangSmith 是一方选项；Langfuse/Phoenix 也有 adapter。
- **CrewAI** —— 自 2025 年末起一等 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** —— 通过 `autogen-core` 的 OpenTelemetry 集成；AgentOps 和 Opik 有 connector。Tracing 粒度是 per-agent-message，不是 per-node。
- **Agno** —— 内置 `monitoring=True` 标志加 OpenTelemetry exporters；与 Langfuse 紧密集成用于 session traces。

### 成本和延迟

四个框架都增加了 per-call 开销（框架逻辑、验证、序列化）。大致按开销递增排序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差异主要取决于框架做了多少额外的 LLM routing。CrewAI 的 hierarchical manager 花费 tokens 决定谁下一个去；AutoGen 的 `GroupChatManager` 同样。LangGraph 只在你写 `llm.invoke` 的地方花费 tokens。Agno 的单 agent 路径很薄。

当每次运行的成本重要时，优先选择 explicit routing（LangGraph edges、AutoGen `speaker_selection_method`）而非 LLM-selected routing。

### 互操作性

- **LangGraph** ↔ **LangChain** tools、retrievers、LLMs。一等 MCP adapter（tools 作为 MCP servers 导入）。
- **CrewAI** ↔ tools 继承自 `BaseTool`；LangChain tools、LlamaIndex tools 和 MCP tools 都可以 adapter 进来。Crew-to-crew delegation 通过 `allow_delegation=True`。
- **AutoGen** → `FunctionTool` 包装任何 Python callable；MCP adapter 可用。与 AG2 生态系统紧密耦合用于 agent-to-agent 模式。
- **Agno** → `@tool` decorator 或 BaseTool subclass；MCP adapter；tools 可以在 agent 和 team 之间共享。

## 技能

> 你能用一句话解释为什么给定框架适合给定的 agent 问题。

预构建检查清单：

1. **画出形状。** 这是一个图（类型化状态、命名转换）？一个角色扮演（专家交接工作）？一个聊天（agent 聊到完）？一个带工具的单 agent？
2. **决定谁分支。** Developer-decided branching → LangGraph。Manager-agent-decided → CrewAI hierarchical。Chat-emergent → AutoGen。Tool-call-decided → Agno。
3. **检查状态预算。** 你需要 resume-from-checkpoint 吗？Time-travel？Human interrupts mid-run？如果是，LangGraph 是默认；Agno sessions 覆盖 conversation-scoped state。
4. **检查成本预算。** LLM-selected routing 每轮花费额外 tokens。如果 agent 每天运行数千次，优先选择 explicit routing。
5. **预算框架开销。** 每个框架都是另一个依赖。如果任务是两次 LLM 调用加一个工具，写 30 行纯 Python；没有框架比没有框架更便宜。

在你能画出图、组织架构图、聊天或 agent 盒子之前，拒绝伸手去拿框架。拒绝迫使你为其状态模型而战的那个，为了你实际需要的东西。

## 决策矩阵

| 问题形状 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 具有类型化状态、人工批准、长时间运行的 Workflow DAG | LangGraph | 一等状态、checkpointer、interrupts、time-travel。 |
| 具有不同角色的 Research / writing pipeline | CrewAI（sequential）或 LangGraph subgraphs | Role-per-task 在 CrewAI 中表达成本低；分支复杂时升级到 LangGraph。 |
| Proposer-critic 或 teacher-student 对话 | AutoGen | 双 agent 聊天是它的原生形状。 |
| 带工具、session、memory 的单 agent | Agno | 最薄的设置，内置存储和 memory。 |
| 具有 reducers 的数千个并行扇出 | LangGraph + `Send` | 唯一具有一等 parallel-dispatch API 的框架。 |
| 快速原型，无框架承诺 | Plain Python + provider SDK | 没有框架是最快的框架。 |

## 练习

1. **简单。** 用同一任务 —— "research Anthropic's headquarters, write a 200-word brief, cite sources" —— 在 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）中实现它。报告每次运行的 token 成本和代码行数。
2. **中等。** 在 AutoGen（researcher ↔ writer 聊天，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools` 和 `write_tools` 的单 agent，加一个 session store）中构建同一任务。对四个实现按（a）每次运行成本、（b）crash 后恢复能力、（c）在 write 步骤前注入人工批准的能力进行排序。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，接受一个简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回带一句话理由的推荐。在你自己设计的六个用例上验证它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Orchestration | "Agent 如何协调" | 决定哪个节点/角色/agent 下一个运行的层。 |
| Durable state | "重启后恢复" | 附加到 checkpoint 或 session store 的 survive process death 的状态。 |
| LLM-selected routing | "让模型决定" | Planner LLM 每轮选择下一步；灵活但每轮决策都支付 tokens。 |
| Explicit routing | "开发者决定" | Python 函数或 static edge 选择下一步；廉价且可审计。 |
| Crew | "A CrewAI team" | 绑定为单个 runnable 的 Roles + tasks + process（sequential 或 hierarchical）。 |
| GroupChat | "AutoGen's multi-agent chat" | 具有 speaker selector 的 N 个 agent 之间的管理对话。 |
| Team (Agno) | "Multi-agent Agno" | 在一组 agent 上的 Route / coordinate / collaborate 模式。 |
| StateGraph | "LangGraph's graph" | 类型化状态、节点、conditional-edge、checkpointer 抽象。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) —— StateGraph、checkpointers、interrupts、time-travel。
- [CrewAI documentation](https://docs.crewai.com/) —— Crews、Flows、Agents、Tasks、Processes。
- [AutoGen documentation](https://microsoft.github.io/autogen/) —— ConversableAgent、GroupChat、teams、tools。
- [Agno documentation](https://docs.agno.com/) —— Agent、Team、Workflow、storage、memory。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 与框架无关的模式库（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个框架包装的那个循环。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) —— AutoGen 的设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) —— CrewAI-style persona stacks 所基于的角色扮演基础。
- Phase 11 · 16（LangGraph）—— 本课用作基准的框架。
- Phase 11 · 19（Reflexion）—— 一个与 LangGraph 清晰映射但与 CrewAI 笨拙映射的模式。
- Phase 11 · 22（Production observability）—— 如何为你选择的任何框架进行埋点。
