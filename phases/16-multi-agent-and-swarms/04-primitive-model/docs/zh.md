# 多智能体原语模型

> 2026 年发布的每个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。四个原语，仅此而已：智能体（agent）、交接（handoff）、共享状态（shared state）、编排器（orchestrator）。本课从零构建它们，在一个玩具系统上运行全部四个，然后将每个主流框架映射到相同的坐标轴上，让你能用一个段落读懂任何新发布。

**类型：** 学习
**语言：** Python（标准库）
**前置条件：** Phase 14（智能体工程），Phase 16 · 01（为什么需要多智能体）
**时间：** ~60 分钟

## 问题

每六个月就有一个新的多智能体框架发布。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每篇新闻稿都声称自己是“正确的抽象”。

如果你试图逐个学习，你会 burnout。API 看起来不同。文档对“智能体”是什么意见不一。一个框架把共享内存叫“黑板（blackboard）”，另一个叫“消息池（message pool）”，第三个叫“StateGraph”。你开始怀疑这个领域只是在空转。

并非如此。营销之下，四个原语是稳定的。学会一次，用一个段落读懂每个新框架。

## 概念

### 四个原语

1. **Agent（智能体）** —— 一个系统提示词（system prompt）加一个工具列表。无状态（stateless）；每次运行都从它的系统提示词和当前消息历史开始。
2. **Handoff（交接）** —— 从一个智能体到另一个智能体的结构化控制权转移。机制上，是一个返回新智能体的工具调用，或一条跟随条件的图边（graph edge）。
3. **Shared state（共享状态）** —— 任何能被多个智能体读取（有时写入）的数据结构。消息池、黑板、键值存储、向量内存。
4. **Orchestrator（编排器）** —— 决定下一个谁说话的那个角色。选项：显式图（确定性）、LLM 说话者选择器（软性）、上一个说话者的交接调用（OpenAI Swarm），或队列上的调度器（集群架构）。

这就是整个设计空间。每个框架为每个坐标轴选择默认值；其余都是表面语法。

### 每个 2026 框架如何映射到它

| 框架 | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的问题 | LLM 的下一个交接调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的说话者选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出链式传递 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` reducer | 图，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | 智能体 + A2A 卡片 | A2A 任务 | A2A 工件 | 宿主决定 |

表面差异看起来巨大。底层：同样的四个旋钮。

### 为什么这很重要

一旦你看到原语，框架比较就变成了一份简短的检查清单：

- 编排器是信任 LLM 来路由（Swarm），还是把路由钉死在代码里（LangGraph）？
- 共享状态是完整历史（GroupChat）还是投影（StateGraph reducer）？
- 智能体能否修改彼此的提示词（CrewAI 管理者），还是只能交接（Swarm）？

这三个问题回答了 80% 的“哪个框架适合给定问题”。你不再寻找“最好的多智能体框架”，而是开始为你真正关心的坐标轴做设计。

### 无状态的洞察

除了共享状态，每个原语都是无状态的。Agent 是 (prompt, tools) 的函数。Handoff 是一次函数调用。Orchestrator 是一个调度器。**系统中唯一有状态的东西是共享状态。** 这就是所有有趣 bug 的藏身之处：内存中毒（Lesson 15）、消息排序、版本控制、写入争用。

隐藏共享状态的框架（Swarm）把问题推给调用者。集中化共享状态的框架（LangGraph checkpoint、AutoGen 池）使其可检查，但将协调成本转移到共享状态实现上。

### 单个原语的解剖

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。两个具有相同系统提示词和工具的智能体是可互换的。任何看起来像每个智能体状态的东西，实际上都在共享状态或交接协议中。

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导：

- **函数返回** —— 工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在其工具模式中携带路由信息。
- **图边** —— LangGraph。边是声明式的。LLM 产出一个值；条件选择下一个节点。
- **说话者选择** —— AutoGen GroupChat。一个选择器函数（有时本身是一次 LLM 调用）读取池子并挑选下一个说话者。

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一个消息列表。通常更多：结构化工件（CrewAI 任务输出）、类型化上下文（LangGraph reducer）、外部内存（MCP、向量数据库）。

两种拓扑：**完整池（full pool）**（每个智能体看到每条消息）和**投影（projected）**（智能体看到角色限定视图）。完整池简单但扩展性差。投影池可扩展但需要预先的模式设计。

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **静态（Static）** —— 图在构建时固定（LangGraph 确定性、CrewAI Sequential）。
- **LLM 选择（LLM-selected）** —— 一个 LLM 读取池子并挑选下一个说话者（AutoGen、CrewAI Hierarchical）。
- **交接驱动（Handoff-driven）** —— 当前智能体通过调用交接工具来决定（Swarm）。
- **队列驱动（Queue-driven）** —— 工作者从共享队列中拉取任务；没有显式的下一个说话者（集群架构、Matrix）。

### 框架之间变化的是什么

一旦原语固定，剩下的设计决策是：

- **内存策略** —— 临时 vs 持久化 checkpointing（LangGraph checkpointer）。
- **安全边界** —— 谁可以批准一次交接（human-in-the-loop，人在回路）。
- **成本核算** —— 每个智能体的 token 预算。
- **可观测性** —— 追踪交接、持久化状态以支持回放。

所有这些都可在原语之上实现。没有一个是新的原语。

## 动手实现

`code/main.py` 用约 150 行标准库 Python 实现了四个原语。没有真正的 LLM——每个智能体都是一个脚本策略，以便焦点保持在协调结构上。

该文件导出：

- `Agent` —— 一个包含名称、系统提示词、工具、策略函数的 dataclass。
- `Handoff` —— 一个返回新智能体的函数。
- `SharedState` —— 一个线程安全的消息池。
- `Orchestrator` —— 三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示运行相同的三智能体流水线（研究 → 撰写 → 审查）通过所有三种编排器类型，并在最后打印消息池。你可以看到输出只在*谁挑选下一个*上有所不同；智能体和共享状态在各次运行中是相同的。

运行：

```
python3 code/main.py
```

预期输出：三次编排器运行，每种模式一次。每次都打印最终消息池。如果研究员提前决定完成，交接驱动的运行会到达更少的智能体——这就是 LLM 路由权衡的缩影。

## 学以致用

`outputs/skill-primitive-mapper.md` 是一个技能，它读取任何多智能体代码库或框架文档并返回四原语映射。在接触新框架发布时运行它，以便在深入阅读文档前获得一个段落的理解。

## 落地交付

在采用新框架前，为它编写原语映射。如果你做不到，说明文档不完整，或者该框架正在发明第五个原语（罕见——检查是否有你没见过的共享状态变体）。

将映射钉在你的架构文档中。当新团队成员加入时，在发 API 文档前先发映射。当框架版本变化时，diff 映射而不是 changelog。

## 练习

1. 用不同的智能体策略运行 `code/main.py` 三次。观察编排器选择如何改变哪些智能体运行。
2. 实现第四种编排器类型：队列驱动型，智能体轮询共享状态以获取工作。可能发生什么死锁，你如何检测它？
3. 拿 LangGraph 快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并用四个原语重写它。LangGraph 的哪些抽象是 1:1 映射，哪些是便利包装？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别四个原语中 Swarm 让哪个最符合人体工学，以及它把哪个推给了调用者。
5. 在表中找一个完全隐藏共享状态的框架。解释当智能体需要在不重新读取历史的情况下跨交接协调时，什么会崩溃。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Agent（智能体） | “一个带工具的 LLM” | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| Handoff（交接） | “控制权转移” | 一个结构化调用，命名下一个智能体和可选载荷。三种实现：函数返回、图边、说话者选择。 |
| Shared state（共享状态） | “内存” / “上下文” | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| Orchestrator（编排器） | “协调器” | 决定下一个谁运行的角色。静态图、LLM 选择器、交接驱动，或队列驱动。 |
| Primitive（原语） | “抽象” | 每个框架都会参数化的四个坐标轴之一。不是框架特性。 |
| Message pool（消息池） | “共享聊天历史” | 完整历史的共享状态。易于推理，扩展性差。 |
| Projected state（投影状态） | “限定视图” | 进入共享状态的角色特定视图。可扩展，需要模式设计。 |
| Speaker selection（说话者选择） | “下一个谁说话” | 编排器模式，其中一个函数（通常是 LLM）从一组中挑选下一个智能体。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) —— 对交接驱动编排最清晰的阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) —— GroupChat + 说话者选择是 LLM 选择编排的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) —— 图边编排和基于 reducer 的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) —— role-goal-backstory 智能体，Sequential / Hierarchical 流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) —— Microsoft 将 v0.4 移入维护后，活跃的 AutoGen v0.2 分支
