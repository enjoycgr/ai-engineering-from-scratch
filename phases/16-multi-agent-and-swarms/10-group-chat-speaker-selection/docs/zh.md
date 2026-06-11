# Group Chat 与 Speaker Selection（群聊与发言者选择）

> AutoGen GroupChat 与 AG2 GroupChat 让 N 个智能体共享同一段对话；selector 函数（LLM、轮询或自定义）决定下一个由谁发言。这是涌现式多智能体对话的原型——智能体不知道自己在静态图中的角色，它们只是对共享池（shared pool）做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中被保留；AutoGen v0.4 将其重写为事件驱动的 actor 模型。微软于 2026 年 2 月将 AutoGen 置于维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月 RC）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中均得以延续——学会一次，到处可用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 分钟

## Problem（问题）

静态图（LangGraph）在工作流已知时表现很好。真实对话并非如此静态：有时是 coder 询问 reviewer，有时是 researcher，有时是 writer。硬编码所有可能的交接会导致边爆炸（edge explosion）。你需要的是*智能体对共享池做出反应*，并由某个函数决定下一个谁发言。

这正是 AutoGen GroupChat 所做的。

## Concept（概念）

### 结构

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个智能体都能看到每条消息。每轮调用 selector 函数来挑选下一个发言者。

### 三种 selector 风格

**Round-robin（轮询）。** 固定循环。确定性。随 N 线性扩展，但忽略上下文——即使话题是法律审查，coder 也会轮到发言。

**LLM-selected（LLM 选择）。** 调用 LLM 读取最近的池并返回最佳下一位发言者。具备上下文感知能力，但较慢：每轮增加一次 LLM 调用。AutoGen 的默认方式。

**Custom（自定义）。** 可写入任意逻辑的 Python 函数。典型做法：LLM-selected 加回退规则（例如，"coder 发言后总是把轮次给 verifier"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有 selector。当某个智能体完成一轮后，manager 调用 selector，selector 返回下一个智能体。循环持续直到满足终止条件。

### Termination（终止）

三种常见模式：

- **Max rounds（最大轮数）。** 总轮数的硬上限。
- **"TERMINATE" token（终止标记）。** 智能体可以发出一个哨兵消息；manager 检测到后即停止。
- **Goal-reached check（目标达成检查）。** 每轮运行一个轻量级 verifier，任务完成时停止对话。

### AutoGen → AG2 分叉与 Microsoft Agent Framework 合并

2025 年初，微软开始围绕事件驱动 actor 模型对 AutoGen 进行大规模重写（v0.4）。社区将 AutoGen v0.2 的 GroupChat 语义分叉为 AG2，保留了早期采用者已集成的 API。

2026 年 2 月，微软宣布 AutoGen 进入维护模式，事件驱动 actor 模型将合并入 **Microsoft Agent Framework**（2026 年 2 月 RC，现已与 Semantic Kernel 合并）。GroupChat 概念在两条路线中均得以延续；实现细节有所不同。对于 v0.2 兼容代码，AG2 是首选的上游版本。

### GroupChat 的适用场景

- **Emergent conversations（涌现式对话）。** 你不想预先连接每一个可能的下一位发言者。
- **Role-mixing tasks（角色混合任务）。** Coder 询问 researcher，researcher 询问 archivist，archivist 又回头询问 coder。流程不是 DAG。
- **Exploratory problem-solving（探索性问题解决）。** 想象"头脑风暴会议"，而非"流水线"。

### 不适用场景

- **Strict determinism（严格确定性）。** LLM selector 可能不一致。相同提示，不同运行，产生不同的下一位发言者。
- **Sycophancy cascades（谄媚级联）。** 智能体顺从发言最自信的那位。需通过显式 prompt 来对抗。
- **Context bloat（上下文膨胀）。** 每个智能体读取每条消息；10 轮之后上下文变得巨大。使用 projection（第 15 课）来限定视图范围。
- **Hot speakers（热点发言者）。** 某个智能体主导对话，因为 selector 偏爱其专长。在 selector 中引入发言者平衡机制。

### Group chat vs supervisor（群聊 vs 监督者）

相同原语，不同默认设置：

- Supervisor：一个智能体做计划，其他执行。Selector 是"询问 planner 该做什么"。
- Group chat：所有智能体对等；selector 是基于共享池的函数。

两者都使用第 04 课的四个原语。Group chat 默认使用 LLM-selected 编排和 full-pool 共享状态。

## Build It（动手实现）

`code/main.py` 用 stdlib 从零实现了一个 GroupChat。三个智能体（coder、reviewer、manager），轮询和 LLM-selected 两种变体，以及基于 `TERMINATE` token 的终止。

演示会打印对话记录以及两种变体下 selector 的决策轨迹。

运行：

```
python3 code/main.py
```

## Use It（应用）

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat selector——轮询 vs LLM-selected vs 自定义，以及使用哪些 selector 输入（最近消息、智能体专长、轮次计数）。

## Ship It（交付）

检查清单：

- **Max rounds cap（最大轮数上限）。** 始终设置。典型任务 10-20 轮。
- **Speaker-balance metric（发言者平衡指标）。** 跟踪每个智能体的轮次；当不平衡超过阈值时告警。
- **Termination token（终止标记）。** `TERMINATE` 或一个专用的 verifier 智能体。
- **Projection or scoped memory（投影或限定范围的记忆）。** 大约 10 条消息后，考虑只给每个智能体提供限定视图，以防止上下文膨胀。
- **Selector logging（Selector 日志）。** 对于 LLM-selected 变体，记录 selector 的输入和选择。否则无法调试。

## Exercises（练习）

1. 运行 `code/main.py`。比较轮询与 LLM-selected 下的对话。每种方式下哪个智能体主导了对话？
2. 在 selector 中添加 "max-speaks-per-agent" 规则。它对对话记录有何影响？
3. 实现一个 goal-reached 终止条件：当 reviewer 返回 "approved" 时停止。它在轮数上限之前触发的频率如何？
4. 阅读 AutoGen 稳定版文档中的 GroupChat（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。识别 `GroupChatManager` 使用的默认 selector。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2）并比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 在具体属性（吞吐量、容错性、可组合性）上增加了什么？

## Key Terms（关键术语）

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| GroupChat | "Agents in one chat room" | 共享消息池（shared message pool）+ selector 函数。AutoGen / AG2 原语。 |
| Speaker selection | "Who talks next" | 挑选下一个智能体的函数。轮询（round-robin）、LLM-selected 或自定义（custom）。 |
| GroupChatManager | "The meeting host" | AutoGen 组件，持有 selector 并循环轮次。 |
| ConversableAgent | "The base agent" | AutoGen 基类；可以发送和接收消息的智能体。 |
| Termination token | "The 'stop' word" | 哨兵字符串（通常是 `TERMINATE`），用于结束对话。 |
| Hot speaker | "One agent dominates" | 失效模式：selector 持续选择同一个智能体。 |
| Context bloat | "Pool grows unbounded" | 每个智能体读取每条先前的消息；上下文随轮次增长。 |
| Projection | "Scoped view" | 针对共享池的角色特定视图，用于防止上下文膨胀。 |

## Further Reading（延伸阅读）

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 repo](https://github.com/ag2ai/ag2) — 社区版 AutoGen v0.2 延续
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — 合并后的继任者，2026 年 2 月 RC
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — 事件驱动 actor 模型重写详情
