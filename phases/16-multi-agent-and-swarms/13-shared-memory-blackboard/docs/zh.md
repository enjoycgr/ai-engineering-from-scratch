# Shared Memory 与 Blackboard 模式（共享内存与黑板模式）

> 2026 年的多智能体系统中并存着两种方法：**message pool（消息池）**（每个智能体都能看到所有智能体的消息，如 AutoGen GroupChat 或 MetaGPT）和 **blackboard with subscription（带订阅的黑板）**（智能体订阅相关事件，如 Context-Aware MCP 或 Matrix 框架）。两者都是多智能体系统中唯一有状态的部分——这意味着有趣的 bug 都出在这里。典型的失效模式是 **memory poisoning（记忆投毒）**：一个智能体幻觉出一个"事实"，其他智能体将其视为已验证，准确性逐渐衰减，这比立即崩溃更难调试。本课用 stdlib 从零构建这两种结构，注入一次投毒攻击，并展示三种在生产环境中实际有效的缓解措施。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 分钟

## Problem（问题）

多智能体系统需要一个地方让智能体共享事实。一个直接的选择是"把所有东西放在消息里传递"——但这用额外复制重新发明了共享状态。另一个是"给每个人一个全局日志"——但全局日志会无界增长且容易被投毒。第三个是"为每个智能体投射一个视图"——可扩展但 schema 繁重。

当其中一个智能体产生幻觉并将幻觉写入共享状态时，每个读取该状态的下游智能体都会将幻觉当作事实采纳。等到人类注意到时，推理链已经深入五步，而根本原因是第三条消息。调试多智能体的准确性衰减比调试崩溃更难。

这就是 memory poisoning（记忆投毒）。它是 MAST 分类法（Cemri 等人，arXiv:2503.13657）中记录第二多的故障家族，而且是结构性的：任何没有 provenance（溯源）和不可写 verifier（验证器）的共享内存设计最终都会表现出来。

## Concept（概念）

### 两种主要拓扑

**Full message pool（完整消息池）。** 每个智能体读取每条消息。AutoGen GroupChat 和 MetaGPT 使用这种方式。简单、透明、可检查，但智能体数量超过约 10 个时无法扩展，因为每个智能体的上下文会被其他智能体的工作填满。

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription（带订阅的黑板）。** 智能体声明对主题的兴趣；底层只路由相关消息。CA-MCP（arXiv:2601.11595）和 Matrix 去中心化框架（arXiv:2511.21686）使用这种方式。扩展性更好，但需要预先设计 schema 才能让订阅有意义。

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### 各自的适用场景

- **Full pool** 在智能体数量少（< 10）、异构、且对话是短视距时胜出。当每个人都看到一切时，推理谁说了什么是微不足道的。
- **Blackboard** 在智能体数量多、角色同质但实例众多（swarm）、且对话长时间运行时胜出。路由节省了 token 成本和上下文污染。

生产系统通常混合使用：顶层（规划层）使用小型 full pool，下层（工作层）使用 blackboards。

### Memory poisoning 场景示例

三个智能体合作完成一个研究任务。智能体 A 是 retrieval agent（检索智能体）。智能体 B 是 summarizer（摘要器）。智能体 C 是 analyst（分析师）。

1. A 获取一个页面并写入共享状态："该研究报告了 42% 的准确率提升。"
2. 实际获取的页面写的是"4.2% 的提升"。A 幻觉了一个小数点。
3. B 读取共享状态，写入："报告了巨大的 42% 准确率提升（来源：A）。"
4. C 读取共享状态，写入："建议采纳——42% 的提升是变革性的。"
5. 最终报告引用了一个从未存在过的 42% 数字。

没有智能体崩溃。没有测试失败。系统"正常工作"。幻觉通过共享状态从一个智能体的上下文进入了每个下游智能体的推理。

### 为什么是结构性的

没有共享状态，智能体 A 的幻觉会留在 A 的上下文中。下游智能体会重新获取或重新推导，可能会发现错误。有了朴素的共享状态，A 的上下文变成了每个人的上下文，幻觉被洗白成事实。

问题不在于共享状态本身——问题在于共享状态**没有 provenance（溯源）和没有独立的 verifier（验证器）**。三种缓解措施解决了这个问题：

1. **在每次写入上标注 provenance（溯源）。** 共享状态中的每个条目记录谁写的、何时写的、在什么提示下写的，以及（如适用）智能体引用的什么来源。下游智能体根据 provenance 带着怀疑阅读。
2. **版本化写入；将它们视为 append-only（仅追加）。** 修正是新的条目，取代旧的，而不是原地更新。审计轨迹被保留。
3. **至少保留一个不能写入共享状态的智能体。** 一个 read-only verifier（只读验证器）智能体抽样条目、重新获取来源、标记不一致。因为它不能写入池，所以不会被池投毒。

### Blackboard 先例（Hayes-Roth, 1985）

Blackboard 模式比 LLM 智能体早四十年。Hayes-Roth（1985，"A Blackboard Architecture for Control"）描述了观察全局 blackboard 的 specialist Knowledge Sources（知识源），贡献 partial solutions（部分解），并触发其他 source。2026 年的 blackboard（CA-MCP、Matrix）是同样的模式，只是用 LLM 智能体作为 Knowledge Sources，用 JSON blob 作为 partial solutions。旧文献中已经记录了写争用、机会主义控制和一致性的解决方案，而现代系统正在重新发现它们。

### Projection vs full view（投影 vs 完整视图）

纯 blackboard 给每个订阅者相同的 projection（投影）（按主题限定范围）。更激进的设计是 **per-agent projection（每智能体投影）**：每个智能体获得为其角色定制的视图。LangGraph 的 state reducers（状态归约器）是 2026 年的典范实现——reducer 函数将全局状态折叠成角色特定的切片。

Per-agent projection 扩展性更好，但需要一个 schema。没有 schema，你就在每个智能体的提示词中重建临时的 projection。

### Write-contention 模式（写争用模式）

多个智能体同时写入是一个并发问题，不仅仅是 LLM 问题。三种模式有效：

- **Sequential writer（单生产者）。** 所有写入都通过一个 coordinator agent（协调器智能体）序列化。简单，但是瓶颈。
- **Optimistic concurrency with versioning（带版本化的乐观并发）。** 每个条目有一个版本；写入者在版本不匹配时失败并重试。经典数据库技术。
- **Topic partitioning（主题分区）。** 不同的智能体拥有不同的主题。没有跨主题争用。需要设计好的分区边界。

大多数 2026 框架默认使用 sequential writer，因为 LLM 调用足够慢，争用很少，瓶颈不会造成影响。

### The unwritable verifier（不可写的验证器）

最承载负载的缓解措施是 read-only verifier。实现规则：

- Verifier 与团队共享状态（读取 blackboard 或 pool）。
- Verifier 没有共享状态的写入句柄——只能写入单独的 verification channel（验证通道）。
- Verifier 独立获取写入中引用的来源。标记不一致。
- Verifier 的输出被路由到人类或单独的 decision agent（决策智能体），从不反馈回池。

没有这种分离，verifier 的输出会变成池中的新条目，这意味着被投毒的池会投毒 verifier，进而投毒其验证结果。

## Build It（动手实现）

`code/main.py` 用 stdlib Python 实现了两种拓扑，外加一个玩具级投毒攻击和三种缓解措施。

- `MessagePool` —— 线程安全的 append-only（仅追加）日志，完整读出。
- `Blackboard` —— 按主题的 pub/sub（发布/订阅），带 per-agent subscriptions（每智能体订阅）。
- `ProvenanceEntry` —— 每次写入记录 (writer, timestamp, prompt_hash, source_uri)。
- `PoisoningScenario` —— 运行一个三智能体研究任务，智能体 A 幻觉一个小数点。打印最终报告。
- `Verifier` —— 一个 read-only agent，重新获取来源并标记不一致。在 verifier 存在的情况下运行相同场景。

运行：

```
python3 code/main.py
```

预期输出：
- 运行 1（无 verifier）：幻觉的 42% 传播到最终报告。
- 运行 2（有 verifier）：verifier 标记不一致，池被标注为 "flagged"，最终报告包含撤回声明。

## Use It（应用）

`outputs/skill-memory-auditor.md` 是一个 skill，用于审计任何多智能体系统的 shared-memory 设计，检查 provenance、versioning 和 verifier separation。在生产环境之前，对新多智能体架构运行它。

## Ship It（交付上线）

对于任何共享内存设计：

- 在每次写入上记录 provenance：`(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`。
- 让日志 append-only。修正是引用被取代条目的新条目。
- 部署至少一个具有独立来源访问权限的 read-only verifier agent。
- 将 verifier 输出路由到单独的通道，而不是反馈回 shared pool。
- 记录 supersessions（取代）占写入的比例——上升的比率是幻觉模式的早期证据。

## Exercises（练习）

1. 运行 `code/main.py`。确认运行 1 传播幻觉，运行 2 捕获它。
2. 添加第二个幻觉：智能体 B 发明了一个数据集大小。Verifier 应该捕获两者，而无需为任一手工调优。
3. 将 full pool 切换为带 topic partitions（主题分区）的 blackboard（`prices`、`summaries`、`analyses`）。主题分区让哪些 poisoning 场景更难实施，哪些它没有帮助？
4. 阅读 Hayes-Roth（1985，"A Blackboard Architecture for Control"）。找出论文中两个本课未讨论、但 2026 系统会受益的控制模式。
5. 阅读 CA-MCP（arXiv:2601.11595）。将其 Shared Context Store 映射到 `code/main.py` 中的 MessagePool 或 Blackboard 类。CA-MCP 在之上添加了哪些原语？

## Key Terms（关键术语）

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Message pool | "Shared chat history" | Append-only log，每个智能体都读取。完全透明，扩展性差。 |
| Blackboard | "Shared workspace" | 按主题的 pub/sub。智能体订阅相关主题。扩展性更好。 |
| Provenance | "Who wrote what" | 每次写入的元数据：writer、timestamp、prompt、sources。 |
| Memory poisoning | "Hallucinations spreading" | 一个智能体的错误进入共享状态，下游智能体将其采纳为事实。 |
| Append-only | "No in-place updates" | 修正是取代旧条目的新条目。保留审计轨迹。 |
| Unwritable verifier | "Independent auditor" | Read-only agent，重新获取来源并标记不一致。 |
| Projection | "Scoped view" | 从全局状态计算的 per-agent view。LangGraph reducers 是 canonical case。 |
| Knowledge Source | "Specialist agent" | Hayes-Roth 1985 年对 blackboard 参与者的术语。 |

## Further Reading（延伸阅读）

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法；memory poisoning 是 coordination-failure（协调失败）子家族
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — 用于协调 MCP 服务器的 Shared Context Store
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — 没有中央编排器的 message-queue-based blackboard
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 生产环境中的 per-agent projection 模式
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产部署中的 provenance 和 verification 笔记
