# Generative Agents and Emergent Simulation (生成式智能体与涌现仿真)

> Park 等人 2023 (UIST '23, arXiv:2304.03442) 用三部分架构填充了 **Smallville**，一个包含 25 个 agent (智能体) 的沙盒：**memory stream (记忆流)**（自然语言日志）、**reflection (反思)**（agent 从自身流中生成的更高层次综合）和 **plan (计划)**（日级行为，然后是子计划）。里程碑式的结果是情人节派对的涌现：一个被植入"想在情人节举办派对"的 agent，在没有进一步脚本的情况下，产生了在人群中传播的邀请，协调了日期，派对发生了——来自 24 个一开始对此一无所知的 agent。消融实验表明，这三个组件对于可信度都是必需的。记录的失败包括空间规范错误（进入关闭的商店、共用单人浴室）。这是 2026 年 agent 仿真和多 agent 社会评估的参考架构。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 分钟

## 问题

大多数多 agent 系统是紧密脚本化的团队：planner 规划、coder 编码、reviewer 审查。这对于定义明确的任务有效。它无法捕捉当 agent 拥有记忆、优先级和开放世界时产生的涌现、非脚本化行为。研究、社会仿真以及越来越多的游戏 AI 需要这第二种。

Smallville 架构是其基准。在 Park 2023 之前，最好的 agent 仿真是浅层的脚本跟随者；之后，该模式成为开放世界中生成式 agent 的默认模式。如果你在 2026 年构建 agent 仿真，你要么使用 Smallville 的三个组件，要么明确解释为什么不使用。

## 概念

### 三个组件

**Memory stream (记忆流)。** 观察、动作、反思和计划的追加日志。每个条目都有时间戳、类型、描述（自然语言）和派生元数据：**recency (新近性)**、**importance (重要性)**（agent 自评 1-10）和 **relevance (相关性)**（与当前查询的余弦相似度）。

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez 问我是否喜欢爵士乐
[2026-02-14 09:14:22] reflection:   我喜欢关于音乐的长时间对话
[2026-02-14 10:05:00] plan:         今晚参加 Isabella 的情人节派对
```

记忆检索结合三个分数：`score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`。Top-k 条目进入当前提示。

**Reflection (反思)。** 定期（每 N 个记忆或重要事件时），agent 从近期记忆中生成更高阶的综合。反思条目回流到流中，并像任何其他记忆一样可检索。这就是 agent 如何构建"理解"——架构中长期信念的等价物。

**Plan (计划)。** 自上而下的分解。首先，一个粗略的日级计划（"去上班，和 Klaus 共进晚餐"）。然后是小时级计划。然后是动作级计划。计划是可修订的：当观察与计划矛盾时，agent 重新规划受影响的部分。

### 为什么三个都重要（消融实验）

Park 等人进行了消融实验，分别去掉了观察、反思和计划。每次消融都会损害可信度：

- 没有 **observation (观察)**，agent 会遗漏上下文，并根据过时的信念行动。
- 没有 **reflection (反思)**，agent 无法形成更高阶的信念；交互保持浅层。
- 没有 **plan (计划)**，行为变成反应性噪声；目标消散。

人类评估者的可信度分数在三个都具备时最高；去掉任何一个都会产生可测量的退化。

### 情人节涌现

一个 agent，Isabella Rodriguez，被植入目标"想在 2 月 14 日下午 5 点在 Hobbs Cafe 举办情人节派对"。其他 24 个 agent 没有收到这样的种子。在模拟的日子里：

1. Isabella 的计划包括邀请人们。
2. 每个邀请成为邻居记忆流中的一个观察。
3. 该邻居的反思产生信念："Isabella 正在举办派对。"
4. 邻居的计划纳入"2 月 14 日参加派对"。
5. 邻居告诉其他邻居。邀请在没有中央协调的情况下传播。
6. 2 月 14 日下午 5 点，几个 agent 汇聚在 Hobbs Cafe。

这是技术意义上的涌现：系统级行为（派对）产生于局部交互（双边邀请 + 个体规划），而没有中央 orchestrator (编排器)。

### 记录的失败模式

Park 等人明确记录了：

- **Spatial norm errors (空间规范错误)。** Agent 走进关闭的商店。Agent 试图使用同一个单人浴室。Agent 在不适合用餐的房间用餐。模型无法仅从环境中推断社会物理规范。
- **Memory overflow (记忆溢出)。** 深度模拟运行导致记忆检索成本增长。实际补救措施：定期记忆压缩（总结-剪枝）和低重要性条目的衰减。
- **Reflection hallucination (反思幻觉)。** 反思可以发明记忆流中不存在的关系。缓解措施：在反思提示中包含源记忆 id，并在检索时验证。

这些是生产相关的失败模式：任何 2026 年的 agent 仿真都会继承它们。

### 三组件实现规则

1. **Memory 是追加的。** 永远不要修改记忆条目。修正是新条目。
2. **Importance 分数很便宜。** 在写入时调用 LLM 对重要性进行 1-10 的评分。缓存分数。
3. **Retrieval 是排序的，不是过滤的。** 按综合分数取 top-k；不要使用硬过滤（会丢失上下文）。
4. **Reflection 定期运行。** 当未处理记忆的重要性总和超过阈值（例如 150）时触发。
5. **Plans 是可修订的。** 当新观察与计划矛盾时，只重新生成受影响的部分，而不是整个计划。

### Smallville 之外的生成式 Agent

2024-2026 年的后续文献扩展了该架构：

- **用于政策/市场研究的多 agent 社会仿真。** 类似 Smallville 的人群模拟用户对新功能的反应。比 A/B 测试更快；准确性有争议。
- **游戏 NPC AI。** 具有 Smallville agent 的 RPG 产生涌现的故事线，而不是脚本化的任务。
- **生成式 agent 评估基准。** 指标不再是任务准确性，而是长期运行的行为的可信度 + 连贯性。

该架构是参考。扩展交换组件（用于记忆的向量存储、检索增强的反思、神经符号计划），但保持三部分结构。

### 为什么这对多 agent 工程很重要

Smallville 是一个概念验证，证明当组件正确时，多 agent 涌现是廉价的。该架构现在已在开源模型上复制（较小的 LLM 优雅地失去可信度，而不是急剧地）。任何需要**涌现社会行为**的生产系统都使用这种形状。任何需要**紧密任务执行**的系统都使用本阶段早期的 supervisor / roles / primitives 模式。

## 构建

`code/main.py` 在 stdlib Python 中实现三个组件，使用脚本化的 agent 策略（没有真正的 LLM）。演示以微型方式重现了情人节派对的涌现：

- `MemoryStream` —— 具有 recency/importance/relevance 检索的追加日志。
- `reflect(stream)` —— 对近期高重要性记忆的脚本化反思。
- `plan(agent_state)` —— 基于当前信念的日级和小时级计划。
- 场景：5 个 agent。Agent 1 以"在下午 5 点举办派对"开始。在模拟的 tick 中，邀请传播并且 agent 汇聚。

运行：

```
python3 code/main.py
```

预期输出：逐 tick 跟踪。在最后一个 tick，5 个 agent 中至少有 3 个将派对显示在他们的计划中，并且他们汇聚在派对地点。单个种子产生了协调到达，没有任何 orchestrator。

## 使用

`outputs/skill-simulation-designer.md` 设计了一个生成式 agent 仿真：agent 数量、记忆 schema、反思节奏、计划范围和评估指标。

## 交付

生产仿真规则：

- **Memory 是数据库。** 大规模时选择真正的存储（向量数据库、Postgres）。内存 stdlib 用于原型。
- **记录检索跟踪。** 对于每个动作，记录驱动它的 top-k 记忆。这是你的调试能力。
- **每个 agent 的 token 预算。** 每个 agent 的每次 tick 的 retrieve + reflect + plan 是 O(k) 次 LLM 调用。N 个 agent × T 个 tick × 每次 tick 的调用数可能会让你的预算相形见绌。
- **定期压缩记忆。** 总结-剪枝低重要性条目。保留策略是一个设计决策，而不是细节。
- **明确检测空间/社会规范违规。** 架构不会学习它们。

## 练习

1. 运行 `code/main.py`。确认 3 个以上的 agent 汇聚在派对。将 agent 增加到 10 个——涌现还会发生吗？
2. 移除 reflection 步骤。行为是什么样的？对应到 Park 2023 中的消融发现。
3. 引入一个竞争性的植入目标（"Klaus 想在下午 5 点做研究报告"）。Agent 会分裂，还是一个目标占主导？什么决定了它？
4. 添加空间约束：Hobbs Cafe 最多容纳 4 个 agent。仿真是否能优雅地处理溢出，还是会碰到"单人浴室"的失败模式？
5. 阅读 Park 等人 (arXiv:2304.03442) 第 6 节（涌现行为实验）。确定一种在你的微型仿真中无法重现的行为。你需要增强架构的哪个组件？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Memory stream | "agent 的日记" | 观察、动作、反思、计划的追加日志。 |
| Recency | "记忆有多新" | 按年龄的指数衰减分数。 |
| Importance | "agent 有多在乎" | 写入时自评 1-10。缓存。 |
| Relevance | "与当前查询有多相关" | 余弦相似度（基于 embedding）。 |
| Reflection | "更高阶的信念" | 从近期记忆生成的综合，作为新记忆重新摄入。 |
| Plan | "日/小时/动作分解" | 自上而下的计划树。当观察矛盾时可修订。 |
| Smallville | "Park 2023 的沙盒" | 25-agent 仿真，产生了情人节涌现。 |
| Believability | "质量指标" | 人类评估者对行为是否像可信 agent 的评分。 |

## 延伸阅读

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) —— 参考架构
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) —— 发表场所
- [Smallville code release](https://github.com/joonspk-research/generative_agents) —— 参考 Python 实现
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) —— 结构化记忆 agent 的先驱
