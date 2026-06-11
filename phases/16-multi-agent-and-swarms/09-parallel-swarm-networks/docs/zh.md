# 并行 / 蜂群（Swarm）/ 网络化架构

> 与 supervisor 形成对比：没有中央决策者。智能体读取共享事件总线（event bus），异步拾取工作，写回结果。LangGraph 明确支持用于去中心化、动态环境的"Swarm Architecture"。Matrix（arXiv:2511.21686）将控制流和数据流都表示为通过分布式队列传递的序列化消息，以消除编排器（orchestrator）瓶颈。这种权衡是明确的：用确定性和可追溯性换取可扩展性。Swarm 适合具有大量独立子问题的任务；不适合需要单一连贯计划的任务。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 分钟

## 问题

Supervisor 可以扩展到几个工作智能体。那几百个呢？Supervisor 本身成为瓶颈：关于谁做什么的每一个决策都要通过一个智能体。一个缓慢的计划步骤就会拖垮整个系统。

Swarm 架构翻转了设计。不是中央规划器派发工作，而是工作者从共享队列中拾取工作。"协调"被烘焙进事件总线的语义中。没有编排器；系统的可扩展性只受限于队列本身。

## 概念

### 结构形态

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

没有编排器。每个工作者重复：拉取任务、处理、写回结果（并可选择性地将后续任务入队）。

### 什么时候适合用 swarm

- **大量独立任务。** 爬取、转换、分类。任务之间互不依赖。
- **可变持续时间的工作。** 如果有些任务耗时 100ms，有些耗时 10s，swarm 会自动平衡负载——快的工作者拉取下一个作业。Supervisor 则必须预先估计持续时间。
- **吞吐量优先于确定性。** 你关心总完成时间，而非严格顺序。

### 什么时候 swarm 会失败

- **有序工作流。** 如果步骤 3 需要步骤 2 的输出，swarm 存在步骤 3 在步骤 2 完成前就触发的风险。
- **全局计划类任务。** 复杂的研究问题受益于规划器。一群研究者智能体产生的是独立事实，而非连贯的报告。
- **调试。** 没有中央日志且工作异步，复现 bug 的成本很高。

### Matrix（arXiv:2511.21686）

Matrix 是 2025 年的论文，将 swarm 推向自然结论：控制流和数据流都是分布式队列上的序列化消息。没有中央协调器。容错来自消息持久化。可扩展性是消息代理的问题，而非系统的问题。

贡献：一种编程模型，其中多智能体协调是"这个智能体订阅什么消息主题？"而非"supervisor 接下来选哪个智能体？"这让系统看起来像一个发布/订阅（pub/sub）事件网格。

### LangGraph 的 Swarm Architecture

LangGraph 2025 文档明确将"Swarm Architecture"描述为多智能体模式之一：智能体是节点，但边构成带有循环的有向图，任何节点都可以从池中激活。工作者按条件而非 supervisor 分配来选取可用工作。

### 失败模式：饥饿与热点

如果所有工作者都拉取最快可用的任务，长时间运行的任务永远不会被拾取，直到它们成为队列中仅剩的任务。经典的队列饥饿（starvation）。

缓解措施：
- 带显式老化（aging）的优先级队列（随等待时间增加优先级）。
- 工作者特化：某些工作者只处理"长"任务。
- 背压（back-pressure）：限制进入队列的快任务数量。

### 与基于内容的路由的关联

Swarm 与基于内容的路由（第 22 课）天然配对。不是通用队列，而是每种消息类型一个队列。专业工作者只订阅自己的类型。这是消息总线架构扩展到数千个智能体的基础。

## 构建

`code/main.py` 实现了 4 个工作线程（worker threads）从一个共享的 `queue.Queue` 拉取任务。任务具有可变持续时间（有些快，有些慢）。演示对比了：

- **顺序基线：** 一个工作者串行处理所有任务。
- **固定分配：** 每个任务预先分配给特定工作者（supervisor 风格）。
- **Swarm：** 工作者从共享队列拉取。

Swarm 自动平衡负载；固定分配会在分配给快工作者的任务是慢任务时让其空闲。

运行：

```
python3 code/main.py
```

输出显示每个工作者的任务计数（swarm 不均匀但最优地分配）和 wall-clock 时间。

## 使用

`outputs/skill-swarm-fit.md` 评估一个任务应该使用 swarm 还是 supervisor。输入：任务独立性、持续时间方差、顺序要求、调试需求。

## 交付

检查清单：

- **带老化的优先级队列。** 防止长任务饥饿。
- **工作者幂等性（idempotency）。** 如果工作者在运行中崩溃，任务可能被拉取多次。工作者必须是幂等的。
- **持久化队列。** 生产环境使用 Kafka、Redis Streams 或数据库-backed 队列。`queue.Queue` 仅存在于内存中。
- **每个任务的可观测性。** 每个任务都有一个 trace ID；每个工作者记录带该 ID 的开始/结束时间。
- **背压（Back-pressure）。** 如果队列增长速度超过工作者消耗速度，减慢生产者。

## 练习

1. 运行 `code/main.py`。在可变持续时间负载下，swarm 比顺序执行快多少？比固定分配快多少？
2. 添加一个优先级队列变体（使用 `queue.PriorityQueue`）。按任务的"重要性"字段分配优先级。观察在持续负载下低优先级任务是否会饥饿。
3. 实现一个热点检测器：当任何工作者处理的任务数量是最慢工作者的 3 倍时记录日志。这说明了任务持续时间分布的什么问题？
4. 阅读 Matrix 论文（arXiv:2511.21686）摘要和第 3 节。找出 Matrix 接受的一个具体权衡（可扩展性收益）和放弃的一个方面（可追溯性、确定性）。
5. 将 swarm 演示改为使用 `(task_type, payload)` 元组的 `queue.Queue`，工作者只订阅特定类型。当任务异构时，什么样的路由规则是合理的？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Swarm architecture | "去中心化智能体" | 工作者从共享队列拉取；没有中央编排器。 |
| Event bus | "智能体订阅主题" | 按类型或内容将任务路由给工作者的消息代理。 |
| Starvation | "任务永远不运行" | 低优先级任务因为高优先级工作持续到达而永远不被拾取。 |
| Hot-spotting | "一个工作者被压垮" | 负载失衡，一个工作者获得大部分任务。 |
| Back-pressure | "减慢生产者" | 当队列填满时向上游发出信号停止生产的机制。 |
| Idempotent worker | "安全重跑" | 任务处理两次产生相同结果。必需，因为工作者可能在运行中崩溃。 |
| Durable queue | " survive crashes" | 由磁盘或复制存储支持的队列；工作者崩溃时任务不会丢失。 |
| Matrix framework | "全消息传递 swarm" | 数据流和控制流都是分布式队列上的序列化消息。 |

## 延伸阅读

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 明确的 swarm 支持
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — 全消息传递 swarm
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — 为什么特定生产系统明确选择 supervisor 而非 swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — 事件驱动的 actor 重写，比 v0.2 的 GroupChat 更接近 swarm
