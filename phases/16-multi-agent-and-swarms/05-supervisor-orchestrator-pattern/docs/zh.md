# Supervisor / Orchestrator-Worker 模式

> 一个主导智能体（lead agent）负责规划与委派；专业化工作智能体（worker）在并行上下文中执行并汇报结果。这一模式正是 Anthropic Research 系统的核心（Claude Opus 4 担任主导，Sonnet 4 担任子智能体），在内部研究评估中相比单智能体 Opus 4 提升了 +90.2%。Anthropic 的工程博客指出，BrowseComp 上 80% 的方差仅由 token 用量即可解释——多智能体的优势很大程度上源于每个子智能体都能获得一个全新的上下文窗口（context window）。本课从基础原语出发构建 supervisor 模式，并涵盖 2026 年生产部署中的工程经验。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 分钟

## 问题

研究类任务是单智能体系统最典型的失败场景。当你提问"2023 到 2026 年间多智能体系统发生了什么变化？"时，单个智能体只能顺序阅读五篇论文，用掉一半的上下文容量来存储论文原文，然后再试图综合所有内容。读到第五篇时，它已经忘了第一篇。它也无法并行处理。

Supervisor 模式解决了这个问题：一个主导智能体规划搜索方向，将每个子问题委派给工作智能体，最后进行综合。每个工作智能体都拥有独立的 200k token 窗口来处理一个狭窄的问题。主导智能体永远不需要阅读原始论文——它只接收工作智能体的摘要。

Anthropic 的生产级 Research 系统报告称，相比单智能体 Opus 4，内部研究评估提升了 +90.2%。同一篇文章还指出，BrowseComp 80% 的方差由 *token 用量本身* 解释。每个子智能体拥有独立的上下文是主要机制。

## 概念

### 模式结构

```
                 ┌──────────────┐
                 │   Lead       │  规划、分解、
                 │  (Opus 4)    │  综合
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

主导智能体从不阅读原始材料。工作智能体在主导智能体进行综合之前也看不到彼此的工作。每个箭头都是一次带着精简产物（artifact）的交接（handoff）。

### 为什么它能赢

三种机制：

1. **每个子智能体拥有全新的上下文。** 探索"FIPA-ACL 遗产"的工作智能体不会携带主导智能体在规划阶段消耗的 40k token。它获得一个 200k 的窗口，只处理一个问题。
2. **通过提示词实现特化。** 主导智能体的提示词是"分解并综合"，而不是"研究"。每个工作智能体的提示词都很狭窄："找出 X 中的变化"。聚焦的提示词产生聚焦的输出。
3. **并行性。** 工作智能体并发运行。 wall-clock 时间大致为 `max(worker_times) + plan + synthesis`，而不是 `sum(worker_times)`。

### 工程经验（Anthropic 2025）

Anthropic 的博客列出了几条在 2026 年仍然适用的生产经验：

- **根据查询复杂度调整投入。** 简单查询：一个智能体，3-10 次工具调用。复杂查询：10+ 个智能体。主导智能体必须自行估算，而不是由调用者决定。
- **先广后窄。** 先分解为宽泛的子问题，如果某个子问题的答案值得深入，再为其派生更多工作智能体。
- **彩虹部署（Rainbow deployments）。** 智能体是长期运行且有状态的。传统的蓝绿部署不适用。Anthropic 使用彩虹部署：新版本逐步推出，同时旧版本继续处理存量任务直至排空。
- **Token 用量占主导。** 多智能体的 token 消耗约为单智能体的 ~15 倍。只在任务价值足以覆盖成本时才运行。

### LangGraph 的转变

LangGraph 最初发布了一个 `langgraph-supervisor` 库，提供高层的 `create_supervisor` 辅助函数。2025 年 LangChain 将推荐做法改为直接通过工具调用（tool-calling）实现 supervisor 模式，因为工具调用能更好地控制 *supervisor 能看到什么*（上下文工程）。该库仍然可用；但文档现在推荐工具调用的形式。

### 失败模式

- **主导智能体幻觉出错误的计划。** 如果主导智能体生成的子问题没有真正分解原始问题，工作智能体就会对错误的目标进行精确研究。
- **工作智能体过度探索。** 没有明确的范围边界时，工作智能体会偏离分配的子问题，污染综合步骤。
- **综合冲突。** 两个工作智能体返回了相互矛盾的事实。主导智能体必须要么重新询问（增加一轮），要么明确标注分歧。最糟糕的失败是静默选择一方：用户永远不知道存在分歧。

### 什么时候不适合用 supervisor

- **顺序任务。** 如果步骤 2 字面意义上需要步骤 1 的输出，并行没有任何收益。使用流水线（CrewAI Sequential、LangGraph 线性图）。
- **简单查询。** 单智能体处理得更快更便宜。在派生工作智能体之前，先由主导智能体进行"调整投入"检查。
- **严格确定性要求。** Supervisor 使用 LLM 选择委派目标。当审计/重放比适应性更重要时，静态图更合适。

## 构建

`code/main.py` 使用 `threading` 实现了一个带有三个并行工作智能体的 supervisor。主导智能体将查询分解为子问题，工作智能体在每个子问题上并发运行，最后由主导智能体进行综合。没有真实的 LLM——工作智能体通过脚本模拟获取-摘要过程。

关键结构：

- `Lead.plan(query)` 将查询拆分为 3 个子问题。
- `Worker.run(sub_q)` 返回一个假摘要（在生产环境中可以是任何使用工具的智能体）。
- `Lead.run(query)` 在线程中启动工作智能体，等待它们完成，然后进行综合。

运行：

```
python3 code/main.py
```

输出展示了计划、并行工作智能体的带时间戳的追踪日志，以及最终综合。你可以看到 wall-clock 收益：三个 0.3 秒的工作智能体在约 0.35 秒内完成，而不是 0.9 秒。

## 使用

`outputs/skill-supervisor-designer.md` 接收用户查询并生成 supervisor 模式的设计：主导系统提示词、工作智能体角色、子问题分解规则，以及综合模板。在构建新的研究型智能体系统之前使用它。

## 交付

部署 supervisor 模式前的检查清单：

- **模型配对。** 主导使用推理级模型（Opus 级别、`o3` 级别）。工作智能体使用更快、更便宜的模型（Sonnet、`o4-mini`）。
- **工作智能体超时。** 任何超过中位运行时间 2 倍的工作智能体将被终止；主导智能体要么用更窄的范围重新派生，要么在没有它的情况下继续。
- **每个工作智能体的 token 上限。** 硬限制（例如预期综合输入的 10 倍）防止失控的工作智能体耗尽预算。
- **可观测性。** 追踪主导智能体的计划、每个工作智能体的工具调用，以及综合过程。这是事后调试的基础。
- **彩虹部署（Rainbow rollout）。** 有状态的长期运行智能体需要渐进式版本过渡，而不是热切换。

## 练习

1. 运行 `code/main.py`，然后修改主导智能体使其派生 5 个工作智能体而非 3 个。观察 wall-clock 效果。在这个演示中，工作智能体数量达到多少时，派生开销会超过并行节省的时间？
2. 实现工作智能体超时：终止任何运行超过 0.5 秒的工作智能体，并让主导智能体综合剩余结果。你需要什么样的可观测性才能知道某个工作智能体被切掉了？
3. 在主导智能体的综合步骤中添加冲突检测：如果两个工作智能体返回矛盾的答案，主导智能体标注分歧而不是选择一方。如何在不调用 LLM 的情况下检测矛盾？
4. 阅读 Anthropic 的 Research 系统工程博客。列出这个玩具演示要投入生产需要采纳的三项实践。
5. 比较 LangGraph 的 `create_supervisor`（遗留方案）与新的工具调用推荐做法。哪一种能让你更好地控制 supervisor 能看到什么？为什么 Anthropic 明确只将子答案而非原始工作智能体上下文传入综合步骤？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Supervisor | "主导智能体" | 一个负责规划、委派和综合的编排器（orchestrator）智能体。不亲自执行工作。 |
| Worker | "子智能体" | 由 supervisor 调用、范围狭窄且拥有自己上下文窗口的聚焦型智能体。 |
| Orchestrator-worker | "Supervisor 模式" | 同一件事，不同名称。2026 年的文献两者都用。 |
| Fresh context | "干净的窗口" | 工作智能体的上下文从其系统提示词和分配的问题开始，而非主导智能体的历史记录。 |
| Rainbow deployment | "渐进式发布" | 长期运行且有状态的智能体需要版本化的排空-替换，而非蓝绿部署。 |
| Token dominance | "上下文是关键变量" | 根据 Anthropic，研究评估 80% 的方差来自总 token 用量，而非模型选择。 |
| Scale effort | "让智能体数量匹配复杂度" | 主导智能体估算查询难度，相应地派生 1 个或 10+ 个工作智能体。 |
| Synthesis conflict | "工作智能体意见不一致" | 两个工作智能体返回矛盾的事实；主导智能体必须暴露分歧，而非静默选择一方。 |

## 延伸阅读

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — supervisor 模式的生产级参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 工具调用型 supervisor 现在是推荐形式
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 遗留辅助函数，2026 年生产环境仍在使用
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 基于交接的 supervisor 变体
