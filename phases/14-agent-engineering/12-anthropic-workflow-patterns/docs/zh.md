# Anthropic 的工作流模式：简单优于复杂

> Schluntz 和 Zhang (Anthropic，2024 年 12 月) 区分了 workflows (工作流，预定义路径) 与 agents (智能体，动态工具使用)。五种工作流模式覆盖了大多数场景。从直接 API 调用开始。只有在步骤无法预测时才添加 agents。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 分钟

## 学习目标

- 说出 Anthropic 的五种 workflow patterns (工作流模式)：prompt chaining (提示链)、routing (路由)、parallelization (并行化)、orchestrator-workers (编排器-工作者)、evaluator-optimizer (评估器-优化器)。
- 解释 agent-vs-workflow (智能体与工作流) 的区分以及各自的工程成本。
- 识别何时选择工作流而非智能体（以及相反的情况）。
- 在标准库中针对 scripted LLM (脚本化大语言模型) 实现全部五种模式。

## 问题背景

团队在面对单个函数调用就能解决的问题时，却去求助 multi-agent frameworks (多智能体框架)。代价是真实的：框架增加了掩盖 prompts、隐藏控制流并引入过早复杂度的层级。Schluntz 和 Zhang 2024 年 12 月的文章是业界引用最多的反驳：从简单开始，只在复杂度能证明其代价时才添加。

## 核心概念

### 工作流 vs 智能体

- **Workflow (工作流)。** LLM 和工具通过预定义的代码路径进行编排。Engineers (工程师) 拥有图的所有权。
- **Agent (智能体)。** LLM 动态地指挥自己的工具并执行自己的步骤。模型拥有图的所有权。

两者都有各自的适用场景。工作流更便宜、更快、更容易调试。智能体解锁了开放式问题，但使 failure modes (失败模式) 更难推理。

### 增强型 LLM

所有五种模式的基础：一个 LLM 内置三种能力 —— search (检索)、tools (工具)、memory (记忆)。任何 API 调用都可以使用这些能力。

### 五种模式

1. **Prompt chaining (提示链)。** 调用 1 的输出是调用 2 的输入。用于具有清晰线性分解的任务。步骤之间可选 programmatic gates (程序化门控)。

2. **Routing (路由)。** 一个 classifier LLM (分类器大语言模型) 选择调用哪个下游 LLM 或工具。用于需要不同处理方式（一级支持 vs 退款 vs 缺陷 vs 销售）的 categorically different inputs (本质不同的输入)。

3. **Parallelization (并行化)。** 并发运行 N 个 LLM 调用，聚合结果。两种形态：sectioning (分段，不同 chunks) 和 voting (投票，相同 prompt，N 次运行，取多数/综合)。

4. **Orchestrator-workers (编排器-工作者)。** 一个 orchestrator LLM (编排器大语言模型) 动态决定运行哪些 workers (也是 LLM)，并综合它们的输出。与 agent loops (智能体循环) 类似，但 orchestrator 不会无限循环。

5. **Evaluator-optimizer (评估器-优化器)。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到 evaluator 通过。这是 Self-Refine (Lesson 05) 的泛化形式。

### 工作流胜过智能体的场景

- **Predictable tasks (可预测的任务)。** 如果你能枚举步骤，你就应该这么做。
- **Cost-bound tasks (成本受限的任务)。** 工作流有 bounded step counts (有界的步骤数)；智能体可能 spiraling (失控螺旋)。
- **Compliance-bound tasks (合规受限的任务)。** 审计人员希望读取图，而不是从轨迹中推断它。

### 智能体胜过工作流的场景

- **Open-ended research (开放式研究)。** 当下一步取决于上一步返回什么时。
- **Variable-length tasks (可变长度任务)。** 从几分钟到几小时的工作，步骤数未知。
- **Novel domains (新领域)。** 当你还不知道正确的工作流时 —— 先探索，后固化。

### 上下文工程的伴生学科

"Effective context engineering for AI agents" (Anthropic 2025) 将相邻学科形式化：200k 窗口是一个 budget (预算)，而不是容器。包含什么、何时 compact (压缩)、何时让上下文增长。在本课程 Phase 14 关于 context compression (上下文压缩) 的课程中有详细覆盖（Phase 14 早期课程 06，在本次重新编号之前）。

## 动手实现

`code/main.py` 针对 `ScriptedLLM` 实现了全部五种工作流模式：

- `prompt_chain(input, steps)` —— 顺序执行。
- `route(input, classifier, handlers)` —— 分类 + 分发。
- `parallel_vote(prompt, n, aggregator)` —— N 次运行，聚合。
- `orchestrator_workers(task, workers)` —— 编排器选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` —— 循环直到通过。

运行方式：

```
python3 code/main.py
```

每种模式打印其 trace。每种模式的代码约 10-15 行；框架的成本以千行计。

## 如何使用

- 大多数任务使用直接 API 调用。
- 只有当模式真正需要 durable state (持久状态)（LangGraph）、actor-model concurrency (Actor 模型并发)（AutoGen v0.4）或 role templating (角色模板)（CrewAI）时才使用框架。
- 当你想要 Claude Code harness shape 而不重建它时，使用 Claude Agent SDK。

## 输出产物

`outputs/skill-workflow-picker.md` 为给定的任务描述选择正确的模式，包括决策理由以及当工作流不足时向 agent 的迁移路径。

## 练习题

1. 用置信度阈值实现 routing。低于阈值 -> 升级到人工。对于一级支持用例，阈值设在哪里？
2. 为 `parallel_vote` 添加超时。当一个调用挂起时会发生什么？如何处理缺失的投票进行聚合？
3. 将 `evaluator_optimizer` 变成 bandit：在迭代中保留 top-2 输出，这样晚期的好结果不会被晚期坏结果覆盖。
4. 将 prompt chaining 与 routing 结合：一个 router 选择三个 chains 之一。测量 token 成本 vs 单个大型 prompt 替代方案。
5. 选择你的一个生产功能。画出工作流图。数步骤。智能体在这里真的会做得更好吗？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| Workflow | "预定义流程" | 工程师拥有的 LLM 和工具调用图 |
| Agent | "自主 AI" | 模型拥有的图；动态工具指挥 |
| Augmented LLM | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；原子单元 |
| Prompt chaining | "顺序调用" | 调用 N 的输出是调用 N+1 的输入 |
| Routing | "分类器分发" | 选择哪个 chain/model 处理输入 |
| Parallelization | "扇出" | N 个并发调用；通过 sectioning 或 voting 聚合 |
| Orchestrator-workers | "调度器智能体" | Orchestrator LLM 动态选择 specialist LLM |
| Evaluator-optimizer | "提议者 + 评判者" | 迭代直到 evaluator 通过；Self-Refine 的泛化形式 |

## 延伸阅读

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 五种工作流模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 伴生学科
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 何时 stateful graphs 能证明其代价
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) —— orchestrator-workers 模式的产品化
