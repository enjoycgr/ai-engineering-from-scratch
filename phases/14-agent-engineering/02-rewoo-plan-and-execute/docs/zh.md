# ReWOO and Plan-and-Execute: Decoupled Planning（解耦规划）

> ReAct 在单一流中交错思考与行动。ReWOO 将它们分离：先制定一个大计划，再执行。token 减少 5 倍，HotpotQA 准确率提升 +4%，并且你可以将 planner 蒸馏到 7B 模型。Plan-and-Execute 将其泛化；Plan-and-Act 将其扩展到网页导航。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 分钟

## Learning Objectives（学习目标）

- 解释为什么 ReWOO 的 Planner / Worker / Solver 拆分比 ReAct 的交错循环更省 token 且更鲁棒。
- 实现一个 plan DAG（计划有向无环图）、依赖排序的 executor（执行器）和组合 worker 输出的 solver（求解器）——全部用 stdlib。
- 使用 2026 年 Anthropic 的"五种工作流模式"框架，决定任务是应该 plan-then-execute（先规划再执行）还是 interleaved ReAct（交错执行）。
- 识别 Plan-and-Act 的合成计划数据何时对长程网页或移动端任务是必需的。

## The Problem（问题）

ReAct 的交错 thought-action-observation 循环简单且灵活，但每次工具调用都必须携带完整的先前上下文——包括每一次先前的思考。token 使用量随深度呈二次增长。更糟的是：当工具在循环中途失败时，模型必须从错误观察中重新推导整个计划。

ReWOO（Xu 等人，arXiv:2305.18323，2023年5月）注意到了这一点并做了一个赌注：先全盘规划，并行获取证据，最后组合答案。一次 LLM 调用制定计划，N 次工具调用获取证据（可并行），一次 LLM 调用求解。代价是灵活性降低（计划是静态的），换来的是更好的 token 效率和更清晰的失败模式。

## The Concept（概念）

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (工具调用，可并行)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 产生一个 DAG。每个节点命名一个工具、其参数，以及它依赖的较早节点（引用如 `#E1`、`#E2`）。Workers 按拓扑序执行节点。Solver 将所有内容缝合在一起。

### 为什么 token 减少 5 倍

ReAct 的 prompt 长度随步数线性增长。第 10 步时，prompt 包含 thought 1 加 action 1 加 observation 1 加 thought 2 加 action 2 加 observation 2，依此类推。每个中间步骤还冗余地包含原始 prompt。

ReWOO 支付一次 planner prompt（较大）、N 次小的 worker prompt（每个仅工具调用，无链条）和一次 solver prompt。在 HotpotQA 上，论文测得 token 减少约 5 倍，同时准确率提升 +4 绝对百分点。

### 为什么更鲁棒

如果 ReAct 中 worker 3 失败，循环必须在中途从错误中推理出来。在 ReWOO 中，worker 3 返回一个错误字符串；solver 在原始计划的上下文中看到它，可以优雅降级。失败定位是 per-node（按节点）的，不是 per-step（按步）的。

### Planner distillation（规划器蒸馏）

论文的第二个结果：因为 planner 不观察 observation，你可以用 175B teacher 的 planner 输出来 fine-tune（微调）一个 7B 模型。小模型负责规划；大模型在推理时不再需要。这现在已成为标准做法——许多 2026 年的生产智能体使用小 planner 和大 executor，反之亦然。

### Plan-and-Execute（LangChain，2023）

LangChain 团队 2023 年8月的文章将 ReWOO 泛化为一个模式名称：Plan-and-Execute。Up-front planner（前置规划器）发出一个步骤列表，executor 执行每个步骤，可选的 replanner（重规划器）可以在观察结果后修订计划。这比 ReWOO 更接近 ReAct（replaner 将 observation 带回了规划），但保留了 token 节省。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长程网页和移动智能体。关键贡献是 synthetic plan data（合成计划数据）：一个 labeled trajectory generator（标注轨迹生成器）产生训练数据，其中计划是显式的。用于 fine-tune（微调）在 WebArena 类任务上能持续工作超过 30–50 步的 planner 模型，而单个 ReAct 轨迹在这些任务上会失去连贯性。

### 何时选择哪种模式

| 模式 | 适用场景 |
|---------|------|
| ReAct | 短任务、未知环境、需要反应式异常处理 |
| ReWOO | 结构化任务、已知工具、对 token 敏感、证据可并行获取 |
| Plan-and-Execute | 类似 ReWOO，但允许部分执行后 replanning |
| Plan-and-Act | 长程（>30 步）、网页/移动/计算机使用 |
| Tree of Thoughts | 值得付出搜索代价时（Lesson 04） |

Anthropic 2024 年12月的指导：从最简单的开始。如果任务是一次工具调用加总结，不要构建 ReWOO。如果任务是 40 步的研究任务，不要单独做 ReAct。

## Build It（动手实现）

`code/main.py` 实现了一个玩具 ReWOO：

- `Planner` —— 一个脚本化策略，从 prompt 发出 plan DAG。
- `Worker` —— 通过注册表分派每个节点的工具调用。
- `Solver` —— 脚本化组合，读取证据并产生最终答案。
- Dependency resolution（依赖解析）—— 引用如 `#E1` 被替换为较早 worker 的输出。

demo 回答 "What is the population of the capital of France, rounded to millions?"，使用两步计划：(1) 查找首都，(2) 查找人口，然后求解。

运行：

```
python3 code/main.py
```

轨迹先显示完整计划，然后显示 worker 结果，然后显示 solver 组合。将 token 数量（我们打印粗略字符数）与 ReAct 式交错运行比较——ReWOO 在这类结构化任务上胜出。

## Use It（应用）

LangGraph 将 Plan-and-Execute 作为 recipe 提供（`create_react_agent` 用于 ReAct，自定义图用于 plan-execute）。CrewAI 的 Flows 直接编码该模式：你先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法仍然主要是研究性的；运行时模式（显式 plan DAG）通过 LangGraph 和 CrewAI Flows 投入生产。

## Ship It（交付）

`outputs/skill-rewoo-planner.md` 根据用户请求和工具目录生成 ReWOO plan DAG。它在移交给 executor 之前验证计划（无环、每个引用已解析、每个工具存在）。

## Exercises（练习）

1. 对独立计划节点并行化 worker 执行。在具有 2 个并行组的 6 节点 DAG 上，这能带来什么好处？
2. 添加一个 replanner 节点，当任何 worker 返回错误时触发。对 ReWOO 做最小的什么改动就能使其变成 Plan-and-Execute？
3. 用一个小模型（7B 级别）替换 `Planner`，保持 `Solver` 在 frontier model 上。比较端到端质量——拆分在哪里失效？
4. 阅读 ReWOO 论文第4节关于 planner distillation。从概念上复现 175B -> 7B 的结果：你需要什么训练数据，以及如何评分计划质量？
5. 将玩具移植到 Plan-and-Act 的轨迹形状：计划是一个序列，不是 DAG。什么权衡发生了变化？

## Key Terms（关键术语）

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| ReWOO | "Reasoning without observations" | 先规划，然后并行获取证据，再求解——规划 prompt 中没有 observation |
| Plan-and-Execute | "LangChain's plan-execute pattern" | 带可选 replanner 节点的 ReWOO |
| Plan-and-Act | "Scaled plan-execute" | 显式 planner/executor 拆分，带合成计划训练数据，用于长程任务 |
| Evidence reference | "#E1, #E2, ..." | 在分派时替换为较早 worker 输出的计划节点占位符 |
| Planner distillation | "Small planner, big executor" | 用 large teacher 的规划轨迹 fine-tune（微调）小模型 |
| Token efficiency | "Fewer round trips" | 论文中 HotpotQA 上比 ReAct 少约 5 倍 token |
| DAG executor | "Topological dispatcher" | 按依赖顺序运行计划节点；每层并行 |

## Further Reading（延伸阅读）

- [Xu 等人, ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) —— 经典论文
- [Erdogan 等人, Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) —— 带合成计划的扩展规划器-执行器
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) —— 框架 recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 选择最简单的有效模式
