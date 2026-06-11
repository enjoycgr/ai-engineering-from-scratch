# Orchestration Patterns: Supervisor, Swarm, Hierarchical（编排模式：监督者、蜂群、层级）

> 2026 年框架中反复出现四种编排模式：supervisor-worker（监督者-工作者）、swarm / peer-to-peer（蜂群 / 对等）、hierarchical（层级）、debate（辩论）。Anthropic 的指导："关键在于为你的需求构建合适的系统。"从简单开始；只有当单个智能体加五种工作流模式不够用时，才添加拓扑结构。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 分钟

## Learning Objectives

- 说出四种反复出现的编排模式及其适用场景。
- 描述 2026 年 LangChain 建议：基于工具调用的监督 vs supervisor 库。
- 解释 Anthropic 的"构建合适系统"规则及其如何约束拓扑选择。
- 在 stdlib 中针对一个共同的脚本化 LLM 实现全部四种模式。

## The Problem

团队在需要之前就伸手去拿"多智能体"。四种模式在框架中反复出现；一旦你能命名它们，就能选择正确的——或者完全跳过拓扑结构。

## The Concept

### Supervisor-worker（监督者-工作者）

- 一个中央路由 LLM 将任务分派给专家智能体。
- 决定：循环回自身、移交给专家、终止。
- 专家之间不直接交谈；所有路由都通过监督者。

框架：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 年 LangChain 建议：** 通过直接工具调用而非 `create_supervisor` 进行监督。提供更精细的上下文工程控制——你精确决定每个专家看到什么。

### Swarm / peer-to-peer（蜂群 / 对等）

- 智能体通过共享工具表面直接交接。
- 无中央路由器。
- 延迟低于 supervisor（跳数更少）。
- 更难推理（无单一控制点）。

框架：LangGraph swarm topology、OpenAI Agents SDK handoffs（当所有智能体可以交接给所有其他智能体时）。

### Hierarchical（层级）

- 监督者管理子监督者，子监督者管理工作者。
- 在 LangGraph 中实现为嵌套子图；在 CrewAI 中为嵌套 crews。
- 以操作复杂性为代价扩展到大量智能体群体。

何时需要：当单个监督者的上下文预算无法容纳所有专家的描述时。

### Debate（辩论）

- 并行提案者 + 迭代交叉批判（Lesson 25）。
- 并非真正的编排——更多是验证——但在框架中显示为拓扑选择。

### CrewAI Crew vs Flow

CrewAI 形式化了两种部署模式：

- **Flow** 用于确定性事件驱动自动化（生产推荐的起点）。
- **Crew** 用于自主角色协作。

这与上面的四种模式正交，但映射到拓扑：Flow 通常是 supervisor 或 hierarchical；Crew 通常是带 LLM 路由器的 supervisor。

### Anthropic 的指导

"LLM 领域的成功不在于构建最复杂的系统。而在于为你的需求构建合适的系统。"

决策顺序：

1. 单个智能体 + 工作流模式（Lesson 12）——从这里开始。
2. Supervisor-worker——当你有 2-4 个专家时。
3. Swarm——当延迟比推理清晰度更重要时。
4. Hierarchical——仅当 supervisor 的上下文预算不足时。
5. Debate——当准确性比成本更重要时。

### 这些模式的常见失效点

- **拓扑优先思维。** 在识别多智能体解决什么问题之前就"我们需要多智能体"。
- **Swarm 中的弹跳转交。** A -> B -> A -> B。使用跳数计数器。
- **假层级。** 三层因为"企业级"；实际只有两个团队。扁平化。

## Build It

`code/main.py` 在 stdlib 中针对脚本化 LLM 实现全部四种模式：

- `Supervisor` —— 中央路由器。
- `Swarm` —— 带直接交接的对等。
- `Hierarchical` —— 监督者的监督者。
- `Debate` —— 并行提案者 + 批判。

每种模式处理相同的三意图任务（退款 / 缺陷 / 销售）。轨迹形状不同。

运行方式：

```bash
python3 code/main.py
```

输出：每种模式的轨迹 + 操作计数。Supervisor 最干净；swarm 最短；hierarchical 最深；debate 最昂贵。

## Use It

- **LangGraph** 用于 supervisor 和 hierarchical（嵌套子图）。
- **OpenAI Agents SDK** 用于 handoffs-as-tools（supervisor 形态）。
- **CrewAI Flow** 用于生产确定性。
- **Custom** 用于 debate 或当你想要精确控制时。

## Ship It

`outputs/skill-orchestration-picker.md` 选择拓扑并实现它。

## Exercises

1. 将 supervisor-worker 转换为 swarm，去掉路由器。什么会坏？什么会改善？
2. 给 swarm 添加跳数计数器：3 次交接后拒绝。它能捕捉到 A->B->A 弹跳吗？
3. 为一个 12 专家领域构建一个两级层级系统。没有嵌套时上下文预算在哪里失败？
4. 在生产级工作负载上分析四种模式。每种在哪个指标上获胜（延迟、成本、准确性、可调试性）？
5. 阅读 Anthropic 的"Building Effective Agents"文章。将你的每个生产流程映射到四种之一。有没有映射不干净的？

## Key Terms

| 术语 | 行业说法 | 实际含义 |
|------|----------|----------|
| Supervisor-worker | "路由器 + 专家" | 中央 LLM 分派给专家；专家之间不直接交谈 |
| Swarm | "对等" | 通过共享工具直接交接；无中央路由器 |
| Hierarchical | "监督者的监督者" | 用于大规模群体的嵌套子图 |
| Debate | "提案者 + 批判" | 并行提案者、交叉批判（Lesson 25） |
| Tool-call-based supervision | "不用库的监督者" | 将 supervisor 实现为直接工具调用以控制上下文 |
| Crew | "自主团队" | CrewAI 的角色协作模式 |
| Flow | "确定性工作流" | CrewAI 的事件驱动生产模式 |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 五种模式 + 智能体 vs 工作流
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor、swarm、hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate 模式
