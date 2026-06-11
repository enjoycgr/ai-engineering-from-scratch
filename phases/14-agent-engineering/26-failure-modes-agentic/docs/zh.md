# Failure Modes: Why Agents Break（故障模式：智能体为何崩溃）

> MASFT（Berkeley, 2025）将 14 种多智能体（multi-agent）故障模式归入 3 个类别。Microsoft 的分类体系（Taxonomy）记录了现有 AI 故障如何在 agentic（具备自主行动能力的）环境中被放大。行业现场数据收敛到五种反复出现的模式：幻觉行为（hallucinated actions）、范围蔓延（scope creep）、级联错误（cascading errors）、上下文丢失（context loss）、工具误用（tool misuse）。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 分钟

## Learning Objectives

- 说出 MASFT 的三个故障类别，并在每个类别中至少列出四种具体模式。
- 解释为什么 agentic 故障会放大现有的 AI 故障模式（偏见、幻觉）。
- 描述五种行业反复出现的模式及其缓解措施。
- 实现一个 stdlib 检测器，为智能体轨迹（agent traces）打上故障模式标签。

## The Problem

团队部署的智能体在 90% 的轨迹上运行正常。剩下的 10% 并非随机噪声——它们可归入少数几种反复出现的类别。一旦你能命名它们，就可以监控并修复它们。

## The Concept

### MASFT（Berkeley, arXiv:2503.13657）

Multi-Agent System Failure Taxonomy（多智能体系统故障分类法）。14 种故障模式聚类为 3 个类别。标注者间 Cohen's Kappa 0.88——类别可靠可区分。

核心主张：故障是多智能体系统中的根本性设计缺陷，而非通过更好的基础模型就能解决的 LLM 局限。

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 现有 AI 故障（偏见、幻觉、数据泄漏）在 agentic 设置中被放大。
- 自主性带来的新故障：大规模非预期行为、工具误用、任务漂移（mission drift）。
- 该白皮书是 agentic 产品的风险登记册。

### Characterizing Faults in Agentic AI（arXiv:2603.06847）

- 故障源于编排（orchestration）、内部状态演化以及环境交互。
- 并非简单的"代码烂"或"模型输出差"。

### LLM Agent Hallucinations Survey（arXiv:2509.18970）

两种主要表现形式：

1. **Instruction-following Deviation（指令遵循偏差）**——智能体不遵循系统提示（system prompt）。
2. **Long-range Contextual Misuse（长程上下文误用）**——智能体遗忘或错误应用早期轮次中的上下文。

子意图错误（Sub-intention errors）：遗漏（Omission，漏掉步骤）、冗余（Redundancy，重复步骤）、错乱（Disorder，步骤顺序错误）。

### 五种行业反复出现的模式

Arize、Galileo、NimbleBrain 2024-2026 现场分析趋同于：

1. **Hallucinated actions（幻觉行为）。** 智能体调用不存在的工具或虚构参数。
2. **Scope creep（范围蔓延）。** 智能体将任务扩展到用户请求之外（创建额外 PR、发送额外邮件）。
3. **Cascading errors（级联错误）。** 一次错误调用触发下游影响。一个幻影 SKU 幻觉触发四次 API 调用——多系统事故。
4. **Context loss（上下文丢失）。** 长程任务遗忘早期轮次的约束。
5. **Tool misuse（工具误用）。** 用错误参数调用正确工具，或完全调用错误工具。

级联（Cascading）是最致命的。智能体无法区分"我失败了"与"任务不可能完成"，并且常在 400 错误上幻觉出一条成功消息来闭环。

### 缓解措施：每一步设闸

在推理链的每一步设置自动验证闸，针对环境状态检查事实依据。具体包括：

- 每步安全分类器（Lesson 21）。
- 工具调用参数验证（Lesson 06）。
- 将检索内容与已知事实交叉核对（Lesson 05, CRITIC）。
- 通过重新探测状态来检测成功幻觉（文件真的创建了吗？）。

### 故障监控的常见错误

- **仅标记崩溃。** 大多数智能体故障会产生看起来有效的输出。需要内容级检查。
- **无基线。** 漂移检测需要上次已知良好的状态；没有它你就不能说"情况在恶化"。
- **过度告警。** 每次失败都触发页面。需要聚类和速率限制。

## Build It

`code/main.py` 实现了一个 stdlib 故障模式标记器：

- 覆盖五种模式的合成轨迹数据集。
- 每种模式的检测函数（工具调用上的签名模式、输出、重复动作）。
- 标记器为每条轨迹打标签并报告模式分布。

运行方式：

```bash
python3 code/main.py
```

输出：每条轨迹的标签 + 聚合分布，低成本复现 Phoenix 的轨迹聚类功能。

## Use It

- **Phoenix** 用于生产漂移聚类（Lesson 24）。
- **Langfuse** 用于会话回放 + 标注。
- **Custom** 用于领域特定的签名，你的可观测性平台无法检测到的那些。

## Ship It

`outputs/skill-failure-detector.md` 生成针对你领域的故障模式检测器，并接入轨迹存储。

## Exercises

1. 添加一个"成功幻觉"检测器：智能体返回成功但目标状态未改变。
2. 为你构建的产品标注 100 条真实轨迹。哪种模式占主导？修复它的成本是多少？
3. 实现一个"级联半径"指标：给定第 N 步的失败，它影响了多少下游步骤？
4. 阅读 MASFT 的 14 种故障模式。挑选三种适用于你产品的模式，编写检测器。
5. 将一个检测器接入 CI 任务：如果 >=5% 的轨迹标记了某种模式，则构建失败。

## Key Terms

| 术语 | 行业说法 | 实际含义 |
|------|----------|----------|
| MASFT | "多智能体故障分类法" | Berkeley 的 14 模式分类 |
| Cascading error | "涟漪故障" | 一个早期错误传播到 N 个步骤 |
| Context loss | "忘了约束" | 长程轮次丢弃早期轮次的事实 |
| Tool misuse | "工具/参数错误" | 调用有效但调用方式错误 |
| Success hallucination | "伪造完成" | 智能体在 400 上声称成功；状态未变 |
| Scope creep | "过度延伸" | 智能体做了比要求更多的事 |
| Instruction-following deviation | "不服从" | 忽略系统提示或用户约束 |
| Sub-intention errors | "计划缺陷" | 计划执行中的遗漏、冗余、错乱 |

## Further Reading

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 种故障模式，3 个类别
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — 风险登记册
- [Arize Phoenix](https://docs.arize.com/phoenix) — 实践中的漂移聚类
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时更简单的模式能完全避免故障模式
