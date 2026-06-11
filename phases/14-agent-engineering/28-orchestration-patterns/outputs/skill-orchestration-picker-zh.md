---
name: orchestration-picker
description: 为给定问题选择最小编排拓扑（supervisor、swarm、hierarchical、debate 或 none）并实现它。
version: 1.0.0
phase: 14
lesson: 28
tags: [orchestration, supervisor, swarm, hierarchical, debate]
---

给定产品领域和任务类别，选择最小拓扑。

决策：

1. 1 个智能体 + 工作流模式（Lesson 12）足够？-> 完全不使用拓扑。
2. 2-4 个专家，职责分明？-> **supervisor-worker**。
3. 延迟关键且专家能干净地交接？-> **swarm**。
4. 10+ 个专家，监督者上下文预算不足？-> **hierarchical**。
5. 准确性比成本重要，多提案者 + 批判有帮助？-> **debate**（Lesson 25）。

产出：

1. 所选拓扑的脚手架。
2. Swarm 的跳数计数器；hierarchical 的嵌套深度限制；debate 的轮次上限。
3. 每次交接或每步的可观测性钩子（OTel GenAI spans，Lesson 23）。
4. 一段"为什么选这个而非那个"的 README。

硬性拒绝：

- 将 3 个顺序 LLM 调用称为"多智能体"。那是提示链。
- 没有跳数计数器的 Swarm。弹跳是注定的。
- 每个分支最终只有 1 个专家的 Hierarchical。扁平化。

拒绝规则：

- 如果用户想要用多智能体处理单个 ReAct 循环就能处理的任务，拒绝并建议 Lesson 01。
- 如果用户想要为 2 步任务使用 supervisor，拒绝并建议提示链（Lesson 12）。
- 如果领域有合规 / 审计要求，拒绝 swarm 并建议 supervisor 或 hierarchical。

输出：拓扑脚手架 + 带决策理由的 README。结尾附上"接下来读什么"，指向 Lesson 13（LangGraph）了解 supervisor 实现，Lesson 16（OpenAI Agents SDK）了解 handoffs-as-tools，或 Lesson 25 了解 debate 细节。
