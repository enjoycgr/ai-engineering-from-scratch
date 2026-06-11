---
name: workflow-picker
description: 为给定任务选择正确的模式（提示链、路由、并行、编排器-工作者、评估器-优化器，或完整智能体），并生成最小实现。
version: 1.0.0
phase: 14
lesson: 12
tags: [anthropic, workflows, agents, patterns, minimal]
---

给定一个任务描述，选择适合的最小模式并生成最小的正确实现。

决策树：

1. 你能枚举步骤吗？ -> **prompt chaining (提示链)** 或 **routing (路由)**。
2. 输出是否需要跨独立运行的聚合？ -> **parallelization (并行化)**（sectioning 或 voting）。
3. 你是否需要一个 specialist pool (专家池)，其成员随任务变化？ -> **orchestrator-workers (编排器-工作者)**。
4. 你是否需要迭代优化直到 judge (评判者) 通过？ -> **evaluator-optimizer (评估器-优化器)**（Self-Refine 形态）。
5. 以上都不是，或步骤数取决于中间结果？ -> **agent loop (智能体循环)**（Lesson 01）。

生成：

- 对于工作流：组合 LLM + 工具调用的纯函数。不使用框架。
- 对于智能体：来自 Lesson 01 的 ReAct loop 加上任务所需的工具注册表。
- 一个 `README.md`，包含决策理由、步骤数、预期 token 成本和可观察的成功标准。

硬性拒绝：

- 当任务只是一个 3-step prompt chain 时，却去求助框架（LangGraph、AutoGen、CrewAI）。过度工程化掩盖了实际问题。
- 将 3-worker orchestrator-worker 描述为 "multi-agent"。Workers 不是 agents；它们是 LLM 调用。为清晰起见使用 "orchestrator-workers"。
- 没有停止条件的 evaluator-optimizer。没有 `max_iter` 和 "fail-pass-through" 后备，循环可能无限旋转。

拒绝规则：

- 如果用户在任务实际上是 router 时要求 "multi-agent"，拒绝并重新命名。Multi-agent 标签带有运营成本（协调、调试、评估），而 routing 不需要。
- 如果用户为开放式研究任务想要工作流，拒绝并建议使用带有回合预算的 agent。Workflows 适用于可预测的轨迹。
- 如果用户为 2-step 任务想要 agent，拒绝并建议 prompt chaining。Agents 增加延迟和失败模式；只在需要时使用。

输出：模式选择 + 最小代码 + README。以 "what to read next" 结尾，指向 Lesson 13（如果 durable state 很重要），Lesson 16（OpenAI Agents SDK，用于 handoffs 和 guardrails），或 Lesson 01（如果你最终还是选择了 agent）。
