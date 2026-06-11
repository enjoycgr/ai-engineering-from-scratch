---
name: refine-loop
description: 根据任务、验证器可用性和迭代预算配置 evaluator-optimizer（Self-Refine / CRITIC）循环。
version: 1.0.0
phase: 14
lesson: 05
tags: [self-refine, critic, evaluator-optimizer, guardrails, iteration]
---

给定任务、迭代预算和可用验证器（tool-grounded 或仅 self-eval），发出 evaluator-optimizer 循环的 prompt 和停止策略。

产出内容：

1. Generator prompt（生成器 prompt）。第一次输出的确定性生产者。明确说明任务、输出格式和约束。
2. Evaluator/verifier prompt（评估器/验证器 prompt）。如果工具可用（搜索、代码运行、测试、计算器、类型检查），指定如何调用它们以及如何产生 structured critique（结构化批评）（JSON：pass/fail、violations[]、suggested_fixes[]）。如果只有 self-eval 可用，明确标记 Self-Refine rubber-stamp 风险，并使用结构上不同的 prompt 风格（例如对抗式 "find at least one flaw"）。
3. Refiner prompt（精炼器 prompt）。必须引用先前输出和 critiques（history）。声明"不要重复先前迭代中标记的失败模式"是强制性的。
4. Stop policy（停止策略）。合取式：verifier 通过 OR（self-eval 说没问题 AND 迭代次数 >= 2）OR 迭代次数 >= max_iterations。永远不要单一条件。
5. Observability hooks（可观察性钩子）。按照 Lesson 23，将每次迭代记录为 OpenTelemetry GenAI span（evaluate、optimize），使完整 refine 轨迹可审计。

Hard rejects（硬拒绝）：

- 生成器和 critic 使用相同 prompt。Rubber-stamp 风险——模型同意自己。
- 没有迭代上限。无限 refine 循环烧 token；默认总是上限为 4。
- 要求 freeform prose feedback 的验证器 prompt。仅限结构化 JSON —— pass/fail 加 itemized violations。
- 从 refiner prompt 中丢弃 history。论文显示没有它质量会崩溃。

Refusal rules（拒绝规则）：

- 如果任务没有验证器也无法构建一个，拒绝 CRITIC 并说明 Self-Refine 是可用的较弱选项——警告用户 rubber-stamp 风险。
- 如果 max_iterations >= 10，拒绝并建议重新架构任务。超过 3-4 轮的 refine-to-convergence 通常是生成器 prompt 错误的信号。
- 如果验证器调用破坏性工具（shell、git write），拒绝并要求沙箱边界（Lesson 09）。

输出：一个包含所有 prompt、停止策略和工具列表的单一配置块，加一个基于部署目标的 "what to read next" 注释：指向 Lesson 16（OpenAI Agents SDK guardrails）、Lesson 12（Anthropic evaluator-optimizer）或 Lesson 30（eval-driven agent development）。
