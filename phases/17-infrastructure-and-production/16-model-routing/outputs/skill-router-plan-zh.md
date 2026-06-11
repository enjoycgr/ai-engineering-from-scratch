---
name: router-plan
description: 设计 LLM 模型路由计划——选择模式（pre-route、cascade、ensemble）、信号（task、length、embedding、confidence）和在线质量关卡。
version: 1.0.0
phase: 17
lesson: 16
tags: [routing, cascade, model-cascade, routellm, notdiamond, cost-reduction]
---

给定工作负载混合（任务分类样本）、质量底线、延迟容忍度和当前月度支出，产出路由计划。

产出：

1. 模式。Pre-route（最快，依赖分类器）、cascade（最佳质量底线）或 ensemble（仅用于样本 A/B）。用质量容忍度 + 延迟预算说明理由。
2. 信号。从以下选择：task classification、prompt length、与 known-hard 的 embedding similarity、self-confidence。说明组合哪些（通常 2-3 个）及组合规则。
3. 廉价/前沿模型对。命名具体模型。示例：Claude Haiku 3.5 + GPT-5。用成本曲线 + 能力说明理由。
4. 预期节省。按推荐拆分计算混合成本；说明预期月度金额 vs 当前。
5. 在线质量关卡。指定实时流量 judge：每路由抽样 5%，由 frontier judge 评估；若 Δ quality > 2% 则告警。跟踪 escalation rate；若一个月内攀升 > 10 个百分点则告警。
6. 发布。Shadow（路由但忽略；离线对比）、按用户 cohort 的 10% canary、通过关卡后扩展。

硬拒绝：
- 无在线质量关卡的路由。拒绝——drift 是头号故障。
- 仅使用 task classification 作为信号。拒绝——遗漏任务内部的难度差异。
- 将 frontier 级任务（代码、数学、多步）不经 cascade fallback 路由到廉价模型。拒绝——质量底线将违规。

拒绝规则：
- 如果质量容忍度声明为"零回退"，拒绝 pre-route 并提议高 escalation rate 的 cascade。
- 如果廉价模型是非 Anthropic/非 OpenAI/非 frontier 且已知有拒绝模式（例如用于 agent tool-use 的未审查模型），拒绝该模型对——它会静默破坏 tool call。
- 如果路由指向不同提供商的廉价模型（跨提供商 cascade），要求 AI gateway 层（Phase 17 · 19）统一 API。

输出：一页计划，含模式、信号、模型对、预期节省、在线关卡、发布计划。结尾附单一指标：滚动 7 天 escalation rate；若变化 > 10 个百分点则触发漂移告警。
