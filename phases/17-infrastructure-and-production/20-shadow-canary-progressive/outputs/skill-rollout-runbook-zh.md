---
name: rollout-runbook
description: 为新 LLM 模型或提示模板设计 shadow → canary → A/B → 100% 发布计划，含五个 canary gates、噪声底感知阈值和秒级回滚路径。
version: 1.0.0
phase: 17
lesson: 20
tags: [rollout, canary, shadow, progressive-delivery, feature-flags, argo-rollouts, flagger, kserve]
---

给定候选变更（新模型、新提示模板、新路由策略）、基线生产指标和风险容忍度，产出发布 runbook。

产出：

1. Shadow 计划。时长（24-72 小时）。记录指标：输出、token 数、延迟、拒绝、错误。告警条件：成本偏移 >20%、输出长度偏移 >30%、任何 schema 违规。
2. Canary 推进。阶段（1% → 10% → 25% → 50% → 75% → 100%）。每阶段时长（基于流量 30m-24h；确保每阶段有足够数据支撑统计置信度）。
3. 五个 gates。指定 latency P99、cost/request、error/refusal、output-length P99、thumbs-down rate 的精确阈值。设在噪声底之上（预期 15% 不可约 variance）。
4. 工具。命名发布控制器（Argo Rollouts、Flagger、KServe）和用于即时回滚的 feature flag 系统。
5. 回滚路径。记录三个动作：翻转 flag → 恢复 pinned digest → 验证。目标时间：端到端 60 秒内。
6. 跳过 A/B？说明理由。改进型变更跳过 A/B；明显不同的变更（新行为、新成本曲线）需要 A/B。

硬拒绝：
- 跳过 shadow mode。拒绝——成本飙升和长度回退会溜过离线评估。
- Gates 比 15% variance 更紧。拒绝——合法发布会被误报拦截。
- 回滚需要 redeploy。拒绝——那不是回滚，是事故报告。

拒绝规则：
- 如果变更是安全关键（例如 PII 处理变更），要求额外 gate：shadow 样本中零 PII 泄露才能开始 canary。
- 如果流量 <100 req/hour，要求延长 canary 阶段——否则 gate 噪声淹没信号。
- 如果团队无法提供五个 canary gates 的基线指标，拒绝发布——基线是前提。

输出：一页 runbook，含 shadow、canary、gates、工具、回滚、A/B 姿态。结尾附回滚演练要求：首次真实部署前演练一次回滚。
