---
name: ai-sre-plan
description: 为团队设计 AI SRE 推广方案 —— 多智能体分类架构、结构化运维手册、对抗性评估、窄范围自动修复和预测性检测策略。
version: 1.0.0
phase: 17
lesson: 23
tags: [ai-sre, multi-agent, runbooks, auto-remediation, adversarial-eval, datadog-bits-ai, neubird, predictive]
---

给定团队规模、事件量、可观测性成熟度和风险承受能力，产出 AI SRE 计划。

产出：

1. 架构。多智能体：监督器 + 日志智能体 + 指标智能体 + 运维手册智能体 + 人类关卡。将专用智能体与现有数据源匹配（Datadog、Grafana、Loki、Confluence）。
2. 运维手册转型。从非结构化 Confluence 转为带 symptom / hypothesis / verify / act 章节的结构化 Markdown。使用 git 版本控制。
3. 产品选择。Datadog Bits AI、Azure SRE Agent、NeuBird Hawkeye、Incident.io Autopilot 或自建。
4. 自动修复范围。窄安全集（重启 Pod、回滚部署、范围内扩展）。明确拒绝列表（拓扑、代码、IAM、数据库）。策略即代码。
5. 对抗性评估。为自动修复指定双模型一致性关卡。不一致时升级。
6. 预测性检测策略。如果考虑（MIT 89% 成果），命名执行策略 —— 告警、预先排空、自动扩展 —— 否则它只是仪表盘。

硬性拒绝：
- 对广范围变更进行无人类关卡的自动修复。拒绝 —— 明确命名安全集。
- 使用非结构化运维手册作为知识库。拒绝 —— 要求结构化、版本化的 Markdown。
- "设置好就不用管"的表述。拒绝 —— 明确界定什么是自主的、什么不是。

拒绝规则：
- 如果事件量 <10/月，拒绝完整 AI SRE 推广 —— 成本超过收益。仅推荐结构化运维手册。
- 如果团队可观测性不成熟（日志不可搜索、指标稀疏），拒绝 —— AI SRE 会放大坏数据。
- 如果团队提议将"预测性检测 → 自动修复"作为首个功能，拒绝 —— 先走一遍执行策略问题。

产出：一页计划，包含架构、运维手册计划、产品选择、自动修复范围、对抗性关卡、预测性策略。最后附 12 周推广时间表：第 1-4 周结构化运维手册，第 5-8 周分类智能体，第 9-12 周窄范围自动修复。
