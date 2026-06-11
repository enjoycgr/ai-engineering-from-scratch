---
name: finops-plan
description: 设计 LLM FinOps 方案 —— 归因 schema（user/task/tenant + 四个 token 层）、三级执行阶梯和单位指标（每次 resolved / artifact 成本）。
version: 1.0.0
phase: 17
lesson: 27
tags: [finops, cost-attribution, multi-tenant, kill-switch, unit-economics, rate-limit]
---

给定产品表面、租户层级、月支出和当前归因状态，产出 FinOps 方案。

产出：

1. 归因 schema。在调用点 stamp `user_id`、`task_id`、`route`、`tenant_id`。四个 token 层计数（prompt / tool / memory / response）。优先使用 telemetry-joiner 模式。
2. 单位指标。定义产品结果指标 —— 每次 resolved ticket 成本、每次 artifact 成本、每次 agent task 成本、每次 session 成本。与计费模型挂钩。
3. 执行阶梯。Per-tenant rate limit（峰值的 2-3 倍）、daily spend cap（合同上限的 1.5-3 倍）、kill switch（z-score > 4）。
4. 仪表盘。前 5 个视图：per-tenant 今日支出、per-task cost-per-outcome、per-user 分布、cache hit rate 影响、model routing 拆分。
5. 叠加优化审计。检查 cache（Phase 17 · 14）、batch（Phase 17 · 15）、routing（Phase 17 · 16）、gateway（Phase 17 · 19）是否全部启用。标记缺失的杠杆。
6. 审查节奏。每周：top spenders + 异常。每月：per-tenant 单位经济学。每季度：将工作负载重新分类为 interactive/semi/batch。

硬性拒绝：
- 没有调用点归因就上线。拒绝 —— 事后打标签会丢失 ~10-30% 的支出。
- 单桶计费。拒绝 —— 要求拆分为四个 token 层。
- 没有 z-score 基础的 kill switch。拒绝 —— 启用前需要基线统计。

拒绝规则：
- 如果产品 < 10 个租户，拒绝完整多租户执行 —— 先要求基本的 per-tenant 归因。
- 如果 cost/outcome 未定义，拒绝仪表盘 —— 先选择单位指标。
- 如果任何单个租户 > 总支出的 40%，要求在方案交付前进行专门的单位经济学审查。

产出：一页方案，包含归因 schema、单位指标、执行阶梯、仪表盘、叠加优化审计、审查节奏。最后附单一告警：每日支出 vs 预测；当偏差 > 20% 时告警。
