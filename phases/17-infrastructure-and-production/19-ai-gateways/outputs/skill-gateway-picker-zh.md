---
name: gateway-picker
description: 根据规模、延迟预算、合规性、运维姿态和定价容忍度，选择 AI gateway（LiteLLM、Portkey、Kong AI、Cloudflare/Vercel）。
version: 1.0.0
phase: 17
lesson: 19
tags: [ai-gateway, litellm, portkey, kong, cloudflare, vercel, bifrost, fallback, rate-limit, guardrails]
---

给定 RPS（当前和 12 个月预测）、延迟预算、合规（是否需要自托管？）、guardrails 需求（PII redaction、jailbreak detection、audit）和定价容忍度，产出 gateway 推荐。

产出：

1. 主 gateway。命名工具。用 RPS 上限、overhead 和功能匹配度说明理由。
2. Fallback 链。三个按顺序排列的提供商；OpenAI → Anthropic → 自托管是经典方案。计算预期可用性。
3. Rate-limit 策略。>500 RPS 推荐 sliding-window；否则 token-bucket 可接受。按租户分层。
4. Guardrails。如需 PII/jailbreak 选 Portkey；如需规模 + guardrails 选 Kong；如仅 dev tier 选 LiteLLM。
5. Observability 交接。指向 Phase 17 · 13 的选择；确认 OTel GenAI 约定可流通。
6. 迁移。如从应用层集成迁移，分阶段发布（gateway 上 1% canary，成功后扩展）。

硬拒绝：
- LiteLLM 在 >2000 RPS。拒绝——Kong 基准显示级联故障；先迁移。
- Portkey 在 TTFT P99 < 100 ms SLA。拒绝——30 ms overhead 吃掉太多预算。
- Cloudflare AI Gateway 用于受监管的内部部署客户。拒绝——仅托管；无自托管。

拒绝规则：
- 如果规模不确定性大（当前 100 RPS，6 个月内计划 2K+），要求在承诺 LiteLLM 前先出迁移计划。
- 如果合规要求 SOC 2 Type II 且所选 gateway 仅为无托管 SLA 的 OSS，要求客户自行完成 SOC 2 鉴证。
- 如果团队无 Kubernetes 却选择 Kong 自托管，拒绝——推荐托管 Kong 或 Portkey 托管。

输出：一页决策文档，含 gateway、fallback 链、rate-limit 策略、guardrail 姿态、observability 流、迁移计划。结尾附单一指标：过去一小时 gateway latency P99；违规时告警。
