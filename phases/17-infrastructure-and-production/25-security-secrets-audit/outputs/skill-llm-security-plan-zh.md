---
name: llm-security-plan
description: 产出 LLM 安全计划，涵盖 secrets vault、使用一致性分词的 PII 脱敏、网络出口白名单、审计日志保留和零信任态势。
version: 1.0.0
phase: 17
lesson: 25
tags: [security, vault, hashicorp, aws-secrets-manager, pii, presidio, egress, audit-log, zero-trust, ci-cd-supply-chain]
---

给定法规范围（SOC 2、HIPAA、GDPR）、当前凭证状态和网络/出口态势，产出安全计划。

产出：

1. Vault 迁移。选择 vault（HashiCorp、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager）。网关模式：应用 → 网关 → 运行时从 vault 拉取。弃用硬编码 env 和配置文件凭证。
2. Secret 扫描。每次提交启用 TruffleHog / GitGuardian / Gitleaks。检测到时阻断 PR。
3. 轮换策略。≤ 90 天。尽可能自动化。CI/CD 凭证专用轮换（更短 —— 建议 30 天）。
4. PII 脱敏。实体识别（Presidio + 正则）。Consistent tokenization（相同值 → 相同占位符）以保留语义。
5. 出口白名单。将 LLM 提供商域名、向量数据库、vault 端点列入白名单。DNS 白名单解析器。
6. 审计日志。仅追加、不可变。必填字段：用户、租户、提示/响应哈希、token、成本、护栏触发。按框架保留（SOC 2 为 1 年 / HIPAA 为 6 年）。
7. CI/CD 卫生。OIDC 身份联邦（无静态云密钥）。 narrowly 限定 CI/CD 凭证范围。引用 2026 年 Vercel 供应链事件作为动机。

硬性拒绝：
- 配置文件中的静态密钥。拒绝。
- 在审计日志中存储原始提示。拒绝 —— 仅保留哈希，除非法规框架明确要求。
- 允许出口到 `*` 或"互联网"。拒绝 —— 使用白名单。

拒绝规则：
- 如果客户无法接受任何 vault（气隙要求），拒绝正常计划并设计基于文件的带轮换回退方案。明确说明其安全性较低。
- 如果因"延迟"原因拒绝 PII 脱敏，拒绝 —— 延迟通常 <20 ms，法规风险远大于此。
- 如果要求 vault root token 的轮换 >90 天，拒绝 —— 它会变成泄露向量。

产出：一页计划，包含 vault、扫描、轮换、脱敏、出口、审计日志、CI/CD 态势。最后给出单一指标：每月 secret 扫描命中数；目标为零。
