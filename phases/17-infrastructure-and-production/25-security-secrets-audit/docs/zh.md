# Security — Secrets, API Key Rotation, Audit Logs, Guardrails

> 通过集中式 vault（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除 secret 蔓延。切勿将凭证存储在配置文件、VCS 中的 env 文件或电子表格中。使用 IAM 角色替代静态密钥；CI/CD 使用 OIDC。AI-gateway（AI 网关）模式是 2026 年的解决方案：应用 → 网关 → 模型提供商，网关从 vault 运行时拉取凭证。在 vault 中轮换，所有应用在几分钟内自动获取新密钥 —— 无需重新部署，无需在 Slack 上问"谁有新密钥"。轮换策略 ≤90 天；每次提交都用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短期令牌、设备态势。PII scrubbing（PII 脱敏）使用实体识别在转发前屏蔽 PHI/PII；consistent tokenization（一致性分词）（Mesh 方法）将敏感值映射到稳定的占位符，使 LLM 保留代码/关系语义。网络出口：LLM 服务位于专用 VPC/VNet 子网中，仅将 `api.openai.com`、`api.anthropic.com` 等列入白名单；阻止所有其他出站流量。2026 年的事件驱动因素：Vercel 供应链攻击，通过泄露的 CI/CD 凭证窃取了数千个客户部署的环境变量。

**类型：** Learn
**语言：** Python（stdlib，玩具级 PII 脱敏器 + 审计日志写入器）
**前置条件：** Phase 17 · 19（AI 网关）、Phase 17 · 13（可观测性）
**时间：** ~60 分钟

## 学习目标

- 列举四种 secret 管理反模式（VCS 中的配置文件、硬编码 env、电子表格、静态密钥）及其替代方案。
- 解释 AI-gateway-pulls-from-vault 模式作为 2026 年生产标准。
- 实现带 consistent tokenization（相同值 → 相同占位符）的 PII 脱敏器，使语义得以保留。
- 说出 2026 年 Vercel 供应链事件及其关于 CI/CD 凭证卫生的教训。

## 问题

一名实习生提交了包含 API 密钥的 `.env`。他们迅速删除了。密钥已在 git 历史记录中 —— GitGuardian 扫描捕获了它，你的轮换流程是"在 Slack 上通知团队，更新 40 个配置文件，重新部署所有服务"。8 小时后，一半服务已上线，一半还在等待部署窗口。

另外，用户提示中包含"My SSN is 123-45-6789"。提示发送到了 OpenAI。你有 BAA，但内部策略要求在转发前屏蔽 PII。你没有做到。

另外，你的 EKS 集群中的 LLM pod 可以访问任何互联网主机。有人通过 DNS 查询将数据泄露到攻击者控制的域名。没有任何东西阻止它。

LLM 服务的安全必须同时解决这三个向量。Vault 支持的凭证。PII 脱敏。网络出口过滤。审计日志。

## 概念

### 集中式 vault + IAM 角色拉取

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。单一事实来源。

**IAM 角色**：应用/网关通过其 IAM 身份而非静态密钥进行认证。Vault 在令牌生命周期内返回 secret。

**AI-gateway 模式**：网关在请求时从 vault 拉取 `OPENAI_API_KEY`。在 vault 中轮换；下一个请求获取新密钥。无需重新部署。

### 轮换策略 ≤ 90 天

所有 API 密钥、vault root token、CI/CD 凭证。尽可能自动化。手动轮换需记录和跟踪。

### Secret 扫描

- **TruffleHog** —— 基于正则 + 熵的提交扫描。
- **GitGuardian** —— 商业产品，高精度。
- **Gitleaks** —— 开源，在 CI 中运行。

每次提交都运行。检测到新 secret 时阻断 PR。

### 零信任态势

- 所有账户强制 MFA。
- 通过 SAML/OIDC 实现 SSO。
- 细粒度访问使用 RBAC（基于角色）或 ABAC（基于属性）。
- 短期令牌（小时级，而非天级）。
- 设备态势 —— 仅允许带有磁盘加密的企业设备。

### PII / PHI 脱敏

在提示离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业产品）。
2. 屏蔽匹配实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. Consistent tokenization（Mesh 方法）：相同值映射到相同占位符，使 LLM 保留关系。
4. 可选的 LLM 响应反向映射。

静态正则过滤器捕获基本模式；NER 捕获更多。两者都用。

### 输入 + 输出护栏

输入：阻止已知的越狱、禁止主题；按用户限流。

输出：正则脱敏以捕获泄露的 secret（API 密钥模式、拒绝上下文中的邮箱模式）、策略违规分类器。

### 网络出口白名单

LLM 服务位于专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、vault 端点。
- 其他所有流量：丢弃。
- DNS 仅通过白名单解析器（避免 DNS 隧道泄露）。

### 审计日志

每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户 / 租户。
- 提示哈希（出于隐私不保留原始提示）。
- 模型 + 版本。
- Token 计数。
- 成本。
- 响应哈希。
- 任何护栏触发。

按法规要求保留（SOC 2 为 1 年，HIPAA 为 6 年）。

### 2026 年 Vercel 事件

供应链攻击：泄露的 CI/CD 凭证窃取了数千个客户部署的环境变量。教训：CI/CD 凭证等同于生产凭证。存储在 vault 中。范围收窄。积极轮换。

### 应该记住的数字

- 轮换策略：≤ 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭证泄露 → 数千个客户环境变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

## 使用

`code/main.py` 实现了带 consistent tokenization 的玩具级 PII 脱敏器和追加式审计日志。

## 交付

本节课产出 `outputs/skill-llm-security-plan.md`。给定法规范围和当前状态，规划 vault 迁移、脱敏器、出口、审计日志。

## 练习

1. 运行 `code/main.py`。发送两条引用相同 SSN 的提示。确认两者获得相同的占位符。
2. 为调用 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署设计网络出口策略。
3. 你在 git 历史记录中发现一个密钥（2 年前）。正确响应是什么 —— 轮换密钥、清理历史记录，还是两者都要？论证。
4. 你的审计日志每天增长 10 GB。设计保留层级（热存储 30 天、温存储 12 个月、冷存储 6 年）。
5. 论证 reverse-tokenization（将真实值替换回 LLM 响应）是否值得其复杂性，还是保持占位符可见更好。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Vault | "secrets 存储" | 集中式凭证管理服务 |
| IAM role | "基于身份认证" | 应用承担的角色；返回短期凭证 |
| OIDC for CI/CD | "云颁发令牌" | CI 中无静态密钥 —— 通过 OIDC 获取身份 |
| TruffleHog / GitGuardian / Gitleaks | "secret 扫描器" | 提交时 secret 检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII scrubbing | "数据脱敏" | 移除或分词敏感实体 |
| Consistent tokenization | "稳定占位符" | 相同值 → 每次相同令牌 |
| Mesh approach | "Mesh 分词" | 保留语义的分词模式 |
| Egress whitelist | "出站白名单" | 仅允许访问许可的域名 |
| Audit log | "不可变历史" | 用于合规的追加式记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
