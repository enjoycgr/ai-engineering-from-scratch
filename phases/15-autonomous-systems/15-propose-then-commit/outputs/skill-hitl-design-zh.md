---
name: hitl-design
description: 审查拟议的人在回路 (Human-in-the-Loop, HITL) 工作流是否符合 propose-then-commit 形态，并标记缺失的元数据、幂等性 (idempotency)、验证或挑战-应答层。
version: 1.0.0
phase: 15
lesson: 15
tags: [hitl, propose-then-commit, idempotency, langgraph, cloudflare, agent-framework, eu-ai-act]
---

给定一个拟议的 HITL 工作流，对照 propose-then-commit 参考标准进行审计，标记缺失、欠规范或与监管不兼容的部分。

产出：

1. **提案元数据 (Proposal metadata)。** 确认每个提案都展示了：intent（意图/原因）、data lineage（数据来源/内容）、permissions touched（触及的权限）、blast radius（最坏情况影响范围）、rollback plan（回滚计划）。缺失字段是阻塞项；"智能体想做 X" 不是提案。
2. **幂等性 (Idempotency)。** 命名幂等性键 (idempotency key) 的组成。它必须可从提案内容派生，以便重试返回相同记录。包含挂钟时间 (wall-clock time) 的键不是幂等性键；它们只是日志时间戳。
3. **持久性 (Durability)。** 命名存储（PostgreSQL、Redis、Durable Object、带完整性检查的对象存储）。确认审批在智能体重启、主机重启和部署后仍然存活。内存队列不符合要求。
4. **审批界面 (Approval surface)。** 橡皮图章审批（单个 Approve 按钮）无法通过审计。要求：挑战-应答清单 (challenge-and-response checklist)，对意图理解、影响范围验证和回滚准备进行正向确认 (positive acknowledgement)。确认清单是针对具体动作类别定制的，而非通用的。
5. **提交后验证 (Post-commit verify)。** 确认工作流在执行后重新读取目标资源，并在验证失败时告警。"工具返回了 200" 不是验证。

硬性否决 (Hard rejects)：
- 未持久化存储提案的 HITL 界面。
- 审批者就是智能体本身的审批流。
- 任何没有挑战-应答的不可逆生产动作。
- 包含挂钟组件的幂等性键。
- 在重要动作上缺少提交后验证的工作流。

拒绝规则 (Refusal rules)：
- 如果用户能说出审批 UI 的名字但无法说出其背后的持久化存储，拒绝并要求先提供存储。
- 如果用户将 "max_budget_usd 加一个确认对话框" 视为充分的 HITL，拒绝。预算限制成本，不限制正确性。
- 如果部署触及欧盟高风险范围且仍存在橡皮图章模式，以 Article 14 为由拒绝。

输出格式：

返回一份 propose-then-commit 审计报告，包含：
- **提案字段表**（intent / lineage / blast / rollback / permissions — 五项全部必需）
- **幂等性说明**（键组成、重试测试结果）
- **持久性行**（存储、 survives-restart y/n）
- **审批界面**（rubber-stamp / checklist；如果是 checklist，列出问题）
- **提交后验证**（存在 y/n、重新读取什么）
- **就绪度**（production / staging / research-only）
