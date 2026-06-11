---
name: ecosystem-blueprint
description: 给定产品需求，产出完整的 Phase 13 生态系统架构；命名原语、安全态势、遥测和打包。
version: 1.0.0
phase: 13
lesson: 23
tags: [mcp, capstone, ecosystem, architecture, a2a, otel]
---

给定产品需求（研究、摘要、自动化、任何智能体驱动的工作流），产出完整架构。

产出：

1. MCP 原语。需要哪些工具、资源、提示和任务。任何 `ui://` 应用？任何异步任务？
2. 安全态势。OAuth 2.1 作用域集、网关 RBAC 矩阵、固定哈希清单、Rule of Two 审计。
3. A2A 协作。识别任何子智能体调用。定义它们的 Agent Cards。
4. 遥测。OTel GenAI span 层次结构。导出器和后端选择。
5. 打包。AGENTS.md、SKILL.md 和部署表面（Docker Compose、K8s）。
6. 映射到 Phase 13 课程。每个设计选择追溯回哪一课。

硬拒绝项：
- 任何在单个 turn 中组合不受信任输入、敏感数据和后果性动作的架构（Rule of Two）。
- 任何跨 MCP 和 A2A 跳跃没有 trace 传播的架构。
- 任何 LLM 层上至少没有一个后备提供商的架构。

拒绝规则：
- 如果产品需求更适合直接 LLM 调用，拒绝搭建完整生态系统。
- 如果团队缺乏网关的 SRE，推荐托管网关（Cloudflare MCP Portals、Portkey）。
- 如果架构涉及支付，将 AP2 标记为具有漂移风险的 A2A 扩展并推荐单独审批。

输出：一份一页蓝图，包含原语、安全态势、A2A 跳跃、遥测计划、打包和课程映射。以一句话识别部署的单一最难运营风险结束。
