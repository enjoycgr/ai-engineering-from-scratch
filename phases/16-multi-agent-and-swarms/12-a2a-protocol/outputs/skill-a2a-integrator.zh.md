---
name: a2a-integrator
description: 设计两个agent (智能体)之间的A2A集成——Agent Card (智能体卡片)、task schemas (任务模式)、auth (认证)、streaming (流式传输)或polling (轮询)。
version: 1.0.0
phase: 16
lesson: 12
tags: [multi-agent, a2a, protocol, interoperability, google]
---

给定两个需要互操作的agent systems (智能体系统)，生成A2A integration plan (集成计划)：Agent Card contents (内容)、task schemas (任务模式)、auth (认证)、transport mode (传输模式)。

生成内容：

1. **Agent Card (智能体卡片)。** Name (名称)、version (版本)、skills (技能)、endpoints (端点)、supported modalities (支持的模态)（text (文本)、structured (结构化)、image (图像)、audio (音频)、video (视频)）、protocol_version (协议版本)、auth declaration (认证声明)。
2. **Task schemas per skill (每个技能的任务模式)。** Input JSON schema (输入JSON模式) + artifact JSON schema (产物JSON模式)。明确说明——clients将验证。
3. **Auth choice (认证选择)。** Bearer token (Bearer令牌)（OAuth2或opaque (不透明)）、mTLS或signed requests (签名请求)。根据threat model (威胁模型)（public internet (公共互联网)、VPC、mixed (混合)）说明理由。
4. **Transport mode (传输模式)。** Polling (轮询) vs SSE streaming (SSE流式传输) vs webhook callbacks (Webhook回调)。对于long-running (长时间运行)或progress-heavy (进度密集)的任务使用streaming (流式传输)；对于short tasks (短任务)使用polling (轮询)。
5. **Rate limits (速率限制)。** Per-client (每客户端)和per-task (每任务)限制。防止滥用。
6. **Idempotency (幂等性)。** 处理重复`POST /tasks`请求的策略（client-side task-key (客户端任务键)、server-side deduplication (服务器端去重)）。
7. **Failure handling (故障处理)。** `failed`之外的task states (任务状态)（retriable (可重试) vs fatal (致命)）、dead-letter policy (死信策略)、error artifact schema (错误产物模式)。
8. **MCP vs A2A split (MCP与A2A的分工)。** 如果远程agent内部使用MCP，记录哪些tools (工具)是exposed (暴露的) vs kept internal (内部保留的)。

Hard rejects (硬性拒绝)：

- 未声明protocol version (协议版本)的Agent Cards。
- 在use case (用例)需要structure (结构)时却是free-form text (自由格式文本)的task schemas。
- public-internet deployments (公共互联网部署)上auth=none (无认证)。

Refusal rules (拒绝规则)：

- 如果两个agent在同一进程中运行，拒绝A2A并推荐直接的Python/JS调用。A2A用于跨系统边界。
- 如果latency requirements (延迟要求)是sub-100ms round-trip (往返低于100毫秒)，拒绝A2A并推荐带shared schema (共享模式)的直接RPC。
- 如果远程agent未声明Agent Card，拒绝集成并推荐先发布一个。

Output (输出)：一页integration brief (集成简报)。以inline (内联)的Agent Card JSON收尾，以便工程团队可以直接放入`/.well-known/agent.json`。
