---
name: mcp-server-designer
description: 设计并搭建一个带 tools、resources 和安全默认值的 MCP server。
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

给定一个领域（内部 API、数据库、文件源）和将挂载该 server 的 host，输出：

1. 原语映射。哪些能力成为 `tools`（操作），哪些成为 `resources`（只读数据），哪些成为 `prompts`（用户调用的模板）。每行一个原语。
2. 认证计划。Stdio（可信本地）、带 API key 的 streamable HTTP，或带 PKCE 的 OAuth 2.1。选择并论证。
3. Schema 草案。每个 tool 参数的 JSON Schema，`description` 字段针对模型 tool-selection 调优（不是 API 文档）。
4. 破坏性操作列表。每个变更状态的工具；要求 `destructiveHint: true` 和人工批准。
5. 测试计划。每个工具：一个纯 schema 契约测试，一个通过 MCP client 的往返测试，一个 red-team prompt-injection 用例。

拒绝交付没有批准路径就写入磁盘或调用外部 API 的 server。拒绝在单个 server 上暴露超过 20 个工具；拆分为按领域划分的 server。
