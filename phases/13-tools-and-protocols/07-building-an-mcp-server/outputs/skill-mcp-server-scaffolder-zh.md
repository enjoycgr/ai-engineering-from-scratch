---
name: mcp-server-scaffolder
description: 为一个领域特定的 MCP 服务器搭建脚手架，具备正确的工具/资源/提示拆分和 SDK 进阶路径。
version: 1.0.0
phase: 13
lesson: 07
tags: [mcp, server, fastmcp, scaffold]
---

给定一个域（笔记、工单、文件、数据库、任何），产出一个 MCP 服务器计划：哪些能力暴露为工具，哪些为资源，哪些为提示，加上进阶到 Python 或 TypeScript SDK 的路径。

产出：

1. 工具列表。用户明确要求执行的原子操作。包含名称、描述（Use-when 模式）、输入 schema 和注释提示。
2. 资源列表。用户想要读取的数据。URI scheme、mime 类型以及是否启用 `resources/subscribe`。
3. 提示列表。宿主应作为斜杠命令暴露的可复用模板。参数列表。
4. 能力声明。服务器在 `initialize` 中返回的确切 `capabilities` 对象。
5. 进阶说明。每部分的 FastMCP（Python）或 TypeScript SDK 等价物。命名一个 SDK 功能（例如 `lifespan`、`context`）来替换脚手架中的手写 stdlib 模式。

硬拒绝项：
- 任何仅作为工具而非资源暴露的 "数据库查询"。正确的拆分是资源用于 `/list` 和 `/read`，工具用于带参数的 `/query`。
- 任何在同一命名空间中混合用户输入工具与特权工具且不带注释的服务器。
- 任何声称 `resources/subscribe` 能力但没有持久通知机制的服务器脚手架。

拒绝规则：
- 如果域没有只读表面，拒绝搭建资源；推荐纯工具服务器。
- 如果域没有自然的斜杠命令模板，拒绝搭建提示。
- 如果用户请求认证方案，拒绝并路由到 Phase 13 · 16（OAuth 2.1）。

输出：一份一页服务器计划，包含三个原语列表、能力对象和 10 行示例 `@app.tool()` 装饰器风格进阶片段。以服务器应设置的单个最重要注释标志结束。
