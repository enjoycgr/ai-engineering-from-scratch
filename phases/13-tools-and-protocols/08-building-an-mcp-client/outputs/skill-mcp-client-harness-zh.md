---
name: mcp-client-harness
description: 给定一个 MCP 服务器的声明式列表（名称、命令、参数），搭建一个多服务器客户端，具备握手、命名空间合并和路由。
version: 1.0.0
phase: 13
lesson: 08
tags: [mcp, client, multi-server, routing, namespace]
---

给定一个要运行的 MCP 服务器配置，产出一个客户端 harness，生成每个服务器、握手每个服务器、将它们的工具列表合并到一个命名空间，并将每个调用路由到拥有服务器。

产出：

1. 服务器配置解析器。映射 `name -> {command, args, env}`。验证命令存在于路径上。
2. 生成计划。使用 subprocess.Popen 配合 stdin/stdout/stderr 管道，`bufsize=1`，文本模式。每个服务器一个后台读取线程。
3. 握手流水线。对于每个会话：发送 `initialize`，等待响应，持久化能力，发送 `notifications/initialized`。
4. 命名空间合并。选择冲突策略：`prefix-on-collision`（默认）、`reject-on-collision` 或 `silent-overwrite`（禁止）。在启动时打印合并的工具列表。
5. 路由函数。`client.call(canonical_name, arguments)` 查找拥有会话并写入 `tools/call` 消息。通过 pending-request 表中的 future 等待匹配 id 的响应。

硬拒绝项：
- 任何不在自己的进程中生成每个服务器的 harness。进程内多路复用破坏了隔离模型。
- 任何以 `silent-overwrite` 为默认冲突策略的 harness。安全风险。
- 任何阻塞主线程在 stdout 读取上的 harness。通知会停滞。

拒绝规则：
- 如果服务器的命令不受信任（不在固定 allowlist 中），拒绝生成并路由到 Phase 13 · 15 进行安全检查。
- 如果用户配置超过 10 个服务器而没有理由，警告并建议网关（Phase 13 · 17）。
- 如果被要求在此处处理 OAuth，拒绝并路由到 Phase 13 · 16。

输出：一个完整的客户端 harness Python 文件（~150 行），包含 Session、合并逻辑、路由和一个主循环，练习每个配置的服务器。以一行摘要结束，命名冲突策略和合并工具数量。
