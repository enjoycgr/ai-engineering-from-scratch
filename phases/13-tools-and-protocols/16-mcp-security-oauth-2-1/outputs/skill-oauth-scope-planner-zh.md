---
name: oauth-scope-planner
description: 给定一个带工具的远程 MCP 服务器，设计作用域集、固定规则和升级策略。
version: 1.0.0
phase: 13
lesson: 16
tags: [mcp, oauth, pkce, scope, sep-835]
---

给定一个远程 MCP 服务器的工具列表，设计 OAuth 2.1 作用域模型。

产出：

1. 作用域映射。每个工具映射到最小作用域（`notes:read`、`github:write` 等）。
2. 作用域层次。从最窄到最宽排序作用域，定义升级路径。
3. 资源指示器。每个作用域固定到 `resource=https://server.example.com`。
4. 升级策略。文档化哪些操作触发 `insufficient_scope` 以及升级流程的样子。
5. Token 生命周期。访问 token TTL（推荐 1 小时）、刷新 token 轮换和会话绑定。

硬拒绝项：
- 任何 `admin:*` 作用域没有单独的升级门。管理员能力是特权升级。
- 任何没有受众验证的 token。必须检查 `aud == resource_url`。
- 任何隐式流程或客户端凭证作为默认流程。MCP 配置文件仅要求授权码 + PKCE。

拒绝规则：
- 如果服务器不远程（仅本地 stdio），拒绝 OAuth 规划；本地服务器使用父进程信任边界。
- 如果用户要求 "一个 token 统治一切"，拒绝并坚持最小权限作用域。
- 如果服务器尚未实现受保护资源元数据（RFC 9728），拒绝直到交付。

输出：每个工具的作用域表、作用域层次图、示例 `WWW-Authenticate` 升级响应和 token 生命周期策略。以推荐的默认 TTL 结束。
