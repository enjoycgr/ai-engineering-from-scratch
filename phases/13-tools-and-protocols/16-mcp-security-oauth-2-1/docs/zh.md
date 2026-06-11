# MCP 安全 II —— OAuth 2.1、资源指示器、增量作用域

> 远程 MCP 服务器需要授权，而非仅仅认证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）对齐。SEP-835 在 403 WWW-Authenticate 上添加了增量作用域同意和升级授权。本课将升级流程实现为状态机，以便你可以看到每一跳。

**类型：** Build
**语言：** Python（stdlib，OAuth 状态机模拟器）
**前置要求：** Phase 13 · 09（传输），Phase 13 · 15（安全 I）
**时间：** ~75 分钟

## 学习目标

- 区分资源服务器与授权服务器的职责。
- 走过 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆副手攻击。
- 实现升级授权：服务器响应 403 并带 WWW-Authenticate 请求更高作用域；客户端重新提示用户同意并重试。

## 问题

早期 MCP（2025 年前）的远程服务器使用临时 API 密钥甚至无认证交付。2025-11-25 规范用完整的 OAuth 2.1 配置文件关闭了该差距。

三个现实需求：

- **普通远程服务器。** 用户安装访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 带 PKCE 是正确的形状。
- **作用域升级。** 被授予 `notes:read` 的笔记服务器稍后可能需要 `notes:write` 进行特定动作。代替重做整个流程，升级（SEP-835）请求额外作用域。
- **防止混淆副手。** 客户端持有为服务器 A 受众作用域的 token。服务器 A 是恶意的并试图向服务器 B 呈现该 token。资源指示器（RFC 8707）将 token 固定到其预期受众。

OAuth 2.1 不是新的。新的是 MCP 的配置文件：特定要求的流程（仅授权码 + PKCE；无隐式、默认无客户端凭证），每次 token 请求上强制资源指示器，以及发布的受保护资源元数据以便客户端知道去哪里。

## 概念

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器。** 颁发 token。可能是与资源服务器相同的服务或单独的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置文件中，资源和授权服务器 CAN 是同一主机但 SHOULD 通过 URL 区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器验证验证器的哈希与存储的 challenge 匹配并颁发访问 token。
6. 客户端使用 token：`Authorization: Bearer ...` 对资源服务器的每个请求。

PKCE 防止授权码拦截攻击。资源指示器防止 token 在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少配置——客户端只需要资源 URL。

### 资源指示器（RFC 8707）

token 请求中的 `resource` 参数将 token 的预期受众固定。颁发的 token 包含 `aud: "https://notes.example.com"`。另一个接收此 token 的 MCP 服务器检查 `aud` 并拒绝它。

### 作用域模型

作用域是空格分隔的字符串。常见 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理员能力（谨慎使用）
- `profile:read` 用于身份

作用域选择应是最小权限：现在请求你需要什么，需要更多时升级。

### 升级授权（SEP-835）

用户授予 `notes:read`。他们稍后要求 Agent 删除笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 insufficient_scope 错误，向用户提示额外作用域的同意对话框，为其执行迷你 OAuth 流程，用新 token 重试请求。

### Token 受众验证

每次请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这阻止跨服务器 token 重用。

### 短寿命 token 和轮换

访问 token SHOULD 是短寿命的（默认 1 小时）。刷新 token 在每次刷新时轮换。客户端在后台处理静默刷新。

### 无 token 透传

采样服务器（Phase 13 · 11）MUST NOT 将客户端的 token 透传给其他服务。采样请求是边界。

### 防止混淆副手

Token 绑定到 `aud`。客户端绑定到 `client_id`。每次请求针对两者验证。规范明确禁止在 pre-MCP 远程工具生态系统中常见的旧 "传递 token" 模式。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 发布其元数据。授权服务器可以获取客户端的元数据文档以发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关和 OAuth

Phase 13 · 17 展示企业网关如何处理 OAuth：网关持有上游服务器的凭证，给客户端的 token 是网关颁发的，上游 token 从不离开网关。这翻转了信任模型——用户与网关认证一次；网关处理 N 个服务器授权。

## 使用它

`code/main.py` 将完整的 OAuth 2.1 升级流程模拟为状态机。它实现：

- PKCE code-verifier / challenge 生成。
- 带资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的 token 验证。
- `insufficient_scope` 上的升级。

本课无 HTTP 服务器；状态机在内存中运行以便你可以跟踪每一跳。Phase 13 · 17 的网关课将其连接到实际传输。

## 交付它

本课产出 `outputs/skill-oauth-scope-planner.md`。给定一个带工具的远程 MCP 服务器，该技能设计作用域集、固定规则和升级策略。

## 练习

1. 运行 `code/main.py`。跟踪双作用域升级流程。注意升级时哪些跳重复。

2. 添加刷新 token 轮换：每次刷新颁发新刷新 token 并使旧的无效。模拟被盗刷新 token 在轮换后使用并确认它失败。

3. 将受保护资源元数据端点实现为使用 stdlib http.server 的真实 HTTP 响应。镜像第 09 课的 /mcp 端点。

4. 为一个 GitHub MCP 服务器设计作用域层次结构：读取仓库、写入 PR、批准 PR、合并 PR、管理员。在每个级别之间使用升级。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 使用与 RFC 示例不同的一个字段。（提示：它涉及 `scopes_supported`。）

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| OAuth 2.1 | "现代 OAuth" | 强制 PKCE 并禁止隐式流程的合并 RFC |
| PKCE | "持有证明" | 击败授权码拦截的 code verifier + challenge |
| 资源指示器 | "Token 受众" | 将 token 固定到一个服务器的 RFC 8707 `resource` 参数 |
| 受保护资源元数据 | "发现文档" | RFC 9728 `.well-known/oauth-protected-resource` |
| 升级授权 | "增量同意" | 按需添加作用域的 SEP-835 流程 |
| `insufficient_scope` | "带 WWW-Authenticate 的 403" | 服务器信号重新同意更大作用域 |
| 混淆副手 | "跨服务 token 重用" | 可信持有者不适当地转发 token 的攻击 |
| 短寿命 token | "访问 token TTL" | 快速过期的 Bearer；刷新 token 续期 |
| 作用域层次 | "最小权限栈" | 带级别间升级的渐进作用域集 |
| 客户端 ID 元数据 | "客户端发现文档" | 客户端发布其自己 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) —— 规范 MCP OAuth 配置文件
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) —— 2025-11-25 变更的走过
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) —— 受众固定 RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) —— 发现文档 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) —— 实际升级流程走过
