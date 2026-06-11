# MCP 生产认证 —— 注册、JWKS 刷新、受众固定 Token

> 第 16 课在内存中建立了 OAuth 2.1 状态机。到 2026 年，每个你交付给真实组织的 MCP 服务器都位于生产认证之后：扩展到无界客户端群的客户端注册（首先是客户端 ID 元数据文档，动态客户端注册作为向后兼容的后备）、授权服务器元数据发现（RFC 8414 或 OpenID Connect 发现）、不会破坏凌晨 3 点 token 验证的 JWKS 缓存刷新，以及拒绝跨资源重播的受众固定 token。本课用三个角色建模完整表面——授权服务器、资源服务器（MCP 服务器）和客户端——以便你可以跟踪从发现到验证工具调用的每一跳。
>
> **规范说明（2025-11-25）：** 2025 年 11 月的 MCP 授权规范将动态客户端注册从 `SHOULD` 降级为 `MAY`，并使 **客户端 ID 元数据文档（CIMD）** 成为推荐的默认注册机制。本课按规范的优先顺序教授两者，代码保留 DCR 用于走过，因为它完全自包含在一个进程中。

**类型：** Build
**语言：** Python（stdlib）
**前置要求：** Phase 13 · 16（OAuth 2.1 状态机），Phase 13 · 17（网关）
**时间：** ~90 分钟

## 学习目标

- 通过 RFC 8414 元数据发现并验证契约来发现授权服务器。
- 实现 RFC 7591 动态客户端注册，使 MCP 客户端无需管理员干预即可注册。
- 按计划缓存和刷新 JWKS 密钥，使签名验证在密钥轮换中存活。
- 使用 RFC 8707 资源指示器将 token 固定到单个 MCP 资源，并拒绝混淆副手重用。
- 干净地分离三个角色——授权服务器、资源服务器、客户端——以便每个只强制执行属于它的检查。
- 阅读 IdP 能力矩阵，并在 IdP 无法满足 MCP 的认证配置文件时拒绝部署。

## 问题

第 16 课模拟器在内存中运行 OAuth 2.1。生产有三个操作差距，纯内存模拟器看不到。

第一个差距是注册。一个真实组织运行数百个 MCP 服务器和数千个 MCP 客户端。运营商不会手动将每个 Cursor 用户注册为 OAuth 客户端。2025-11-25 规范为客户端提供了解决此问题的优先顺序：如果你有预注册的 `client_id` 就使用它，否则使用 **客户端 ID 元数据文档**（客户端用它控制的 HTTPS URL 作为 `client_id`，授权服务器在 OAuth 流程中 *拉取* 元数据），否则回退到 **RFC 7591 动态客户端注册**（客户端 *推送* `POST /register` 并在现场接收 `client_id`），否则提示用户。CIMD 是推荐默认，因为它完全消除了每服务器注册，同时保持基于 DNS 的信任模型；DCR 为向后兼容性保留。两者都从授权服务器的元数据中发现入口点：CIMD 的 `client_id_metadata_document_supported`，DCR 的 `registration_endpoint`。

第二个差距是密钥轮换。JWT 验证依赖授权服务器发布的签名密钥，以 JSON Web Key Set（JWKS）形式发布。授权服务器按计划轮换它们（通常每小时，有时在事件响应中更快）。在启动时获取一次 JWKS 的 MCP 服务器在轮换窗口后验证正常——然后每次请求都失败直到重启。生产将 JWKS 连接为带刷新作业的缓存值，在旧密钥到期前覆盖缓存，加上缓存未命中时的回退获取，以应对 token 在计划刷新前由全新的密钥签名到达的情况。

第三个差距是受众绑定。第 16 课介绍了 RFC 8707 资源指示器。在生产中，该指示器变为每次请求的硬性声明检查。MCP 服务器将 `token.aud` 与其自己的规范资源 URL 比较，并在不匹配时拒绝。这是上游 MCP 服务器（或持有用于一个服务器的 token 的恶意客户端）针对同一信任网格中另一个服务器重播该 token 的唯一防御。

本课将每个差距映射到表面的具体部分。元数据文档是 HTTP 端点。JWKS 缓存刷新是计划作业加键值缓存。JWT 验证是资源服务器在分发任何工具前运行的例程。保持三个角色分离，每个只强制执行属于它的检查：授权服务器颁发和轮换密钥，资源服务器缓存和验证，客户端发现并注册。

## 概念

### RFC 8414 —— OAuth 授权服务器元数据

`/.well-known/oauth-authorization-server` 处的文档描述了客户端需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

给定 MCP 资源 URL 的客户端链式发现：RFC 9728 的 `oauth-protected-resource`（资源服务器的文档）命名颁发者，然后 `oauth-authorization-server`（本 RFC）命名每个端点。客户端从不硬编码授权 URL。

你在信任 IdP 进行 MCP 之前验证的契约：

- `code_challenge_methods_supported` 包含 `S256`（PKCE 每 RFC 7636）。规范明确：如果此字段 **缺失**，授权服务器不支持 PKCE，客户端 **必须** 拒绝继续。
- `grant_types_supported` 包含 `authorization_code` 并拒绝 `password` 和 `implicit`。
- 至少一个注册路径被广告：`client_id_metadata_document_supported: true`（CIMD，优先）**或** `registration_endpoint`（RFC 7591 DCR，后备）。任一满足契约；你不再硬要求 DCR。
- `response_types_supported` 对 OAuth 2.1 恰好是 `["code"]`。

如果 `S256` 缺失，MCP 服务器拒绝针对此 IdP 部署——PKCE 没有降级模式。如果 *两个* 注册路径都未广告且你没有预注册的 `client_id`，你也无法注册；部署清单错误，而非代码错误。

### RFC 9728（回顾）—— 受保护资源元数据

第 16 课涵盖了 RFC 9728。生产中的增量：此文档是客户端查找 *此* MCP 服务器信任的授权服务器的唯一位置。单个 MCP 服务器可能接受来自多个 IdP 的 token（一个用于员工，一个用于合作伙伴）。RFC 9728 声明该集合；RFC 8414 记录每个 IdP 支持什么。

### 客户端 ID 元数据文档（推荐默认）

CIMD 将注册从 *推送* 反转为 *拉取*。代替要求授权服务器铸造 `client_id`，客户端用它控制的 HTTPS URL **作为** 其 `client_id`。该 URL 解析为 JSON 元数据文档；授权服务器在 OAuth 流程中按需获取它。信任植根于 DNS：如果服务器运营商信任 `app.example.com`，它就信任从 `https://app.example.com/client.json` 服务的客户端。无需注册往返，无 `client_id` 命名空间耗尽，无每服务器状态需要同步。

客户端托管的元数据文档：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文档中的 `client_id` 值 **必须** 等于它自己服务的 URL（授权服务器验证这一点；不匹配被拒绝）。授权服务器以 `client_id_metadata_document_supported: true` 在其 RFC 8414 元数据中广告支持。

规范对此直截了当地说明两个安全事实：

- **SSRF。** 授权服务器获取攻击者提供的 URL。它必须防御服务器端请求伪造（不获取内部/管理端点）。
- **本地主机冒充。** CIMD 单独无法阻止本地攻击者声领合法客户端的元数据 URL 并绑定任何 `localhost` 重定向。授权服务器 **必须** 在同意期间清晰显示重定向 URI 主机名，并 **应该** 对仅 `localhost` 重定向发出警告。

因为 CIMD 不需要服务器端状态，所以不需要像 DCR 那样站立注册器。客户端侧是只读的：从你的静态 HTTPS 端点服务元数据文档，让授权服务器拉取它。

### RFC 7591 —— 动态客户端注册（后备 / 向后兼容）

DCR 现在是 `MAY`，为 2025-11-25 前部署和不支持 CIMD 的 IdP 保留向后兼容性。没有它（且没有 CIMD 或预注册），每个 MCP 客户端（Cursor、Claude Desktop、自定义 Agent）都需要与 IdP 管理员进行带外交换。使用 DCR，客户端推送：

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

服务器以 `client_id` 和 `registration_access_token` 响应，供以后更新：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "**********",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` 是运行在用户设备上的 MCP 客户端的正确默认。它们只获得 `client_id`——无 `client_secret` 可渗出。PKCE 补偿了公共客户端所需的持有证明。

三个生产陷阱：

- 注册端点必须按源 IP 限速。没有它，敌对行为者脚本化数百万假注册并耗尽 `client_id` 命名空间。在注册器处理请求前运行限速检查。
- `software_statement`（为客户端担保的签名 JWT）被一些企业 IdP 要求。本课的模拟跳过它；生产连接验证步骤，拒绝来自非 localhost 重定向 URI 的未签名注册。
- `registration_access_token` 必须作为哈希而非明文存储。盗窃此 token 意味着攻击者可以重写客户端的重定向 URI。

### RFC 8707（回顾）—— 资源指示器

第 16 课建立了形状。生产规则：每个 token 请求包含 `resource=<canonical-mcp-url>`，MCP 服务器在每次调用时验证 `token.aud` 与其自己的资源 URL 匹配。规范 URI 是服务器的 *最具体* 标识符：它使用小写 scheme 和主机，无片段，且习惯上无尾部斜杠。当需要识别单个 MCP 服务器时，保留路径组件。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的规范 URI。每个服务器选一个并将其精确固定为 `aud`。

### RFC 7636（回顾）—— PKCE

PKCE 在 OAuth 2.1 中是强制性的。本课的授权码流程始终携带 `code_challenge` 和 `code_verifier`。服务器拒绝任何不带验证器或验证器哈希与存储的 challenge 不匹配的 token 请求。

### MCP 规范 2025-11-25 认证配置文件

MCP 规范（2025-11-25）精确规定了 MCP 服务器授权层必须做什么：

- 实现 RFC 9728 受保护资源元数据，并通过 `WWW-Authenticate: Bearer resource_metadata="..."` 头部在 401 上提供其位置 **或** 通过 well-known URI `/.well-known/oauth-protected-resource`（SEP-985 使头部可选，well-known 回退）。元数据 `authorization_servers` 字段 **必须** 命名至少一个服务器。
- 在 **每次** 请求上仅通过 `Authorization: Bearer ...` 接受 token——绝不在查询字符串中，绝不在仅在会话开始时验证。
- 每次请求验证 `aud`、`iss`、`exp` 和所需作用域。服务器 **必须** 验证 token 专门颁发给它（受众）；缺失或不匹配的 `aud` 被拒绝，绝不视为通配符。
- 在 401/403 上，返回携带 `error=...` 的 `WWW-Authenticate: Bearer`、用于元数据文档 URL（*而非* 裸资源）的 `resource_metadata="<PRM-URL>"` 参数，以及在 `insufficient_scope`（403）上的 `scope="..."`。注意：参数是 `resource_metadata`，一个发现指针——challenge 中没有 `resource` 参数。
- 授权服务器发现接受 **任一** RFC 8414 OAuth 元数据 **或** OpenID Connect Discovery 1.0；客户端必须按优先顺序尝试两个 well-known 后缀。
- 客户端（而非服务器）防御 **混合攻击**：它在重定向前记录预期的 `issuer` 并在兑换前验证授权响应的 `iss` 参数（RFC 9207）。单独的 PKCE 无法阻止混合，因为客户端将 `code_verifier` 交给它被引导到的任何 token 端点。

OAuth 2.1 草案是底层；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是表面；MCP 规范是配置文件。

### IdP 能力矩阵

不是每个 IdP 都支持完整的 MCP 配置文件。下表记录了 2025-11-25 规范的事实能力说明。它是 *部署门控*，而非推荐。

CIMD 在 2025-11-25 规范中交付，底层 OAuth 草案仅在 2025 年 10 月被采纳，所以供应商支持仍在到来——将下面的 "CIMD" 视为 "现状，在你的租户中验证"，而非永久声明。

| IdP 类别 | AS 元数据（8414/OIDC） | CIMD | RFC 7591 DCR | RFC 8707 资源 | RFC 7636 S256 PKCE | 说明 |
|---|---|---|---|---|---|---|
| 自托管（Keycloak） | 是 | 新兴 | 是 | 是（自 24.x） | 是 | 本课 MCP 配置文件的参考 IdP；完整的 DCR 路径端到端，CIMD 跟踪新规范。 |
| 企业 SSO（Microsoft Entra ID） | 是 | 新兴 | 是（高级层） | 是 | 是 | DCR 可用性因租户层而异；在目标租户中部署前验证。 |
| 企业 SSO（Okta） | 是 | 新兴 | 是（Okta CIC / Auth0） | 是 | 是 | DCR 在 Auth0（现为 Okta CIC）上可用；经典 Okta 组织需要管理员预注册。 |
| 社交登录 IdP（通用） | 各异 | 否 | 很少 | 很少 | 是 | 大多数社交 IdP 将客户端视为静态合作伙伴；无自助注册。仅用作身份源，在你自己的 MCP 感知授权服务器之上分层。 |
| 自定义 / 自制 | 取决于 | 取决于 | 取决于 | 取决于 | 取决于 | 如果你交付自己的，交付完整配置文件并优先使用 CIMD。跳过 PKCE 或受众绑定会破坏 MCP 认证契约。 |

部署清单的拒绝规则：如果所选 IdP 未在 `code_challenge_methods_supported` 中列出 `S256`，MCP 服务器拒绝启动——PKCE 没有降级模式。注册是更软的门槛：你需要 *一个* 工作路径（预注册的 `client_id`、 `client_id_metadata_document_supported: true` 或 `registration_endpoint`）。单独的 DCR 缺失不再是拒绝触发器，因为 CIMD 或预注册可以覆盖它。

### JWKS 刷新模式（在 AS 轮换，在资源服务器刷新）

保持两个动词分离，因为混淆它们是真实的生产 bug：

- **轮换** 是 *授权服务器* 做的：铸造新签名密钥，在 JWKS 中发布它，稍后退役旧密钥。资源服务器不参与此过程且不能做——它不持有 IdP 的私钥。
- **刷新** 是 *资源服务器* 做的：重新 `GET` 已发布的 JWKS 到其缓存中。这是资源服务器唯一做的 JWKS 动作。

生产失败模式是陈旧缓存。用一个计划刷新作业加一个键值缓存解决它。资源服务器运行一个作业（cron、计时器，你的运行时提供的任何东西），以固定间隔获取 `<issuer>/.well-known/jwks.json` 并覆盖 `cache[issuer] = {keys, fetched_at}`。验证器从该缓存读取。缓存中缺失 `kid` 的 token 触发 **一次** 同步刷新作为回退，然后重新检查。这同时处理两种情况：计划刷新，以及 token 由比缓存更新的密钥签名且在下次计划刷新前到达的密钥重叠窗口。

回退 **必须是重新获取，绝不能是轮换**。如果你将缓存未命中路径连接到轮换并铸造，两件事会破坏：(1) 铸造新密钥仍然产生不匹配 token 的 `kid`，所以查找仍然失败；(2) 用随机 `kid` 值喷射 token 的攻击者强制无界系列密钥创建——一种自我导致的 DoS。重新获取是幂等的，所以错误的 `kid` 最多花费一次浪费的获取。

缓存形状：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

两个密钥同时存在是稳态。授权服务器通过引入下一个密钥（`k_2026_04`）在退役前一个（`k_2026_03`）之前轮换，因此在旧密钥下颁发的 token 在到期前保持有效。缓存持有联合；验证器按 `kid` 选择。

### 验证例程

MCP 服务器在分发任何工具前运行验证。`code/main.py` 使用的形状：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解码 JWT，从 JWKS 缓存解析签名密钥（在缺失时刷新一次），验证签名，然后对照允许列表检查 `iss`、对照此服务器的规范资源检查 `aud`、检查 `exp` 和所需作用域——在第一次失败时返回 `WWW-Authenticate` 质询。在资源服务器上保持为单一例程意味着每个入口点（每次工具调用、每次传输）都经过相同检查；没有任何路径能在未先验证的情况下到达工具。

### 受众重播走过（访问 token 权限限制）

服务器 A（`notes.example.com`）和服务器 B（`tasks.example.com`）都针对同一授权服务器注册。服务器 A 被入侵。攻击者获取用户的笔记 token 并将其针对服务器 B 重播。

服务器 B 的验证器：

1. 解码 JWT，按 `kid` 获取 JWKS，验证签名。
2. 对照其受保护资源元数据的 `authorization_servers` 检查 `iss`。（通过——同一 IdP。）
3. 检查 `aud == "https://tasks.example.com"`。（失败——token 的 `aud` 是 `https://notes.example.com`。）
4. 返回 401 并带 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

受众声明是协议层对此攻击的唯一防御。为性能跳过它是最常见的生产错误；验证器必须在每次请求上运行，而不仅在会话开始时。规范称之为 **访问 token 权限限制**：MCP 服务器 `必须` 拒绝任何未在受众中命名它的 token。

> **命名说明。** 规范保留术语 *混淆副手* 用于相关但不同的问题：充当 OAuth **代理** 到第三方 API 的 MCP 服务器，使用静态客户端 ID，在不获得每客户端用户同意的情况下转发 token。上面的受众绑定修复了重播；混淆副手修复是每客户端同意 **加** 绝不将入站 token 透传给上游 API（MCP 服务器 `必须` 获得自己的单独上游 token）。

### 混合攻击（客户端侧防御，服务器无法提供）

客户端在其生命周期中与许多授权服务器对话。恶意 AS 可以尝试让客户端在攻击者的 token 端点兑换诚实 AS 的授权码。受众绑定在这里无帮助——攻击发生在任何 token 存在之前。防御存在于客户端（RFC 9207）：

1. 重定向前，客户端从验证的 AS 元数据记录预期的 `issuer`。
2. 在授权响应上，客户端将返回的 `iss` 参数与该记录的颁发者进行对照（简单字符串比较，无规范化）后再将 code 发送到任何地方。
3. 不匹配（或当 AS 广告了 `authorization_response_iss_parameter_supported` 时 `iss` 缺失）→ 拒绝，并且甚至不显示 `error` 字段。

单独的 PKCE 无法阻止混合，因为客户端将 `code_verifier` 交给它被引导到的任何 token 端点。这就是为什么规范要求每个请求记录颁发者，与 PKCE 验证器和 `state` 一起。

### 失败模式

- **陈旧 JWKS。** 验证器在 AS 轮换密钥后拒绝有效 token。修复是上面的 cron-refresh + cache-miss-refetch 模式。永远不要无刷新作业地缓存 JWKS。
- **轮换作为回退。** 将缓存未命中路径连接到轮换并铸造而非重新获取是真实的 bug：它仍然不产生缺失的 `kid`，并且它将攻击者控制的 `kid` 值转变为密钥创建 DoS。回退必须是幂等的 `refresh-jwks`。
- **缺失 `aud` 声明。** 一些 IdP 除非 token 请求中存在 `resource`，否则默认省略 `aud`。验证器必须拒绝 `aud` 缺失的 token，不将缺失视为通配符。
- **通过缺失 `iss` 检查的混合。** 不重定向前验证 RFC 9207 `iss` 授权响应参数与它所记录的颁发者的客户端可以被引导到在攻击者的 token 端点兑换诚实 AS 的 code。这是客户端侧失败；资源服务器无法补偿。
- **作用域升级竞态。** 同一用户的两个并发升级流程都可以成功并产生两个不同作用域的访问 token。验证器必须使用请求上呈现的 token，而非查找 "用户当前作用域"——这会创建 TOCTOU 窗口。
- **注册 token 盗窃。** 泄漏的 `registration_access_token` 让攻击者重写重定向 URI。静态时哈希它们；要求客户端在每次更新时呈现明文；在怀疑时轮换。
- **`iss` 未固定。** 接受任何 `iss` 的验证器让攻击者站立自己的授权服务器，为目标受众注册客户端，并颁发 token。受保护资源元数据的 `authorization_servers` 列表是允许列表；强制执行它。

## 使用它

`code/main.py` 用 stdlib Python 和三个角色——`AuthorizationServer`、`ResourceServer` 和 `Client`——走过完整生产流程。流程：

1. 授权服务器在 `/.well-known/oauth-authorization-server` 发布 RFC 8414 元数据。
2. MCP 客户端调用元数据端点并检查其注册选项（CIMD 的 `client_id_metadata_document_supported`，DCR 的 `registration_endpoint`）和 `S256` PKCE 支持。
3. 走过采用 DCR 回退路径：客户端 POST 到 `/register`（RFC 7591）并接收 `client_id`。（CIMD 客户端会代替呈现自己的 HTTPS `client_id` URL 并跳过此步骤。）
4. MCP 客户端运行带资源指示器（RFC 8707）的 PKCE 保护授权码流程（RFC 7636）。
5. MCP 客户端以 `Authorization: Bearer ...` 调用 MCP 服务器上的工具。
6. MCP 服务器运行 `validate`，从 JWKS 缓存解析签名密钥。
7. IdP 轮换密钥；计划刷新将 JWKS 重新拉入缓存。
8. 下一次调用针对刷新的密钥验证而无需重启，且前一个 token 在重叠窗口期间仍然验证。
9. 针对不同 MCP 资源的受众重播尝试获得 401 并带 `audience mismatch` 和 `resource_metadata` 指针。

本课的 JWT 使用共享密钥的 HS256（以便课程仅在 stdlib 上运行）。生产使用上面的 JWKS 模式的 RS256 或 EdDSA；验证逻辑除此之外相同。因为 IdP 和资源服务器在一个进程中，`refresh_jwks` 直接读取授权服务器的密钥列表；在电线上它是到 `jwks_uri` 的 HTTP `GET`。

## 交付它

本课产出 `outputs/skill-mcp-auth.md`。给定 MCP 服务器配置和 IdP 能力集，该技能发出要站立的认证表面——受保护资源元数据、要使用的注册路径（CIMD、预注册或 DCR 回退）、JWKS 刷新计划、作用域映射，以及当 IdP 不支持完整 RFC 配置文件时应用的拒绝规则。

## 练习

1. 运行 `code/main.py`。跟踪流程。注意 IdP 在第 6 步如何轮换密钥，计划的 `refresh_jwks` 重新拉取已发布集合，以及旧 token（重叠窗口）和新 token 都无需重启验证。

2. 向受保护资源元数据的 `authorization_servers` 列表添加新 IdP。用新 IdP 签名的 token 颁发并确认验证器接受它。用未列出的 IdP 签名的 token 颁发并确认验证器以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒绝。

3. 向 `register_client` 添加速率限制检查，在注册器接受请求前运行。使用按源 IP 持有的小 dict 的 token bucket。

4. 阅读 RFC 7591 并找出本课 `/register` 处理器未验证的两个字段。添加验证。（提示：`software_statement` 和 `redirect_uris` URI scheme。）

5. 添加客户端 ID 元数据文档路径。服务一个 `client.json`，其 `client_id` 等于它自己的 URL，并让授权服务器获取并验证它（如果 `client_id` ≠ URL 则拒绝）。确认 CIMD 客户端无需 `register_client` 调用即可注册。

6. 证明 DoS 修复。向验证器发送带随机 `kid` 的 token 并确认 `refresh_jwks` 最多运行一次且授权服务器的密钥计数不增长。然后故意将回退重新连接到轮换并铸造，并观察每个伪造 token 的密钥计数攀升——之后恢复重新获取。

7. 实现客户端侧 RFC 9207 `iss` 检查：记录授权请求前的预期颁发者，然后拒绝 `iss` 不匹配的授权响应。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| ASM | "OAuth 元数据文档" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "客户端元数据 URL" | 客户端 ID 元数据文档——用作 `client_id` 的 HTTPS URL；AS 拉取 JSON。自 2025-11-25 以来的推荐默认 |
| DCR | "自助客户端注册" | RFC 7591 `POST /register` 流程；在 2025-11-25 中降级为 `MAY` 后备 |
| JWKS | "JWT 验证的公钥" | 从 `jwks_uri` 获取、按 `kid` 索引的 JSON Web Key Set |
| 轮换 vs 刷新 | "更新密钥" | *轮换* = AS 铸造/退役签名密钥；*刷新* = 资源服务器重新获取已发布集合。资源服务器只刷新 |
| 资源指示器 | "受众参数" | 将 token 固定到一个服务器的 RFC 8707 `resource` 参数 |
| `aud` 声明 | "受众" | 验证器对照规范资源 URL 比较的 JWT 声明 |
| 受众重播 | "Token 重播" | 为服务器 A 颁发的 token 呈现给服务器 B；由受众验证防御（规范：访问 token 权限限制） |
| 混淆副手 | "代理 token 滥用" | 使用静态客户端 ID 的 MCP 代理在不进行每客户端同意的情况下将 token 转发到上游 API；与受众重播不同 |
| 混合攻击 | "错误的 token 端点" | 客户端被引导到在攻击者的端点兑换诚实 AS 的 code；由客户端通过 RFC 9207 `iss` 防御 |
| `iss` 允许列表 | "可信授权服务器" | 受保护资源元数据的 `authorization_servers` 中命名的集合 |
| `resource_metadata` | "哪里找到 PRM 文档" | 401/403 上命名 RFC 9728 元数据 URL 的 `WWW-Authenticate` 参数 |
| 公共客户端 | "原生或浏览器客户端" | 无 `client_secret` 的 OAuth 客户端；PKCE 补偿 |
| `WWW-Authenticate` | "401/403 响应头部" | 承载驱动客户端恢复的 `Bearer error=...` 指令 |

## 延伸阅读

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) —— 本课实现的 MCP 认证配置文件
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) —— 2025-11-25 的变更（CIMD、XAA、DCR 降级）
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) —— CIMD-over-DCR 原理
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) —— CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) —— 发现契约
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) —— DCR（回退路径）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) —— 公共客户端持有证明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) —— 受众固定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) —— 资源服务器发现
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) —— 防御混合攻击的 `iss` 参数
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) —— 合并的 OAuth 底层
