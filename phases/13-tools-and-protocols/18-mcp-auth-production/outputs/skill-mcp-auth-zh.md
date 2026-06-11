---
name: mcp-auth-wiring
description: 搭建生产 MCP 认证（RFC 8414、CIMD、7591、8707、7636 PKCE、9728、9207）——受保护资源元数据、注册、JWKS 刷新和每次请求 token 验证。
version: 1.1.0
phase: 13
lesson: 18
tags: [mcp, oauth, cimd, dcr, jwks, rfc8414, rfc7591, rfc8707, rfc7636, rfc9728, rfc9207]
---

给定 MCP 服务器配置和 IdP 能力集，发出构成生产 MCP 授权层的认证表面和拒绝规则。

输入：

- `mcp_resource_url` —— 规范资源 URL（最具体的标识符；仅保留路径以区分共托管服务器时），用作 `aud` 和受保护资源元数据的 `resource` 值。
- `idp_metadata_url` —— IdP 的 `/.well-known/oauth-authorization-server`（或 OpenID Connect Discovery）URL。
- `idp_capabilities` —— `code_challenge_methods_supported`、`grant_types_supported`、`client_id_metadata_document_supported`（CIMD）、`registration_endpoint`（DCR）、`response_types_supported`、`authorization_response_iss_parameter_supported`（RFC 9207）的观察值。
- `tools` —— 每个所需作用域的 MCP 工具列表。

产出：

1. **拒绝门控。** 如果任何硬条件失败，拒绝接线并停止：
   - `S256` 缺失于 `code_challenge_methods_supported`（PKCE 无降级模式）。
   - `authorization_code` 缺失于 `grant_types_supported`。
   - `response_types_supported` 不是恰好 `["code"]`。
   - 无注册路径存在：预注册的 `client_id`、`client_id_metadata_document_supported: true`（CIMD）或 `registration_endpoint`（DCR）都不可用。任一个都足够——单独的 DCR 缺失不再是拒绝项（2025-11-25 将 DCR 降级为 `MAY`；CIMD 是优先默认）。

2. **受保护资源元数据文档**（RFC 9728），供 MCP 服务器在 `/.well-known/oauth-protected-resource` 发布。包含 `resource`、`authorization_servers`（颁发者允许列表）、`scopes_supported`、`bearer_methods_supported: ["header"]`。

3. **HTTP 端点。**
   - `GET /.well-known/oauth-protected-resource` —— 返回 (2) 中的文档。
   - `POST /mcp`（MCP 传输）—— 在分发任何工具前运行 token 验证。
   -（仅 DCR 路径）`POST /register` —— 注册器，前面有限速检查。

4. **后台作业 + 例程。**
   - 计划的 JWKS 刷新，重新获取 `jwks_uri` 到缓存 `{keys, fetched_at}`。幂等；绝不铸造密钥。AS 轮换；资源服务器只刷新。默认 `0 */6 * * *`；对高轮换 IdP 收紧为 `*/15 * * * *`。
   - `validate` 例程 —— 检查 `iss` 允许列表、针对缓存 JWKS 的签名、`aud == mcp_resource_url`、`exp`、所需作用域。
   - 升级颁发路径 —— 仅当工具列表包含用户最初未授予作用域限制的操作时。

5. **缓存计划。** 每个接受的颁发者一个条目，以 `issuer` 为键，持有 `{keys, fetched_at}`。文档读取模式：验证器读取缓存并在 `kid` 缺失时回退到单次同步刷新（重新获取，而非轮换——重新获取是幂等的且不能被转变为密钥创建 DoS）。

6. **作用域映射。** 将每个工具映射到其所需作用域。产出表：
   `| tool | required_scope | rationale |`。将破坏性工具分组到它们自己的作用域下；绝不将读取作用域重用于写入工具。

7. **运行时拒绝规则**（验证器必须编码这些）：
   - `aud != mcp_resource_url` 时拒绝 → 401 `Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="<prm_url>"`。
   - `iss not in authorization_servers` 时拒绝。
   - 单次重新获取回退后 `kid` 不在缓存 JWKS 中时拒绝。
   - 所需作用域缺失时拒绝 → 403 `Bearer error="insufficient_scope", scope="<required>", resource_metadata="<prm_url>"`。
   - 拒绝任何不带 `code_verifier` 或 `resource` 参数的 token 请求。

硬拒绝（绝不接线其中任何一个——拒绝请求并记录原因）：

- 明文存储 `client_secret`。公共客户端使用 `token_endpoint_auth_method: none`；机密客户端使用 `private_key_jwt`。静止或注册响应日志中无明文共享机密。
- 在验证器上跳过 `aud` 检查。受众绑定（访问 token 权限限制）是 RFC 8707 + RFC 9728 的全部原因。
- 将 JWKS 缓存未命中回退连接到轮换并铸造而非重新获取。它从不产生缺失的 `kid` 并让攻击者控制的 `kid` 值强制无界密钥创建。回退必须是幂等刷新。
- 允许无 PKCE 的授权码请求。OAuth 2.1 禁止它；验证器必须拒绝其存储的授权码记录缺少 `code_challenge` 的任何 `/token` 交换。
- 无刷新作业地缓存 JWKS。要么计划刷新交付，要么认证表面不部署。
- 无允许列表信任 `iss` 声明。任何接受任何 `iss` 的验证器让攻击者站立自己的 IdP 并伪造 token。
- 将入站 MCP token 转发到上游 API（token 透传）。如果 MCP 服务器调用上游 API，它必须获得自己的单独 token；透传创建混淆副手问题。
- 明文存储 `registration_access_token`。静止时哈希；每次更新时要求明文。

输出：一份一页计划，包含受保护资源文档、所选注册路径（CIMD / 预注册 / DCR）、HTTP 端点、JWKS 刷新作业、缓存计划、作用域映射表和编码的运行时拒绝规则。以对所选 IdP 最可能浮出的单一部署阻塞差距结束——通常是 CIMD 是否已支持，回退到企业 SSO 的 DCR 可用性。
