---
name: mcp-apps-spec
description: 为受益于交互式 UI 的工具产出完整的 MCP Apps 契约。
version: 1.0.0
phase: 13
lesson: 14
tags: [mcp, apps, ui, sep-1724]
---

给定一个会从交互式 UI 受益的工具（可视化、仪表板、编辑器、地图），产出完整的 MCP Apps 契约。

产出：

1. `ui://` URI 设计。单一、可缓存的 URI，如果数据内联则静态，如果主机每次重新读取则动态。
2. CSP 策略。默认 `default-src 'self'`、`script-src 'self'`（无 unsafe-inline 用于生产）、`connect-src 'self'`、`img-src 'self' data:`。
3. 权限请求。仅列出 UI 实际需要的（相机、麦克风、地理位置、网络域）。
4. postMessage 入口点。记录 UI 将调用的 `host.callTool`、`host.readResource` 和 `host.getPrompt` 调用。
5. HTML 骨架。包含 `ui/initialize` 握手和错误处理的最小 HTML 包。

硬拒绝项：
- 任何 `connect-src: *` 的 CSP。默认严格，仅在需要时放松。
- 任何没有 nonce 或哈希的 `script-src 'unsafe-inline'`。内联脚本需要完整性检查。
- 任何请求过多权限的 UI（例如仪表板不需要相机）。

拒绝规则：
- 如果工具输出是纯文本且没有交互价值，拒绝设计 MCP App。
- 如果 UI 需要外部 API 调用但不声明 `connect-src`，拒绝直到 CSP 修复。
- 如果宿主不支持 MCP Apps（检查能力协商），拒绝并提供降级为纯文本。

输出：`tools/call` 响应 JSON、`_meta.ui` 块、CSP 对象、权限列表和 20 行 HTML 骨架。以 iframe 沙盒假设结束。
