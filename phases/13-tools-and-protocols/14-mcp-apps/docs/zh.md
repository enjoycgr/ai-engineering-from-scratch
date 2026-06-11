# MCP Apps —— 通过 `ui://` 的交互式 UI 资源

> 纯文本工具输出限制了 Agent 能展示的内容。MCP Apps（SEP-1724，2026 年 1 月 26 日官方发布）让工具返回沙盒化的交互式 HTML，在 Claude Desktop、ChatGPT、Cursor、Goose 和 VS Code 中内联渲染。仪表板、表单、地图、3D 场景，全部通过一个扩展。本课走过 `ui://` 资源 scheme、`text/html;profile=mcp-app` MIME、iframe 沙盒 postMessage 协议，以及让服务器渲染 HTML 带来的安全表面。

**类型：** Build
**语言：** Python（stdlib，UI 资源发出器），HTML（示例应用）
**前置要求：** Phase 13 · 07（MCP 服务器），Phase 13 · 10（资源）
**时间：** ~75 分钟

## 学习目标

- 从工具调用返回 `ui://` 资源并设置正确的 MIME 和元数据。
- 用 `_meta.ui.resourceUri`、`_meta.ui.csp` 和 `_meta.ui.permissions` 声明工具的关联 UI。
- 实现 iframe 沙盒 postMessage JSON-RPC 用于 UI 到宿主通信。
- 应用 CSP 和权限策略默认值，防御源自 UI 的攻击。

## 问题

2025 年的 `visualize_timeline` 工具可以返回 "Here are 14 notes organized chronologically: ..."。那是一段文字。用户实际上想要交互式时间线。MCP Apps 之前，选项是：客户端特定的小部件 API（Claude artifacts、OpenAI Custom GPT HTML），或根本没有 UI。

MCP Apps（SEP-1724，2026 年 1 月 26 日交付）标准化了契约。工具结果包含一个 URI 为 `ui://...` 且 MIME 为 `text/html;profile=mcp-app` 的 `resource`。宿主在沙盒 iframe 中渲染它，具备有限的 CSP 和除非显式授予否则无网络访问。UI 内的 iframe 通过一个小型 postMessage JSON-RPC 方言向宿主发送消息。

每个兼容客户端（Claude Desktop、ChatGPT、Goose、VS Code）以相同方式渲染相同的 `ui://` 资源。一个服务器，一个 HTML 包，通用 UI。

## 概念

### `ui://` 资源 scheme

工具返回：

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

宿主随后在 `ui://notes/timeline` URI 上调用 `resources/read` 并取回：

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe 沙盒

宿主在沙盒 `<iframe>` 中渲染 HTML：

- `sandbox="allow-scripts allow-same-origin"`（或根据服务器声明更严格）
- 通过响应头部应用服务器声明的 CSP。
- 无 cookie，无来自宿主来源的 localStorage。
- 网络访问限于 CSP 中的 `connectSrc`。

### postMessage 协议

Iframe 通过 `window.postMessage` 与宿主通信。一个小型 JSON-RPC 2.0 方言：

始终将 `targetOrigin` 固定到对等方的确切来源，在接收端验证 `event.origin` 是否在 allowlist 中，然后再处理任何载荷。永远不要在此通道的任何一侧使用 `"*"`——主体携带工具调用和资源读取。

```js
// iframe 到宿主（固定到宿主来源）
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 宿主到 iframe（固定到 iframe 来源）
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 两侧的接收器
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // 安全处理 event.data
});
```

UI 可以调用的宿主端方法：

- `host.callTool(name, arguments)` —— 调用服务器工具。
- `host.readResource(uri)` —— 读取 MCP 资源。
- `host.getPrompt(name, arguments)` —— 获取提示模板。
- `host.close()` —— 关闭 UI。

每次调用仍通过 MCP 协议并继承服务器的权限。

### 权限

`_meta.ui.permissions` 列表请求额外能力：

- `camera` —— 访问用户相机（用于扫描文档 UI）。
- `microphone` —— 语音输入。
- `geolocation` —— 位置。
- `network:*` —— 比 `connectSrc` 单独允许的更广泛的网络访问。

每个权限都是用户在 UI 渲染前看到的提示。

### 安全风险

iframe 中的 HTML 仍然是 HTML。新的攻击表面：

- **通过 UI 的提示注入。** 恶意服务器 UI 可以展示看起来像系统消息的文本并欺骗用户。宿主渲染应明显区分服务器 UI 和宿主 UI。
- **通过 `connectSrc` 的渗透。** 如果 CSP 允许 `connect-src: *`，UI 可以将数据发送到任何地方。默认值应严格。
- **点击劫持。** UI 覆盖宿主 chrome。宿主必须防止 z-index 操纵并强制执行不透明度规则。
- **窃取焦点。** UI 获取键盘焦点并捕获下一条消息。宿主必须拦截。

Phase 13 · 15 深入讲解这些作为 MCP 安全的一部分；本课介绍它们。

### `ui/initialize` 握手

Iframe 加载后，它通过 postMessage 发送 `ui/initialize`：

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

宿主以能力和会话 token 响应。UI 在每次后续宿主调用上使用会话 token。

### AppRenderer / AppFrame SDK 原语

ext-apps SDK 暴露两个便利原语：

- `AppRenderer`（服务器端）—— 包装 React / Vue / Solid 组件并发出具备正确 MIME 和元数据的 `ui://` 资源。
- `AppFrame`（客户端）—— 接收资源，挂载 iframe，并调解 postMessage。

你可以使用这些或手写 HTML 和 JSON-RPC。

### 生态系统状态

MCP Apps 于 2026 年 1 月 26 日交付。截至 2026 年 4 月的客户端支持：

- **Claude Desktop。** 2026 年 1 月起完全支持。
- **ChatGPT。** 通过 Apps SDK 完全支持（相同底层 MCP Apps 协议）。
- **Cursor。** Beta；通过设置启用。
- **VS Code。** 仅 Insider 构建。
- **Goose。** 完全支持。
- **Zed、Windsurf。** 已规划。

生产中的服务器：仪表板、地图可视化、数据表、图表构建器、沙盒 IDE 预览。

## 使用它

`code/main.py` 扩展了笔记服务器，添加一个 `visualize_timeline` 工具，返回 `ui://notes/timeline` 资源，加上该 URI 上 `resources/read` 的处理器，返回一个带 SVG 时间线的小型但完整的 HTML 包。HTML 是 stdlib 模板化的——无构建系统。postMessage 在 JS 注释中草拟，因为 stdlib 无法驱动浏览器。

看点：

- 工具响应上的 `_meta.ui` 携带 resourceUri、CSP、权限。
- HTML 在无网络访问的情况下渲染；所有数据都是内联的。
- JS 通过 `window.parent.postMessage` 调用 `host.callTool`（在 stdlib 演示中已记录但无效）。

## 交付它

本课产出 `outputs/skill-mcp-apps-spec.md`。给定一个会从交互式 UI 受益的工具，该技能产出完整的 MCP Apps 契约：`ui://` URI、CSP、权限、postMessage 入口点和一个安全清单。

## 练习

1. 运行 `code/main.py` 并检查发出的 HTML。直接在浏览器中打开 HTML；验证 SVG 渲染。然后草拟 UI 将用于调用 `host.callTool("notes_update", ...)` 的 postMessage 契约。

2. 收紧 CSP：移除 `'unsafe-inline'` 并使用基于 nonce 的脚本策略。HTML 生成代码中有什么变化？

3. 添加第二个 UI 资源 `ui://notes/editor`，带一个就地编辑笔记的表单。当用户提交时，iframe 调用 `host.callTool("notes_update", ...)`。

4. 审计 UI 的攻击表面。恶意服务器可以在哪里注入内容？iframe 沙盒防御什么、不防御什么？

5. 阅读 SEP-1724 规范并找出本玩具实现未使用的 MCP Apps SDK 中的一个能力。（提示：组件级状态同步。）

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| MCP Apps | "交互式 UI 资源" | 2026-01-26 交付的 SEP-1724 扩展 |
| `ui://` | "App URI scheme" | UI 包的资源 scheme |
| `text/html;profile=mcp-app` | "MIME" | MCP App HTML 的内容类型 |
| Iframe 沙盒 | "渲染容器" | 具备 CSP 和权限的浏览器 UI 沙盒 |
| postMessage JSON-RPC | "UI 到宿主的线路" | 用于宿主调用的小型 JSON-RPC-over-postMessage 方言 |
| `_meta.ui` | "工具-UI 绑定" | 将工具结果链接到 UI 资源的元数据 |
| CSP | "内容安全策略" | 声明脚本、网络、样式的允许来源 |
| AppRenderer | "服务器 SDK 原语" | 将框架组件转换为 `ui://` 资源 |
| AppFrame | "客户端 SDK 原语" | 挂载 iframe 并调解 postMessage 的助手 |
| `ui/initialize` | "握手" | UI 到宿主的第一个 postMessage |

## 延伸阅读

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) —— 参考实现和 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) —— 正式规范文档
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) —— 高级文档
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) —— 2026 年 1 月发布文章
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) —— JSDoc 风格 SDK 参考
