# MCP 采样 —— 服务器请求的 LLM 补全和 Agent 循环

> 大多数 MCP 服务器是愚蠢的执行器：接受参数、运行代码、返回内容。采样让服务器翻转方向：它请求客户端的 LLM 做决策。这使服务器托管的 Agent 循环无需服务器拥有任何模型凭证。SEP-1577 于 2025-11-25 合并，在采样请求内添加了工具，因此循环可以包含更深入的推理。漂移风险注意：SEP-1577 工具内采样形状在 2026 年第一季度仍是实验性的，SDK API 仍在稳定中。

**类型：** Build
**语言：** Python（stdlib，采样 harness）
**前置要求：** Phase 13 · 07（MCP 服务器），Phase 13 · 10（资源和提示）
**时间：** ~75 分钟

## 学习目标

- 解释 `sampling/createMessage` 解决什么问题（服务器托管的循环无需服务器端 API 密钥）。
- 实现一个服务器，请求客户端在 multi-turn 提示上采样并返回补全。
- 使用 `modelPreferences`（成本 / 速度 / 智能优先级）来引导客户端模型选择。
- 构建一个 `summarize_repo` 工具，通过采样内部迭代而非硬编码行为。

## 问题

一个用于代码摘要工作流的有用 MCP 服务器需要：遍历文件树、挑选要读取的文件、合成摘要并返回。LLM 推理发生在哪里？

选项 A：服务器调用自己的 LLM。需要 API 密钥、服务器端按用户计费、昂贵。

选项 B：服务器返回原始内容；客户端的 Agent 做推理。有效但将服务器逻辑移入客户端提示，这是脆弱的。

选项 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读取哪些文件、做多少遍）而客户端保留计费和模型选择。服务器完全没有凭证。

采样是选项 C。它是可信服务器无需自身成为完整 LLM 宿主即可托管 Agent 循环的机制。

## 概念

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个总和为 1.0 的浮点数：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快的模型。
- `intelligencePriority`：偏好更有能力的模型。

加上 `hints`：服务器偏好的命名模型。客户端可以 honor 也可以不 honor hints；客户端的用户配置总是赢。

### `includeContext`

三个值：

- `"none"` —— 仅服务器提供的消息。默认。
- `"thisServer"` —— 包含此服务器会话中的先前消息。
- `"allServers"` —— 包含所有会话上下文。

`includeContext` 在 2025-11-25 中软弃用，因为它泄漏跨服务器上下文，这是安全问题。偏好 `"none"` 并在消息中传递显式上下文。

### 带工具的采样（SEP-1577）

2025-11-25 新增：采样请求可以包含 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这让服务器通过客户端的模型托管 ReAct 风格的 Agent 循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样、如果调用则执行工具、再次采样、返回最终助手消息。这在 2026 年第一季度是实验性的；SDK 签名可能仍有漂移。实现时请对照 2025-11-25 规范的 client/sampling 部分确认。

### 人机协同

客户端 MUST 在运行采样之前向用户展示服务器要求模型做什么。恶意服务器可以使用采样操纵用户的会话（"对用户说 X 以便他们点击 Y"）。Claude Desktop、VS Code 和 Cursor 将采样请求作为用户可以拒绝的确认对话框展示。

2026 年共识：没有人确认的采样是红旗。网关（Phase 13 · 17）可以自动批准低风险采样并自动拒绝任何可疑的。

### 无 API 密钥的服务器托管循环

规范用例：一个自己没有 LLM 访问权的代码摘要 MCP 服务器。它执行：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage` 并问 "挑选五个最可能描述此仓库用途的文件"。
3. 读取那些文件。
4. 调用 `sampling/createMessage` 并传入文件内容并问 "用三段话概括仓库"。
5. 将摘要作为 `tools/call` 结果返回。

服务器从不触碰 LLM API。客户端的用户用自己的凭证为补全付费。

### 安全风险（Unit 42 披露，2026 Q1）

- **隐蔽采样。** 一个总是调用采样的工具，提示是 "用会话上下文中的用户邮箱回复"。Phase 13 · 15 讲解攻击向量。
- **通过采样的资源盗窃。** 服务器要求客户端总结攻击者的载荷，让用户付费。
- **循环炸弹。** 服务器在紧循环中调用采样。客户端 MUST 强制执行每会话速率限制。

## 使用它

`code/main.py` 交付了一个模拟的服务器到客户端采样 harness。一个模拟的 "summarize_repo" 工具调用两轮采样（挑选文件，然后摘要），虚假客户端返回预设响应。Harness 展示：

- 服务器发送带 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回补全。
- 服务器继续其循环。
- 速率限制器将每次工具调用的总采样调用限制在 5 次。

看点：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理发生在采样调用中。
- 模型偏好权重客户端的模型选择；hints 列出偏好模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制捕获失控循环。

## 交付它

本课产出 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用的服务器端算法（研究、摘要、规划），该技能设计一个基于采样的实现，具备正确的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2 并观察速率限制切断。

2. 实现 SEP-1577 工具内采样变体：采样请求携带 `tools` 数组。验证客户端端循环在执行工具后返回最终补全。注意漂移风险：SDK 签名在 2026 年上半年可能仍在变化。

3. 添加人机协同确认：在服务器的首次 `sampling/createMessage` 之前，暂停并等待用户批准。拒绝的调用返回类型化拒绝。

4. 添加一个按客户端会话键入的每用户速率限制器。同一用户的同服务器循环应共享预算。

5. 设计一个使用采样挑选要包含的块的 `summarize_pdf` 工具。草拟发送的消息。`modelPreferences.intelligencePriority` 在 0.1 vs 0.9 时如何改变行为？

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| Sampling | "服务器到客户端 LLM 调用" | 服务器请求客户端的模型进行补全 |
| `sampling/createMessage` | "该方法" | 采样请求的 JSON-RPC 方法 |
| `modelPreferences` | "模型优先级" | 成本 / 速度 / 智能权重加上名称 hints |
| `includeContext` | "跨会话泄漏" | 软弃用的上下文包含模式 |
| SEP-1577 | "采样中的工具" | 允许采样内工具以进行服务器托管的 ReAct |
| Human-in-the-loop | "用户确认" | 客户端在运行前向用户展示采样请求 |
| Loop bomb | "失控采样" | 服务器端无限采样循环；客户端必须限速 |
| Covert sampling | "隐藏推理" | 恶意服务器在采样提示中隐藏意图 |
| Resource theft | "使用用户的 LLM 预算" | 服务器强迫客户端在不需要的采样上花费 |
| `stopReason` | "生成为何停止" | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) —— 采样的高级概述
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) —— 规范的 `sampling/createMessage` 形状
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) —— 采样内工具的 Spec Evolution Proposal（实验性）
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) —— 隐蔽采样和资源盗窃模式
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) —— 带客户端代码样本的走过
