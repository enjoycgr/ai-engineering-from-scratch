# A2A —— 智能体到智能体协议

> MCP 是智能体到工具。A2A（Agent2Agent）是智能体到智能体——一个开放协议，让基于不同框架构建的不透明智能体协作。由 Google 于 2025 年 4 月发布，2025 年 6 月捐赠给 Linux Foundation，2026 年 4 月达到 v1.0，拥有 150+ 支持者包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP 并添加了 AP2 支付扩展。本课走过 Agent Card、Task 生命周期和两个传输绑定。

**类型：** Build
**语言：** Python（stdlib，Agent Card + Task harness）
**前置要求：** Phase 13 · 06（MCP 基础），Phase 13 · 08（MCP 客户端）
**时间：** ~75 分钟

## 学习目标

- 区分智能体到工具（MCP）和智能体到智能体（A2A）用例。
- 在 `/.well-known/agent.json` 发布 Agent Card，包含技能和端点元数据。
- 走过 Task 生命周期（submitted → working → input-required → completed / failed / canceled / rejected）。
- 使用带 Parts（text、file、data）的 Messages 和作为输出的 Artifacts。

## 问题

客户服务智能体需要将报告撰写委托给专门的写作智能体。A2A 之前的选项：

- 自定义 REST API。有效但每对配对都是一次性的。
- 共享代码库。要求两个智能体运行相同框架。
- MCP。不合适：MCP 用于调用工具，而非两个智能体在保留每个智能体不透明内部推理的同时协作。

A2A 填补了空白。它将交互建模为一个智能体向另一个智能体发送 Task，具备生命周期、消息和工件。被调用智能体的内部状态保持不透明——调用者只看到任务状态转换和最终输出。

A2A 是 "让跨框架的智能体彼此对话" 的协议。它不替代 MCP；两者是互补的。

## 概念

### Agent Card

每个符合 A2A 的智能体在 `/.well-known/agent.json` 发布卡片：

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

发现是基于 URL 的：获取卡片，学习 A2A 端点 URL，枚举技能。

### 签名 Agent Cards（AP2）

AP2 扩展（2025 年 9 月）向 Agent Cards 添加加密签名。发布者用自己的 JWT 签名其卡片；消费者验证。防止冒充。

### Task 生命周期

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (通过消息循环)
```

客户端用 `tasks/send` 启动。被调用智能体转换状态；客户端通过 SSE 或轮询订阅状态更新。

### Messages 和 Parts

消息携带一个或多个 Parts：

- `text` —— 纯内容。
- `file` —— 带 mimeType 的 base64 blob。
- `data` —— 用于被调用智能体的结构化输入的键入 JSON 载荷。

示例：

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

输出是 Artifacts，而非原始字符串。Artifact 是命名的、类型化的输出：

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifacts 可以作为块流式传输。调用者累积它们。

### 两个传输绑定

1. **HTTP 上的 JSON-RPC。** `/a2a` 端点，POST 用于请求，可选 SSE 用于流式传输。默认绑定。
2. **gRPC。** 用于 gRPC 原生存在的企业环境。

两个绑定都携带相同的逻辑消息形状。

### 不透明性保留

一个关键设计原则：被调用智能体的内部状态是不透明的。调用者看到任务状态和工件。被调用智能体的思维链、其工具调用、其子智能体委托——全部不可见。这与 MCP 不同，在 MCP 中工具调用是透明的。

原理：A2A 让竞争对手协作而无需暴露内部。A2A 可以是 "调用此客户服务智能体" 而调用者不了解该智能体如何实现服务。

### 时间线

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐赠给 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 扩展（Agent Payments）交付。
- **2026-04。** v1.0 发布，拥有 150+ 支持组织。

### 与 MCP 的关系

| 维度 | MCP | A2A |
|-----------|-----|-----|
| 用例 | 智能体到工具 | 智能体到智能体 |
| 不透明性 | 透明工具调用 | 不透明内部推理 |
| 典型调用者 | 智能体运行时 | 另一个智能体 |
| 状态 | 工具调用结果 | 具备生命周期的 Task |
| 授权 | OAuth 2.1（Phase 13 · 16） | 带 AP2 的 JWT 签名 Agent Cards |
| 传输 | Stdio / Streamable HTTP | HTTP 上的 JSON-RPC / gRPC |

当你想要调用特定工具时使用 MCP。当你想要将整个任务委托给另一个智能体时使用 A2A。许多生产系统同时使用两者：智能体使用 MCP 作为其工具层，使用 A2A 作为其协作层。

## 使用它

`code/main.py` 实现了一个最小的 A2A harness：一个研究智能体发布其卡片，一个写作智能体接收带 Parts 的 `tasks/send`（包括 PDF 和文本指令），转换 working → input_required → working → completed，并返回文本工件。全部标准库；使用进程内传输专注于消息形状。

看点：

- Agent Card JSON 形状。
- Task id 分配和状态转换。
- 混合类型 Parts 的消息。
- 任务中的 Input-required 分支。
- 完成时的 Artifact 返回。

## 交付它

本课产出 `outputs/skill-a2a-agent-spec.md`。给定一个应该被其他智能体调用的新智能体，该技能产出 Agent Card JSON、技能 schema 和端点蓝图。

## 练习

1. 运行 `code/main.py`。跟踪完整的 Task 生命周期，包括被调用智能体请求澄清的 input-required 暂停。

2. 添加签名 Agent Card。用卡片规范 JSON 上的 HMAC 签名。编写验证器并确认它在变异卡片上失败。

3. 实现任务流式传输：写作智能体通过 SSE 发出三个增量工件块，调用者累积它们。

4. 设计一个包装 MCP 服务器的 A2A 智能体。将每个 MCP 工具映射到 A2A 技能。注意权衡——丢失了哪些不透明性？

5. 阅读 A2A v1.0 公告并找出截至 2026 年 4 月任何框架都尚未实现的一个功能。（提示：它与多跳任务委托有关。）

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| A2A | "智能体到智能体协议" | 不透明智能体协作的开放协议 |
| Agent Card | "`/.well-known/agent.json`" | 描述智能体技能和端点的已发布元数据 |
| Skill | "可调用单元" | 智能体支持的命名操作（MCP 工具的类比） |
| Task | "委托单元" | 具备生命周期和最终工件的工作项 |
| Message | "任务输入" | 携带 Parts（text、file、data） |
| Part | "类型化块" | 消息的 `text` / `file` / `data` 元素 |
| Artifact | "任务输出" | 完成时返回的命名、类型化输出 |
| AP2 | "Agent Payments Protocol" | 用于信任和支付的签名 Agent Cards 扩展 |
| 不透明性 | "黑盒协作" | 被调用智能体的内部对调用者隐藏 |
| Input-required | "任务暂停" | 智能体需要更多信息时的生命周期状态 |

## 延伸阅读

- [a2a-protocol.org](https://a2a-protocol.org/latest/) —— 规范 A2A 规范
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) —— 参考实现和 SDK
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) —— 2025 年 6 月治理转移
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) —— 路线图和合作伙伴势头
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) —— v1.0 发布说明和向后兼容指导
