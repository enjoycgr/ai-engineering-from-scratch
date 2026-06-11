# A2A — Agent-to-Agent Protocol (智能体间协议)

> Google 于 2025 年 4 月宣布 A2A；到 2026 年 4 月，规范位于 https://a2a-protocol.org/latest/specification/，150 多个组织支持它。A2A 是 MCP（第 13 课）的水平补充：MCP 是垂直的（agent ↔ 工具），A2A 是点对点的（agent ↔ agent）。它定义了 Agent Card (智能体卡片)（发现）、带 artifact (产物) 的 task (任务)（文本、结构化数据、视频）、opaque task lifecycle (不透明任务生命周期) 以及 auth (认证)。生产级 multi-agent (多智能体) 系统越来越多地将 MCP 与 A2A 配对使用。Google Cloud 在 2025-2026 年期间将 A2A 支持引入 Vertex AI Agent Builder。

**Type:** Learn + Build
**Languages:** Python (stdlib, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 分钟

## Problem (问题)

你的 agent 需要调用另一个系统上的另一个 agent。怎么做？你可以暴露一个 HTTP endpoint (端点)，定义一个 bespoke JSON schema (定制 JSON 模式)，并希望对方能懂它。每对 agent 都变成了一个 custom integration (定制集成)。

A2A 就是那个调用的通用 wire protocol (有线协议)。标准的发现机制、标准的 task model (任务模型)、标准的 transport (传输)、标准的 artifact。就像 HTTP+REST，但将 agent 作为一等公民。

## Concept (概念)

### The four elements (四个要素)

**Agent Card (智能体卡片)。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述 agent：名称、技能、端点、支持的 modalities (模态)、auth 要求。发现机制通过读取卡片完成。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task (任务)。** 工作单元。一个 async (异步)、stateful (有状态) 的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送 task，轮询或订阅更新。

**Artifact (产物)。** Task 产生的结果类型。文本、结构化 JSON、图像、视频、音频。Artifact 是 typed (有类型的)，因此不同 modalities 都是一等公民。

**Opaque lifecycle (不透明生命周期)。** A2A 不规定远程 agent 如何解决 task。客户端看到 state transitions (状态转换) 和 artifacts；实现可以自由使用任何框架。

### The MCP/A2A split (MCP 与 A2A 的分工)

- **MCP**（第 13 课）：agent ↔ tool (工具)。Agent 通过 JSON-RPC 读写 tool server。默认无状态。
- **A2A**：agent ↔ agent。Peer protocol (对等协议)；双方都是具有自己推理能力的 agent。

生产级 multi-agent 系统两者都用。一个 A2A peer 在其侧调用 MCP tools。这种分工让两个关注点保持清晰。

### Discovery flow (发现流程)

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或使用 streaming (流式)：通过 SSE 订阅 `/tasks/{id}/events` 获取 push updates (推送更新)。

### Auth (认证)

A2A 支持三种常见模式：

- **Bearer token (持有者令牌)** — OAuth2 或 opaque (不透明令牌)。
- **mTLS (双向 TLS)** — 组织间相互证明身份。
- **Signed requests (签名请求)** — 对 payload 进行 HMAC 签名。

Auth 在 Agent Card 中声明；客户端发现并遵守。

### 150+ organizations by April 2026 (截至 2026 年 4 月，150 多个组织)

企业采用推动了 A2A 的规模。 headline 是：A2A 成为企业 agent 系统跨越 trust boundaries (信任边界) 的方式。Google Cloud 推出了 Vertex AI Agent Builder A2A 支持；Microsoft Agent Framework 支持它；大多数主流框架（LangGraph、CrewAI、AutoGen）都提供 A2A adapters (适配器)。

### Where A2A wins (A2A 的优势场景)

- **Cross-organization calls (跨组织调用)。** 公司 A 的 agent 调用公司 B 的 agent。没有 A2A，每对组合都是一份 bespoke contract (定制合约)。
- **Heterogeneous frameworks (异构框架)。** LangGraph agent 调用 CrewAI agent 调用自定义 Python agent。A2A 将其规范化。
- **Typed artifacts (有类型的产物)。** 视频结果、结构化 JSON、音频——全都是一等公民。
- **Long-running tasks (长时间运行的任务)。** Opaque lifecycle + polling 让数小时长的任务变得简单。

### Where A2A struggles (A2A 的劣势场景)

- **Latency-sensitive micro-calls (延迟敏感的微调用)。** A2A 的 lifecycle 是异步的。亚毫秒级的 agent-to-agent 不适合；使用 direct RPC (直接 RPC)。
- **Tight-coupled in-process agents (紧耦合的进程内 agent)。** 如果两个 agent 运行在同一个 Python 进程中，A2A 的 HTTP round-trip (往返) 是多余的。
- **Small teams (小团队)。** Spec overhead (规范开销) 是真实的；仅内部使用的 agent 可能不需要这种正式性。

### A2A vs ACP, ANP, NLIP

2024-2026 年间出现了几个相关规范：

- **ACP** (IBM/Linux Foundation) — A2A 的前身，范围更窄。
- **ANP** (Agent Network Protocol) — 偏重 peer-discovery (对等发现)，去中心化优先。
- **NLIP** (Ecma Natural Language Interaction Protocol，2025 年 12 月标准化) — natural-language content type (自然语言内容类型)。

截至 2026 年 4 月，A2A 是应用最广泛的 peer protocol。比较详见 arXiv:2505.02279（Liu 等人，"A Survey of Agent Interoperability Protocols"）。

## Build It (动手实现)

`code/main.py` 使用 `http.server` 和 JSON 实现了一个 A2A-minimal (最小化 A2A) 服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理 task state (任务状态)，
- 在 `GET /tasks/{id}` 上返回 artifacts。

客户端：

- 获取 Agent Card，
- 提交 task，
- 轮询直到完成，
- 读取 artifact。

运行：

```
python3 code/main.py
```

脚本在后台线程中启动服务器，然后针对它运行客户端。你可以看到完整的流程：发现、提交、轮询、artifact。

## Use It (使用它)

`outputs/skill-a2a-integrator.md` 设计一个 A2A 集成：Agent Card 内容、task schemas (任务模式)、auth 选择、streaming vs polling (流式 vs 轮询)。

## Ship It (交付上线)

Checklist (检查清单)：

- **Pin the spec version (固定规范版本)。** A2A 仍在演进；Agent Card 应声明 protocol version (协议版本)。
- **Idempotent task creation (幂等的任务创建)。** 重复提交（网络重试）应产生一个 task。
- **Artifact schemas (产物模式)。** 声明 agent 返回什么形状；消费者应验证。
- **Rate limits + auth (速率限制 + 认证)。** A2A 是面向公众的；应用标准 Web 安全。
- **Dead-letter for failed tasks (失败任务的死信)。** 长期检查 recurring failure types (反复出现的故障类型) 的模式。

## Exercises (练习)

1. 运行 `code/main.py`。确认客户端发现服务器并接收到正确的 artifact。
2. 为服务器添加第二个 skill（例如，"summarize"）。更新 Agent Card。编写一个根据 task type 选择 skill 的客户端。
3. 实现一个 SSE streaming endpoint：`/tasks/{id}/events`，用于 emit (发出) state changes (状态变更)。客户端需要做什么不同的事？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。找出规范强制要求而本演示未实现的三件事。
5. 比较 A2A（Agent Card 发现）和 MCP（通过 `listTools` 进行服务器端能力列表）。Self-describing agents (自描述智能体) 和 capability-probing (能力探测) 之间的 tradeoff (权衡) 是什么？

## Key Terms (关键术语)

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| A2A | "Agent-to-agent" | Agent 跨系统调用其他 agent 的 peer protocol。Google 2025。 |
| Agent Card (智能体卡片) | "The agent's business card" | 位于 `/.well-known/agent.json` 的 JSON，描述技能、端点、auth。 |
| Task (任务) | "The unit of work" | 具有生命周期的 async stateful (异步有状态) 对象；完成时产生 artifacts。 |
| Artifact (产物) | "The result" | Typed output：文本、结构化 JSON、图像、视频、音频。First-class media (一等媒体)。 |
| Opaque lifecycle (不透明生命周期) | "How it's solved is the agent's business" | 客户端看到 state transitions；服务器可自由选择框架/工具。 |
| Discovery (发现) | "Finding the agent" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP vs A2A | "Tools vs peers" | MCP：垂直 agent ↔ tool。A2A：水平 agent ↔ agent。 |
| ACP / ANP / NLIP | "Sibling protocols" | 相邻规范；A2A 是 2026 年应用最广泛的。 |

## Further Reading (延伸阅读)

- [A2A specification](https://a2a-protocol.org/latest/specification/) — 规范原文
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布文章
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 比较
