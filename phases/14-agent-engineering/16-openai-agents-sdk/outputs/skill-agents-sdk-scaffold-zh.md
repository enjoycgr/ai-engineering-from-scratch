---
name: agents-sdk-scaffold
description: 搭建 OpenAI Agents SDK 应用，包含 triage (分流) agent (智能体)、handoff (交接)、input/output/tool guardrail (输入/输出/工具护栏)、session store (会话存储) 和 trace processor (追踪处理器)。
version: 1.0.0
phase: 14
lesson: 16
tags: [openai, agents-sdk, handoffs, guardrails, tracing, session]
---

给定一个产品领域和一组 specialist agent (专业智能体)，搭建一个 OpenAI Agents SDK 应用。

产出：

1. 每个 specialist (专业) 一个 `Agent`，再加上一个只有 handoff (交接) 的 `triage` (分流) agent（没有领域工具）。
2. 每个领域工具一个 `FunctionTool`，带类型化的 input schema (输入模式)、清晰的 description（告诉模型何时使用它）和执行 sandbox (沙箱)。
3. 从 triage (分流) 到每个 specialist (专业) 的 `Handoff`。验证工具名遵循 `transfer_to_<agent>` 约定。
4. `InputGuardrail` 用于 PII (个人身份信息)、策略、范围。默认使用 parallel (并行) 模式，除非 guardrail (护栏) LLM 相对于主模型很大——此时使用 blocking (阻塞)。
5. `OutputGuardrail` 用于长度、PII (个人身份信息)、策略。生产中对安全关键输出始终使用 blocking (阻塞)。
6. 触及网络或文件系统的函数工具上的 per-tool guardrail (工具护栏)。
7. `Session` store (存储)（默认 SQLite；生产 Redis）。
8. `add_trace_processor` 将 span (跨度) 扇出到你的后端，与 OpenAI 的追踪 UI 并存。

Hard rejects：

- Triage agent (分流智能体) 带有领域工具。Triage (分流) 只应 handoff (交接)；混合会稀释 router (路由器) 的决策。
- 会修改 input/output (输入/输出) 的 guardrail (护栏)。Guardrail (护栏) 只能批准或拒绝——不能重写。
- 静默的 handoff loop (交接循环)。要求 hop counter (跳转计数器)（默认最大 3）。

Refusal rules：

- 如果用户想要"没有 guardrail (护栏)，只管快速推进"，对于任何触及付费用户或 PII (个人身份信息) 的产品，拒绝。
- 如果产品只有 2 个 specialist (专业)，建议通过 `Agents` 使用 direct classifier (直接分类器)（Lesson 12）而不是 triage+handoff (分流+交接)——token (令牌) 成本更低。
- 如果 tracing (追踪) 在生产中被禁用，拒绝上线。没有 trace (追踪)，多步故障无法调试。

输出：`agents.py`、`tools.py`、`guardrails.py`、`app.py`、`README.md`，包含 triage agent (分流智能体) 原理、guardrail (护栏) 模式、trace processor (追踪处理器) 和 session backend (会话后端)。结尾附 "what to read next"，指向 Lesson 23（OTel GenAI）、Lesson 24（observability backends），或 Lesson 17（Claude Agent SDK 迁移）。
