---
name: agent-loop
description: 在任何目标语言/运行时中编写一个正确、极简的 ReAct agent loop（智能体循环），包含工具、停止条件和轮次预算。
version: 1.0.0
phase: 14
lesson: 01
tags: [react, agent-loop, tools, observability, stop-condition]
---

给定一个目标运行时（Python async、Python sync、Node、Rust async、Go）和一个工具列表（名称、输入 schema、可调用对象），在第一次尝试时就产出一个正确的 ReAct agent loop（智能体循环）。

产出内容：

1. 一个 message-buffer（消息缓冲区）类型，包含角色 {user, assistant, tool, final}，以及目标提供商所期望的 schema（Anthropic `tool_use` / `tool_result` 块、OpenAI function-calling 消息、Responses API reasoning channel）。绝不在提供商之间静默交换 schema。
2. 一个 tool registry（工具注册表），带名称 -> 可调用对象分派、输入验证和类型化结果。错误必须被捕获并转换为 observation（观察）字符串，绝不能抛给循环。
3. 一个循环，运行直到满足以下之一：显式 `finish` action、助手轮中没有工具调用、最大轮数、最大总 token 数，或 guardrail 触发。恰好选择一个主停止条件；其余为安全网。
4. 一个按任务类别缩放的 turn budget（轮次预算）——短任务 10、computer-use（计算机使用）200、deep research（深度研究）400。明确说明选择理由。
5. 一个 trace record（轨迹记录），记录每个 thought（思考）、action（行动）、observation（观察）和停止原因。当运行时有 OTel SDK 时，发出 OpenTelemetry GenAI span（`invoke_agent`、`tool_call`）。

Hard rejects（硬拒绝）：

- 没有 turn cap（轮次上限）的循环。这是可靠性问题，不是优化问题。
- 将工具错误吞并为空 observation。模型必须看到失败文本，以便纠正。
- 将检索到的内容视为可信指令。所有工具输出都是 untrusted input（不可信输入）——只有用户消息携带许可（参见 OpenAI CUA 文档）。
- 在没有 schema-translation layer（schema 转换层）的情况下混合提供商。Anthropic 和 OpenAI 有不同的工具 schema 和消息形状。

Refusal rules（拒绝规则）：

- 如果目标是"no framework, bash only（无框架，仅 bash）"，拒绝并推荐至少一个类型化的消息 schema；agent loop（智能体循环）对无类型的 shell glue 来说太容易出错。
- 如果用户要求"auto-retry on failed tool call without feedback to the model（工具调用失败时自动重试，但不向模型反馈）"，拒绝。重试必须要么通过模型（CRITIC/Self-Refine，Lesson 05），要么作为工具自身幂等契约的一部分。
- 如果工具列表包含没有 human-in-the-loop（人在回路中）确认的破坏性工具，拒绝并指向 Lesson 09（permissions + sandboxing，权限 + 沙箱）。

输出：每种语言目标一个文件，外加一个 `README.md`，解释 stop-condition（停止条件）的选择、turn budget（轮次预算）的理由，以及一个展示每步 thought-action-observation 的完整 trace。结尾附 "what to read next（下一步阅读建议）"：如果任务是 long-horizon（长程），指向 Lesson 02（ReWOO planning）；如果是 repeat-of-previous（重复之前），指向 Lesson 03（Reflexion）；如果工具接触不可信内容，指向 Lesson 27（prompt injection）。
