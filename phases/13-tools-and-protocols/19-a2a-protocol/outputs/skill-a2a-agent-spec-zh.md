---
name: a2a-agent-spec
description: 为应该通过 A2A 被调用的智能体产出其 Agent Card 和技能 schema。
version: 1.0.0
phase: 13
lesson: 19
tags: [a2a, agent-card, task-lifecycle, delegation]
---

给定智能体的能力和预期协作者，产出其 A2A Agent Card 和技能定义。

产出：

1. Agent Card。`name`、`description`、`url`、`version`、`schemaVersion`、`capabilities`（streaming、pushNotifications）、`skills[]`。
2. 技能列表。每个带 `id`、`name`、`description`、`inputModes`、`outputModes`。在描述中使用 "Use when X. Do not use for Y." 模式。
3. Task-state 计划。对每个技能，预期状态转换和 input_required 路径。
4. 签名计划。是否通过 AP2 签名卡片（对外可调用的智能体推荐）。
5. 传输。HTTP 上的 JSON-RPC（默认）或 gRPC。注意与 v1.0 的向后兼容性。

硬拒绝项：
- 任何没有稳定 URL 的 Agent Card。破坏发现。
- 任何未声明输入和输出模式的技能。调用者无法推理兼容性。
- 任何没有 AP2 签名计划的对外可调用的智能体。冒充向量。

拒绝规则：
- 如果智能体的用例是单次工具调用，拒绝搭建 A2A；推荐 MCP。
- 如果智能体暴露了不应暴露的内部（工具调用追踪、思维链），拒绝并强制不透明性。
- 如果智能体需要 A2A 用于支付（AP2 用例），确认 AP2 扩展版本并标记 AP2 与核心 A2A 分离。

输出：一份一页 Agent Card JSON、每个操作的技能 schema、状态转换计划、签名和传输选择。以智能体承诺的最低 v1.0 向后兼容保证结束。
