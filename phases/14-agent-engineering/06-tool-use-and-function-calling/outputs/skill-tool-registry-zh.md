---
name: tool-registry
description: 构建生产级工具目录和注册表，包含 JSON Schema 校验、并行分发和可观测性。
version: 1.0.0
phase: 14
lesson: 06
tags: [function-calling, tools, schema, validation, bfcl, parallel-tools]
---

给定一个任务域，生成一个工具目录，供智能体在 BFCL V4 (Berkeley Function Calling Leaderboard V4) 各维度（agentic、multi-turn、live、non-live、hallucination）上可靠使用。

产出：

1. 工具定义。对每个工具：`name`（snake_case）、`description`（告知模型何时使用、何时不使用）、带类型化 properties 的 JSON Schema 输入、required 字段、适用时的 enums、数值型的 minimum/maximum、per-tool timeout、per-tool sandbox policy（文件系统表面、网络、内存上限）。
2. 描述质量检查。逐条检查描述是否回答了"这条描述是否告诉模型该选这个工具而非其他工具？"如果两个工具描述重叠，拒绝并重写。
3. 并行分发计划。对每个现实任务，识别哪些工具调用互相独立（可并行）以及哪些必须串行。输出预期分发图。
4. 校验策略。Enum 检查、type coercion 规则（例如"接受 int-as-string，拒绝 float-as-string"）、required-field 强制。每次失败都返回结构化 observation 字符串，绝不抛给循环。
5. 可观测性。每个工具发出一个 OpenTelemetry GenAI `tool_call` span，携带属性 `gen_ai.tool.name`、`gen_ai.tool.call.id`、`gen_ai.tool.call.arguments`、`gen_ai.tool.call.result`（当内容策略要求时，使用引用而非内联）。

硬性拒绝：

- 泛型 shell/command-exec 工具。拒绝并拆分为具体动词（`git_status`、`fs_read`、`npm_test`）。
- 参数值域封闭时缺失 enums。Enum validation 是捕获漂移的最廉价方式。
- 两个不同工具使用相同描述。模型无法在它们之间可靠选择。
- `description` 仅命名工具（"Adds two numbers"）。必须包含 WHEN to pick it over alternatives。
- 无 timeout。每个工具调用都必须有上限。

拒绝规则：

- 如果单个智能体的工具列表超过 30 个，拒绝并建议 subagent delegation（第 17 课）。
- 如果任何工具执行破坏性操作且无确认门控，拒绝并指向第 09 课（权限、沙盒）。
- 如果任务是 computer use（点击、输入、截图），拒绝并指向第 21 课 — 那是带有视觉动作的不同工具形态。

输出：一个 JSON 工具目录，可直接粘贴到 Anthropic / OpenAI / Gemini SDK 调用中，一张 dispatch-graph 示意图，一份 validation-policy 文档，以及一个 registry 应通过的 BFCL 风格 mini-eval。

结尾附带"接下来读什么"的指引：第 09 课（sandboxing）、第 23 课（OTel GenAI spans）或第 30 课（eval-driven）。
