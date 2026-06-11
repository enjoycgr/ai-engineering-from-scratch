---
name: provider-portability-audit
description: 审计一个提供商上的函数调用集成，列出移植到另外两个提供商时会破坏的每个字段重命名、行为差异和硬限制冲突。
version: 1.0.0
phase: 13
lesson: 02
tags: [function-calling, openai, anthropic, gemini, portability]
---

给定一个提供商（OpenAI、Anthropic 或 Gemini）上的函数调用集成，产出一份可移植性审计，列出当相同逻辑部署到另外两个提供商时出现的每个字段重命名、行为差异和硬限制冲突。

产出：

1. 声明差异。对于集成中的每个工具，展示信封 / 字段重命名 / schema 翻译到另外两个提供商所需的内容。标记目标提供商不支持的任何 JSON Schema 构造（Gemini：OpenAPI 3.0 子集；OpenAI 严格模式：无 `$ref`，无歧义 `oneOf`）。
2. 响应差异。记录工具调用在每个提供商响应形状中的位置（`tool_calls[]` vs `content[]` 块 vs `parts[]` 条目）以及谁负责解析 `arguments`（OpenAI 上是字符串，Anthropic 和 Gemini 上是对象）。
3. `tool_choice` 差异。将集成的当前选择设置（auto / forbid / force / required）映射到目标提供商形状；标记缺失的模式。
4. 限制冲突。报告工具数量（128 / 64 / 64）、schema 深度（5 / 10 / 实际上无限制）和每个参数长度上限。在任何集成超过目标提供商限制时提升 block 严重级别。
5. 严格模式映射。说明严格模式语义是否在目标上保留。OpenAI `strict: true` 在 Anthropic 上没有精确等价物；Gemini `responseSchema` 近似但在请求级别。

硬拒绝项：
- 任何假设 `arguments` 在非 OpenAI 目标上是字符串的集成。将静默产生错误结果。
- 任何工具数量超过 64 且未通过路由器移植到 Anthropic 或 Gemini 的集成。
- 任何在目标为 OpenAI 严格模式时在 schema 中使用 `$ref` 的集成。

拒绝规则：
- 如果被要求移植依赖无类似物提供商特定功能的集成（例如 OpenAI Responses API 有状态 turns、Anthropic computer-use 块），拒绝并解释哪个功能没有目标等价物。
- 如果被要求选出赢家，拒绝。选择取决于宿主的严格模式需求、成本概况和并行调用需求。

输出：一份一页审计，包含每个工具的差异表、限制表和每个目标提供商的最终 "移植裁决"（ship / needs-router / blocked-by-feature）。以一句命名最高杠杆迁移变更的话结束。
