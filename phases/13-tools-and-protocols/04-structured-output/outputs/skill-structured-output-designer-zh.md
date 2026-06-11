---
name: structured-output-designer
description: 为自由文本提取目标设计严格模式兼容的 JSON Schema 加 Pydantic 模型，附带类型化拒绝和重试处理 stub。
version: 1.0.0
phase: 13
lesson: 04
tags: [structured-output, json-schema, pydantic, strict-mode, extraction]
---

给定一个自由文本提取目标（发票、简历、支持工单、研究摘要），产出一份生产就绪的提取契约：JSON Schema 2020-12、Pydantic 模型、拒绝处理器和重试策略。

产出：

1. JSON Schema 2020-12。每个属性都有类型。`required` 列出每个属性。每个对象上都有 `additionalProperties: false`。闭值集合使用 enum。无 `$ref`。无歧义 `oneOf` / `anyOf`。针对 OpenAI 严格模式要求验证。
2. Pydantic v2 BaseModel。用 Python 类型镜像 schema。`model_json_schema()` 必须产出与 (1) 等价的 schema。
3. 拒绝处理器。类型化 `Refusal(reason: str, category: str)` 结果。列出类别：`safety`、`input_mismatch`、`insufficient_info`。
4. 重试策略。三种重试形状：(a) 注入验证错误并重试一次（严格模式之外）；(b) 接受拒绝为最终结果（严格模式）；(c) 重复拒绝时升级到更强的模型。
5. 测试向量。十个输入，覆盖快乐路径、对抗性字段、部分输入和拒绝触发案例。每个都有预期结果。

硬拒绝项：
- 任何含未类型化字段的 schema。严格模式和验证器都会失败。
- 任何缺少 `additionalProperties: false` 的 schema。泄漏幻觉。
- 任何使用无 discriminator 字段的 `oneOf` 的 schema。歧义解码。
- 任何未检查其 JSON Schema 往返的 Pydantic 模型。

拒绝规则：
- 如果目标域包含个人身份数据且没有记录目的，拒绝并路由到 Phase 18（伦理）进行合法依据论证。
- 如果用户要求一个无法用 JSON Schema 2020-12 表达的 schema（例如递归任意图），拒绝并提出最接近的可表达松弛。
- 如果提取目标是 "从任何东西中提取结构化数据"，拒绝并要求具体域。

输出：一份一页契约，包含 schema JSON、Pydantic 类、拒绝和重试策略以及十个测试向量。以关于首先定位哪个提供商以及为什么的说明结束。
