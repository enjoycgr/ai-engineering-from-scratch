# 结构化输出 —— JSON Schema、Pydantic、Zod、约束解码

> "礼貌地请求模型返回 JSON" 即使在前沿模型上也有 5% 到 15% 的失败率。结构化输出通过约束解码弥合这一差距：模型在字面上被阻止发出会违反 schema 的 token。OpenAI 的严格模式、Anthropic 的 schema 类型化工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一理念的五种表面形式。本课构建 schema 验证器和严格模式契约，学习者将在每个生产提取管道中使用它们。

**类型：** Build
**语言：** Python（stdlib，JSON Schema 2020-12 子集）
**前置要求：** Phase 13 · 02（函数调用深度解析）
**时间：** ~75 分钟

## 学习目标

- 使用正确的约束（enum、min/max、required、pattern）为提取目标编写 JSON Schema 2020-12。
- 解释为什么严格模式和约束解码给出的保证与 "生成后验证" 不同。
- 区分三种失败模式：解析错误、schema 违规、模型拒绝。
- 交付一个带有类型化修复和类型化拒绝处理的提取管道。

## 问题

读取采购订单邮件的智能体需要将自由文本转换为 `{customer, line_items, total_usd}`。三种方法。

**方法一：提示要求 JSON。** "用 customer、line_items、total_usd 字段回复 JSON。" 在前沿模型上 85% 到 95% 有效。以六种方式失败：缺失花括号、尾随逗号、错误类型、幻觉字段、token 限制处截断、泄漏散文如 "Here is your JSON:"。

**方法二：生成后验证。** 自由生成，解析，针对 schema 验证，失败时重试。可靠但昂贵——每次重试都要付费，截断 bug 每次出现都要额外花费一个 turn。

**方法三：约束解码。** 提供商在解码时强制执行 schema。无效 token 被从采样分布中屏蔽掉。输出保证能解析并保证能通过验证。失败坍缩为一种模式：拒绝（模型决定输入不符合 schema）。

每个 2026 前沿提供商都提供了方法三的某种形式。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}` 加上模型拒绝时响应中的 `refusal`。
- **Anthropic。** `tool_use` 输入上的 schema 强制执行；`stop_reason: "refusal"` 不存在，但 `end_turn` 且没有工具调用是信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 为选定类型推出了 token 级语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 发出一个结构化 `RunResult`，类型化为 `InvoiceModel`。
- **Zod (TypeScript)。** 运行时解析器，将提供商输出针对 Zod schema 进行验证；与 OpenAI 的 `beta.chat.completions.parse` 配对。

共同线索：声明一次 schema，端到端强制执行。

## 概念

### JSON Schema 2020-12 —— 通用语言

每个提供商都接受 JSON Schema 2020-12。你最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子 schema 的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的闭集。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子 schema。
- `additionalProperties`：`false` 禁止额外字段（默认值因模式而异）。

OpenAI 严格模式增加三个要求：每个属性必须列在 `required` 中，无处不在的 `additionalProperties: false`，以及无未解析的 `$ref`。如果你违反这些，API 在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从数据类形状的模型生成 JSON Schema。Pydantic AI 包装了这一点，所以你写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架将 schema 转换为 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出以类型化的 `Invoice` 实例返回。验证错误引发带有类型化错误路径的 `ValidationError`。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TS 等价物。OpenAI 的 Node SDK 暴露 `zodResponseFormat(Invoice)`，它转换为 API 的 JSON Schema 载荷。

### 拒绝

严格模式无法强制模型回答。如果输入无法匹配 schema（"邮件是一首诗，而非发票"），模型发出一个包含原因的 `refusal` 字段。你的代码必须将其作为一等结果处理，而非失败。拒绝作为安全信号也很有用：被要求在受保护内容邮件中提取信用卡号的模型返回带有安全原因附着的拒绝。

### 开放领域中的约束解码

开放权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从 schema 构建确定性有限自动机；每一步，屏蔽会违反 FSM 的 token 的 logits。
2. **带 JSON 解析器的 logit 掩码**：与模型锁步运行流式 JSON 解析器；每一步，计算有效的下一个 token 集合。
3. **带验证器的投机解码**：廉价草稿模型提议 token，验证器强制执行 schema。

商业提供商在幕后选择其中之一。2026 年最先进的状态是，对于短结构化输出比纯生成更快，对于长输出大致相同速度。

### 三种失败模式

1. **解析错误。** 输出不是有效 JSON。在严格模式下不可能发生。在非严格提供商上仍可能发生。
2. **Schema 违规。** 输出能解析但违反 schema。在严格模式下不可能发生。在外部很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你处于严格模式之外（Anthropic 工具使用、非严格 OpenAI、旧版 Gemini）时，恢复模式是：

```
生成 -> 解析 -> 验证 -> 如果失败，注入错误并重试，最多 3 次
```

一次重试通常足够。三次重试捕获弱模型上的偶发错误。超过三次是 schema 不好的迹象：模型无法满足某些输入的 schema，提示或 schema 需要修复。

### 小模型支持

约束解码在小模型上有效。一个带语法强制执行的 3B 参数开放模型，在结构化任务上胜过带原始提示的 70B 参数模型。这是结构化输出对生产至关重要的主要原因：它将可靠性与模型大小解耦。

## 使用它

`code/main.py` 交付了一个 stdlib 中的最小 JSON Schema 2020-12 验证器（类型、required、enum、min/max、pattern、items、additionalProperties）。它包装了一个 `Invoice` schema 并通过验证器运行一个虚假的 LLM 输出，演示解析错误、schema 违规和拒绝路径。在生产中，将虚假输出替换为任何提供商的真实响应。

看点：

- 验证器返回带有 path 和 message 的类型化 `[ValidationError]` 列表。这就是你希望呈现给重试提示的形状。
- 拒绝分支不retry（重试）。它记录并返回一个类型化拒绝。Phase 14 · 09 将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示严格模式为什么能关上幻觉字段的门。

## 交付它

本课产出 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该技能产出严格模式兼容的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，附带类型化拒绝和重试处理 stub。

## 练习

1. 运行 `code/main.py`。添加第四个测试用例，其 `total_usd` 为负数。确认验证器以 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带 discriminator（鉴别器）的 `oneOf`。常见情况：`line_item` 是产品或服务，由 `kind` 标记。严格模式在这里有微妙规则；检查 OpenAI 的结构化输出指南。

3. 将相同的 Invoice schema 写成 Pydantic BaseModel 并比较 `model_json_schema()` 输出与你手写的 schema。找出 Pydantic 默认设置但手写版本省略的一个字段。

4. 测量拒绝率。构造十个不应可提取的输入（一首歌歌词、一个数学证明、一封空白邮件）并通过严格模式的实际提供商运行它们。统计拒绝 vs 幻觉输出。这是你拒绝感知重试的 ground truth（真实基准）。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它在严格模式下明确禁止的一个构造，而该构造在普通 JSON Schema 中是允许的。然后设计一个非本质性地使用该禁止构造的 schema，并将其重构为严格兼容。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "Schema 规范" | 每个现代提供商都使用的 IETF 草案 schema 方言 |
| Strict mode（严格模式） | "保证 schema" | OpenAI 标志，通过约束解码强制执行 schema |
| Constrained decoding（约束解码） | "Logit 掩码" | 解码时强制执行，屏蔽无效的下一个 token |
| Refusal（拒绝） | "模型拒绝" | 当输入无法匹配 schema 时的类型化结果 |
| Parse error（解析错误） | "无效 JSON" | 输出未解析为 JSON；严格模式下不可能 |
| Schema violation（schema 违规） | "形状错误" | 已解析但违反了类型 / required / enum / 范围 |
| `additionalProperties: false` | "不允许额外字段" | 禁止未知字段；OpenAI 严格模式必需 |
| Pydantic BaseModel | "类型化输出" | 发出并验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于提供商输出验证的 TS 运行时 schema |
| Grammar enforcement（语法强制执行） | "开放权重约束解码" | 基于 FSM 的 logit 掩码，如 outlines / guidance |

## 延伸阅读

- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) —— 严格模式、拒绝和 schema 要求
- [OpenAI — Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) —— 2024 年 8 月发布文章，解释解码保证
- [Pydantic AI — Output](https://ai.pydantic.dev/output/) —— 序列化到每个提供商的 typed output_type 绑定
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) —— 规范 spec
- [Microsoft — Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) —— 企业部署说明和严格模式注意事项
