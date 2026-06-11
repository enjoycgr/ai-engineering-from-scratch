# 工具接口 —— 为什么智能体需要结构化输入输出

> 语言模型生成 token。程序执行动作。这两者之间的鸿沟就是工具接口：一份契约，让模型能够请求动作，让宿主执行它。每一个 2026 年的技术栈——OpenAI、Anthropic 和 Gemini 上的函数调用；MCP 的 `tools/call`；A2A 的任务部件——都是同一个四步循环的不同编码。本课为这个循环命名，并展示运行它所需的最小机械装置。

**类型：** Learn
**语言：** Python（仅标准库，无 LLM）
**前置要求：** Phase 11（LLM 补全 API）
**时间：** ~45 分钟

## 学习目标

- 解释为什么只能生成文本的 LLM 无法独自对现实世界采取行动。
- 画出四步工具调用循环（describe → decide → execute → observe），并说明每一步由谁负责。
- 将工具描述写为三个部分：名称、JSON Schema 输入，以及一个确定性的执行器函数。
- 区分纯工具（pure）和副作用工具（consequential），并说明这种区分对安全为什么重要。

## 问题

LLM 对下一个 token 输出概率分布。这就是它的全部输出表面。如果你问一个聊天模型"班加罗尔现在的天气怎么样"，它可以写出一个看似合理的句子，但它无法拨入天气 API。这个句子可能是巧合正确，也可能是三天前的旧数据。

弥合这一鸿沟就是工具接口的目的。宿主程序——你的智能体运行时、Claude Desktop、ChatGPT、Cursor 或自定义脚本——向模型广告一系列可调用工具。模型在决定需要采取行动时，会发出一个结构化载荷，指明工具及其参数。宿主解析该载荷，实际运行工具，并将结果反馈回去。循环持续进行，直到模型决定不再需要调用。

该契约的第一个版本于 2023 年 6 月以 OpenAI 的 "functions" 参数发布。Anthropic 随后在 Claude 2.1 中推出了 `tool_use` 块。Gemini 几个月后添加了 `functionDeclarations`。每个提供商现在都暴露了相同的形状：JSON-Schema 类型的工具列表输入，JSON 载荷的工具调用输出。Model Context Protocol（2024 年 11 月）将该契约泛化，使一个工具注册表可以服务所有模型。A2A（2026 年 4 月，v1.0）则为智能体到智能体的委托层叠了相同的原语。

四步循环是所有这些不变量。Phase 13 中其他所有内容都是对它的 elaboration（细化）。

## 概念

### 第一步：describe（描述）

宿主用三个字段声明每个工具。

- **Name（名称）。** 一个稳定、机器可读的标识符。`get_weather`，而非 "weather thing"。
- **Description（描述）。** 一段自然语言的简要说明。"当用户询问特定城市的当前状况时使用。不要用于历史数据。"
- **Input schema（输入模式）。** 一个 JSON Schema 对象（draft 2020-12），描述工具的参数。

模型接收到这个列表。现代提供商使用提供商特定的模板将这些声明序列化到系统提示中，因此作为调用者的你只需要处理结构化形式。

### 第二步：decide（决策）

给定用户消息和可用工具，模型选择三种行为之一。

1. **直接以文本回答。** 不调用工具。
2. **调用一个或多个工具。** 发出结构化调用对象。在 `parallel_tool_calls: true` 下（OpenAI 和 Gemini 默认开启，Anthropic 需 opt-in），模型可以在一个 turn 中发出多个调用。
3. **拒绝。** 严格模式的结构化输出可以产生一个类型化的 `refusal` 块，而非调用。

一个工具调用载荷有三个稳定字段：调用 `id`、工具 `name` 和 JSON `arguments` 对象。id 存在的原因是宿主可以将后续结果与特定调用关联起来，这在并行调用以乱序返回时尤为重要。

### 第三步：execute（执行）

宿主接收到调用，根据声明的 schema 验证参数，并运行执行器。无效参数意味着模型 hallucinated（幻觉化）了一个字段或使用了错误的类型——这在弱模型上是一种非常常见的失败模式。生产宿主对无效参数做以下三种之一：快速失败并将错误呈现给模型，用约束解析器修复 JSON，或将验证错误包含在提示中重试模型。

执行器本身是普通代码。Python、TypeScript、shell 命令、数据库查询。它产生一个结果，通常是一个字符串，但可以是任何 JSON 值或结构化内容块（MCP 中的文本、图像或资源引用）。结果必须是可序列化的。

### 第四步：observe（观察）

宿主将工具结果追加到对话中（作为具有匹配 `id` 的 `tool` 角色消息），并重新调用模型。模型现在有了工具输出作为上下文，可以产生最终答案或请求更多调用。这持续进行，直到模型停止发出调用或宿主达到安全限制上的迭代次数。

### 信任划分

工具分为两种对安全有重要影响的类型。

- **Pure（纯工具）。** 只读、确定性、无副作用。`get_weather`、`search_docs`、`get_current_time`。可以安全地投机性调用。
- **Consequential（后果性工具）。** 改变状态、花钱、触碰用户数据。`send_email`、`delete_file`、`execute_trade`。必须经过 gate（门禁）。

Meta 2026 年的智能体安全 "双规则"（Rule of Two）说，单个 turn 最多只能组合以下两项中的两项：不受信任的输入、敏感数据、后果性动作。工具接口就是你执行该规则的地方——通过拒绝调用、要求用户确认或升级权限。完整的安全章节见 Phase 13 · 15，智能体级权限策略见 Phase 14 · 09。

### 循环所在之处

| 上下文 | 谁描述 | 谁决策 | 谁执行 |
|---------|---------------|-------------|--------------|
| 单轮函数调用（OpenAI/Anthropic/Gemini） | 应用开发者 | LLM | 应用开发者 |
| MCP | MCP 服务器 | 通过 MCP 客户端的 LLM | MCP 服务器 |
| A2A | Agent Card 发布者 | 调用智能体 | 被调用智能体 |
| Web 浏览器（函数调用智能体） | 浏览器扩展 / WebMCP | LLM | 浏览器运行时 |

无处不在，相同的四步。列名改变；结构不变。

### 为什么不直接提示模型输出 JSON？

"让模型以 JSON 回复" 是函数调用出现之前的模式。它在 frontier 模型上失败约 5% 到 15%，在小模型上失败率更高。失败模式包括缺失花括号、尾随逗号、幻觉字段和错误类型。然后你需要 JSON 修复通道、重试或约束解码器。

原生函数调用更好，原因有三。第一，提供商端到端地在确切的调用形状上训练模型，因此严格模式下的有效 JSON 率攀升至 98% 到 99%。第二，调用载荷位于自己的协议槽位中，不在自由文本内部——因此工具调用永远不会泄漏到用户可见的回复中。第三，提供商使用约束解码强制执行 schema 合规性（OpenAI 的 strict mode、Anthropic 的 `tool_use`、Gemini 的 `responseSchema`）。输出保证能通过验证。

Phase 13 · 02 将并排讲解三个提供商 API。Phase 13 · 04 深入讲解结构化输出。

### 断路器

当模型停止发出调用或宿主达到最大 turn 数时，循环终止。生产宿主将此设置为 5 到 20 个 turn。超过这个范围，你几乎肯定处于一个模型无法退出的循环中。Claude Code 默认为 20；OpenAI Assistants 为 10；Cursor 的智能体模式为 25。

另一种选择——无界循环——每六个月就会出现一次 "智能体一夜之间在 API 调用上花了 400 美元" 的事后分析。不要在没有边界的情况下发布产品。

Phase 14 · 12 深入讲解错误恢复和自我修复；Phase 17 讲解生产速率限制。

### Phase 13 接下来的方向

- 第 02 到 05 课打磨提供商级别的工具调用表面。
- 第 06 到 14 课将循环泛化为 MCP。
- 第 15 到 18 课防御循环免受敌对服务器、对抗性用户和未经认证的远程认证表面的攻击。
- 第 19 到 22 课将模式扩展到智能体间协作、可观测性、路由和打包。
- 第 23 课使用每个原语交付一个完整的生态系统。

每一节剩余的课都是这个四步循环的 elaboration（细化）。将其作为不变量牢记在心。

## 使用它

`code/main.py` 在没有 LLM 的情况下运行四步循环。一个虚假的 "decider" 函数通过模式匹配用户消息来模拟模型；执行器、schema 验证器和观察步骤的 harness（框架）是真实的。运行它以查看完整的请求/响应编排，以及可打印的中间状态，然后在后续课程中用任何真实的提供商替换虚假 decider。

看点：

- 工具注册表为每个工具持有三个字段：名称、描述、schema 和一个执行器引用。
- 验证器是一个最小的 JSON Schema 子集（类型、required、enum、min/max），仅使用标准库编写。Phase 13 · 04 会提供一个更完整的版本。
- 循环将迭代次数限制在五轮。生产智能体需要的就是这种断路器。

## 交付它

本课产出 `outputs/skill-tool-interface-reviewer.md`。给定一个工具定义草稿（名称 + 描述 + schema + 执行器大纲），该技能对其进行循环适配性审计：名称是否机器稳定，描述是否是完整的使用简介，schema 是否正确使用 JSON Schema 2020-12，以及纯/后果性分类是否明确。

## 练习

1. 在 `code/main.py` 中添加第四个工具 `get_stock_price(ticker)`。将其描述写为 "当用户按 ticker 询问当前股价时使用。不要用于历史价格或市场摘要。" 运行 harness 并确认虚假 decider 将提及 ticker 的查询路由到新工具。

2. 破坏 schema 验证器。传入一个 `arguments` 对象缺失 required 字段的调用，并确认宿主在执行前拒绝它。然后传入一个带有额外未知字段的调用。决定：宿主应该拒绝还是忽略？用安全论证为你的选择辩护。

3. 将 harness 中的每个工具分类为纯工具或后果性工具。向需要它的注册表条目添加 `consequential: true` 标志，并更改循环以在选择后果性工具时打印一行 "would confirm with user"。这是每个生产宿主都需要的确认 gate 的形状。

4. 在纸上画出四步循环，并为你最喜欢的客户端（Claude Desktop、Cursor、ChatGPT 或自定义栈）填写上面的提供商列表格。与 Phase 13 · 06 中的 MCP 特定变体进行交叉参考。

5. 从头到尾阅读 OpenAI 的函数调用指南。找出本课呈现的四步循环中缺失的一个字段。解释它增加了什么，以及为什么它是方便而非本质的。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| Tool（工具） | "模型可以调用的东西" | 名称 + JSON-Schema 类型输入 + 执行器函数的三元组 |
| Function calling（函数调用） | "原生工具使用" | 提供商级别的 API 支持，用于发出结构化工具调用而非散文 |
| Tool call（工具调用） | "模型的行动请求" | 模型发出的、带有 `id`、`name`、`arguments` 的 JSON 载荷 |
| Tool result（工具结果） | "工具返回的东西" | 执行器的输出，包装在具有匹配 id 的 `tool` 角色消息中 |
| Parallel tool calls（并行工具调用） | "一次多个调用" | 一个模型 turn 中的多个调用对象，独立且可按 id 排序 |
| Strict mode（严格模式） | "保证 JSON" | 约束解码，强制模型输出符合声明的 schema |
| Pure tool（纯工具） | "只读工具" | 无副作用；可安全重跑 |
| Consequential tool（后果性工具） | "动作工具" | 改变外部状态；需要 gate、审计或用户确认 |
| Four-step loop（四步循环） | "工具调用周期" | describe → decide → execute → observe |
| Host（宿主） | "智能体运行时" | 持有工具注册表、调用模型并运行执行器的程序 |

## 延伸阅读

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) —— OpenAI 风格工具声明和调用形状的规范参考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) —— Claude 的 `tool_use` / `tool_result` 块格式
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) —— Gemini 中的 `functionDeclarations` 和并行调用语义
- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— 工具接口的提供商无关泛化
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) —— 每个现代工具 API 使用的 schema 方言
