# 并行工具调用与工具流式传输

> 三个独立的天气查询串行化就是三次往返。并行运行它们，总时间坍缩到最慢的单次调用。每个前沿提供商现在都在单个 turn 中发出多个工具调用。回报是真实的；管道是微妙的。本课讲解两个半部分：并行扇出和流式参数重组，重点强调 id 关联陷阱。

**类型：** Build
**语言：** Python（stdlib，线程池 + 流式 harness）
**前置要求：** Phase 13 · 02（函数调用深度解析）
**时间：** ~75 分钟

## 学习目标

- 解释 `parallel_tool_calls: true` 存在的原因以及何时禁用它。
- 在并行扇出期间，将流式参数块关联到正确的工具调用 id。
- 将部分 `arguments` 字符串重组为完整 JSON，而不提前解析。
- 运行一个三城市天气基准测试，演示串行 vs 并行延迟。

## 问题

没有并行调用时，回答 "班加罗尔、东京和苏黎世的天气怎么样" 的智能体会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> 运行执行器，回复结果
LLM -> call get_weather(Tokyo)
host -> 运行执行器，回复结果
LLM -> call get_weather(Zurich)
host -> 运行执行器，回复结果
LLM -> 最终文本答案
```

三次 LLM 往返，每次还要支付执行器延迟。大约是理想挂钟时间的 4 倍。

使用并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> 并发运行三个执行器，回复三个结果
LLM -> 最终文本答案
```

一次 LLM 往返。执行器时间是三个中的最大值，而非总和。OpenAI、Anthropic 和 Gemini 的生产基准测试显示，扇出工作负载的挂钟时间减少了 60% 到 70%。

代价是关联复杂性。当三个调用以乱序完成时，你的结果必须携带匹配的 `tool_call_id`，以便模型能对齐它们。当结果流式传输时，你必须在執行前将部分参数片段组装成完整 JSON。Gemini 3 添加唯一 id 部分是为了解决一个真实世界问题，即对同一工具的两次并行调用无法区分。

## 概念

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设为 `false` 以强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 实现并行（Claude 3.5 及以上默认）。设为 `true` 以强制串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型决定。

当工具存在排序依赖（先 `create_file` 再 `write_file`）、一个调用的输出告知另一个调用的输入，或速率限制器无法处理扇出时，禁用并行。

### Id 关联

模型发出的每次调用都有一个 `id`。宿主返回的每个结果都必须包含相同的 id。没有它，结果就ambiguous（模棱两可）。

- **OpenAI。** 每个工具角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 按名称匹配，这破坏了同名并行调用）。

### 并发运行调用

宿主在自己的线程、协程或远程工作者上运行每次调用的执行器。最简单的 harness 使用线程池；生产环境使用 asyncio 配合 `asyncio.gather` 或结构化并发。完成顺序不可预测——id 是标识符。

一个常见错误：按调用列表顺序回复结果而非完成顺序。这通常能工作，因为模型只关心 `tool_call_id`，但如果结果丢失或重复，乱序提交会使调试更难。优先按完成顺序以显式 id 回复。

### 流式工具调用

当模型流式传输时，`arguments` 分片到达。三条独立的流式调用块在电线上交错。你需要每个 id 一个累加器。

按提供商的形状：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（调用列表中的位置）。你按索引累积，在首次出现时读取 `id`，并在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流式事件是 `message_start`，然后每个块一个 `content_block_start`，类型为 `tool_use`（包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）发出带有 `functionCallId` 的块，因此调用可以干净地交错。Gemini 3 之前，流式传输一次返回一个完整调用。

### 部分 JSON 与提前解析陷阱

你不能在 `arguments` 完成前解析它。部分 JSON 如 `{"city": "Beng` 无效且会引发异常。正确的 gate 是提供商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`、或 Gemini 的流结束事件。只有那时才尝试 `json.loads`。更稳健的方法使用增量 JSON 解析器，在结构完成时产出事件；OpenAI 的流式指南推荐此方法，用于显示实时 "thinking" 指示器的 UX。花括号计数作为完整性测试不可靠（引号字符串内或转义内容中的花括号会导致误报），只应作为非正式调试启发式方法使用。

### 乱序完成

```
call_A: 快速 API，第一个返回
call_B: 慢速 API，第二个返回
call_C: 中等 API，第三个返回
```

宿主回复仍必须引用 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

回复中的顺序在 OpenAI 或 Anthropic 上不影响正确性。Gemini 接受任何顺序，只要 id 匹配。

### 基准测试：串行 vs 并行

`code/main.py` 中的 harness 模拟三个执行器，延迟分别为 400、600 和 800 毫秒。串行运行总耗时 1800 毫秒。并行运行耗时 max(400, 600, 800) = 800 毫秒。差异是恒定的，而非成比例的，因此随着工具数量增加，节省也会增加。

现实世界注意事项：并行调用对下游 API 造成压力。对一个速率受限的服务进行 10 路扇出将会失败。Phase 13 · 17 讲解网关级背压；重试语义计划在未来阶段中介绍。

### 流式扇出挂钟时间

如果模型本身流式传输，你可以在一个调用的参数完成后立即开始执行，而不是等待所有调用最终化。这是 OpenAI 文档记录的优化，但并非所有 SDK 都暴露。本课的 harness 正是这样做的：一旦模拟流产出完整的参数对象，宿主就启动该调用。

## 使用它

`code/main.py` 有两个半部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印挂钟时间。第二部分重放一个虚假的流式响应——三个并行调用的 `arguments` 块在一条流上交错——并用 `StreamAccumulator` 按 id 重组它们。无 LLM，无网络，只有重组逻辑。

看点：

- 串行计时器命中 1.8 秒。在相同的虚假延迟下，并行计时器命中 0.8 秒。
- 累加器通过按 id 缓冲并仅在每次调用的 JSON 完成时才解析，来处理乱序到达的块。
- 执行器在一个 id 的参数最终化后立即启动，而非在所有流结束后。

## 交付它

本课产出 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该技能审计哪些工具可以安全并行化、哪些有排序依赖、哪些会压垮下游速率限制——返回一个带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py` 并改变模拟延迟。确认并行与串行的比率约为 `max/sum`（实际运行会偏离理想值，因为线程调度、序列化和 harness 开销）。在什么延迟分布下，并行不再重要？

2. 扩展累加器以处理 "调用在流式传输中途被取消" 的情况，通过丢弃其缓冲区并发出 `cancelled` 事件。哪个提供商明确记录了这种情况？检查 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。你应该在异步上看到小幅度胜利，因为上下文切换成本更低，但前提是有实际 I/O 的执行器。

4. 选择两个不应并行化的工具（例如 `create_file` 然后 `write_file`）。向注册表添加一个 `ordering_dependency` 图，并根据该图 gate 并行扇出。这是依赖感知调度的最小机械装置，未来的智能体工程阶段将对其进行形式化。

5. 阅读 OpenAI 的并行函数调用部分和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 推荐禁用并行化的一个真实世界工具类型。（提示：对同一资源的后果性变更。）

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| Parallel tool calls（并行工具调用） | "一轮中的扇出" | 模型在单个助手消息中发出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用发出 |
| `disable_parallel_tool_use` | "Anthropic 的反义标志" | 退出标志；默认启用并行 |
| Tool call id（工具调用 id） | "关联句柄" | 结果消息必须回显的每次调用标识符 |
| Accumulator（累加器） | "流式缓冲区" | 用于部分 `arguments` 块的每个 id 字符串缓冲区 |
| Out-of-order completion（乱序完成） | "最快优先" | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| Dependency graph（依赖图） | "排序约束" | 输出馈入其他工具输入的工具；不能并行化 |
| Parse-early trap（提前解析陷阱） | "JSON.parse 爆炸" | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 功能" | 每次调用带唯一 id 的流式参数块 |
| Completion-order reply（按完成顺序回复） | "不等全部" | 按到达顺序以 id 为键回复结果 |

## 延伸阅读

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) —— 默认行为和退出标志
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) —— `disable_parallel_tool_use` 和结果批处理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) —— Gemini 3 的 id 关联并行调用
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) —— OpenAI 流的块参数重组
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) —— 带 `input_json_delta` 的 `content_block_delta`
