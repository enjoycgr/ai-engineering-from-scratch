# OpenAI Agents SDK: Handoffs, Guardrails, Tracing

> OpenAI Agents SDK 是构建在 Responses API 之上的轻量级多 agent (智能体) 框架。五大原语：Agent (智能体)、Handoff (交接)、Guardrail (护栏)、Session (会话)、Tracing (追踪)。Handoff 是名为 `transfer_to_<agent>` 的工具。Guardrail 在输入或输出时触发拦截。Tracing 默认开启。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 分钟

## Learning Objectives

- 说出 OpenAI Agents SDK 的五大原语。
- 解释 handoff (交接)：为什么它被建模为工具，模型看到的命名形状是什么，以及上下文如何转移。
- 区分 input guardrail (输入护栏)、output guardrail (输出护栏) 和 tool guardrail (工具护栏)；解释 `run_in_parallel` 与阻塞模式的区别。
- 用标准库实现一个支持 handoff (交接) + guardrail (护栏) + span-style tracing (追踪) 的运行时。

## The Problem

无法干净地委托任务的 agent (智能体) 最终会把所有内容塞进一个 prompt (提示词) 里。没有 guardrail (护栏) 的 agent 会泄露 PII (Personally Identifiable Information，个人身份信息)、输出违反策略的内容，或者永远循环。OpenAI 的 SDK 将三个原语编码化，使多 agent 工作变得可管理。

## The Concept

### 五大原语

1. **Agent (智能体)。** LLM (Large Language Model，大语言模型) + instructions (指令) + tools (工具) + handoffs (交接)。
2. **Handoff (交接)。** 委托给另一个 agent (智能体)。对模型表现为一个名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail (护栏)。** 对输入（仅第一个 agent）、输出（仅最后一个 agent）或工具调用（每个函数工具）进行验证。
4. **Session (会话)。** 跨轮次自动保存对话历史。
5. **Tracing (追踪)。** 为 LLM 生成、工具调用、handoff (交接) 和 guardrail (护栏) 内置的 span (跨度)。

### Handoffs 作为工具

模型会在其工具列表中看到 `transfer_to_billing_agent`。调用它会向运行时发出信号：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 版将其折叠）。
2. 用其 instructions (指令) 初始化目标 agent (智能体)。
3. 用目标 agent (智能体) 继续运行。

这就是 supervisor pattern (监督者模式)（Lesson 13 / Lesson 28）的产品化形式。

### Guardrails

三种类型：

- **Input guardrail (输入护栏)。** 在第一个 agent (智能体) 的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **Output guardrail (输出护栏)。** 在最后一个 agent (智能体) 的输出上运行。捕获 PII (个人身份信息) 泄露、策略违规、格式错误的响应。
- **Tool guardrail (工具护栏)。** 在每个函数工具上运行。验证参数、检查权限、审计执行。

模式：

- **Parallel (并行，默认)。** Guardrail (护栏) LLM 与主 LLM 并行运行。尾部延迟更低。如果触发，主 LLM 的工作会被丢弃（浪费 token (令牌)）。
- **Blocking (阻塞， `run_in_parallel=False`)。** Guardrail (护栏) LLM 先运行。如果触发，不会浪费主调用的 token (令牌)。

触发时会抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

默认开启。每次 LLM 生成、工具调用、handoff (交接) 和 guardrail (护栏) 都会发射一个 span (跨度)。`OPENAI_AGENTS_DISABLE_TRACING=1` 可关闭。`add_trace_processor(processor)` 可将 span (跨度) 扇出到你自己的后端，与 OpenAI 的并存。

### Sessions

`Session` 在后端（SQLite、Redis、自定义）存储对话历史。`Runner.run(agent, input, session=session)` 自动加载并追加。

### 这个模式何时会出错

- **Handoff drift (交接漂移)。** Agent A handoff (交接) 给 Agent B，Agent B 又 handoff (交接) 回 Agent A。添加一个 hop counter (跳转计数器)。
- **Guardrail bypass (护栏绕过)。** Tool guardrail (工具护栏) 只在函数工具上触发；内置工具（文件读取器、网页获取）需要单独的策略。
- **Over-tracing (过度追踪)。** Span (跨度) 中包含敏感内容。结合 OTel GenAI content-capture 规则（Lesson 23）——外部存储，按 ID 引用。

## Build It

`code/main.py` 用标准库实现了 SDK 的形态：

- `Agent`、`FunctionTool`、`Handoff`（作为具有 transfer 语义的函数工具）。
- 带 input/output/tool guardrail (护栏)、handoff (交接) 分发和 hop counter (跳转计数器) 的 `Runner`。
- 一个简单的 span (跨度) 发射器来展示追踪结构。
- 一个分流 agent (智能体)，根据用户查询 handoff (交接) 给 billing (计费) 或 support (支持)；guardrail (护栏) 在一个输入上触发。

运行方式：

```
python3 code/main.py
```

追踪显示两次成功的 handoff (交接)、一次 input guardrail (输入护栏) 触发，以及一个 span tree (跨度树)，镜像真实 SDK 发射的内容。

## Use It

- 对于 OpenAI 优先的产品，使用 **OpenAI Agents SDK**。
- 对于 Claude 优先的产品，使用 **Claude Agent SDK**（Lesson 17）。
- 当你需要显式状态和持久恢复时，使用 **LangGraph**（Lesson 13）。
- 当你需要精确控制（语音、多提供商、联邦部署）时，使用 **Custom**（自定义）。

## Ship It

`outputs/skill-agents-sdk-scaffold.md` 搭建了一个 Agents SDK 应用，包含分流 agent (智能体)、handoff (交接)、input/output/tool guardrail (输入/输出/工具护栏)、session store (会话存储) 和 trace processor (追踪处理器)。

## Exercises

1. 添加一个 handoff hop counter (交接跳转计数器)：在 N 次转移后拒绝。追踪该行为。
2. 将 `nest_handoff_history` 实现为一个选项——在 transfer 之前将先前的消息折叠成一个摘要。
3. 编写一个阻塞 output guardrail (输出护栏)。在会触发它的 prompt (提示词) 与不会触发的 prompt (提示词) 上比较延迟。
4. 将 `add_trace_processor` 连接到 JSON logger。每个 span (跨度) 发射什么形状？
5. 阅读 SDK 文档。将你的标准库玩具移植到 `openai-agents-python`。你建模错了什么？

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Agent (智能体) | "LLM + instructions (指令)" | SDK 中的 Agent (智能体) 类型；拥有工具和 handoff (交接) |
| Handoff (交接) | "Transfer (转移)" | 模型调用来委托给另一个 agent (智能体) 的工具 |
| Guardrail (护栏) | "Policy check (策略检查)" | 对输入 / 输出 / 工具调用的验证 |
| Tripwire (触发器) | "Guardrail trip (护栏触发)" | Guardrail (护栏) 拒绝时抛出的异常 |
| Session (会话) | "History store (历史存储)" | 在多次运行之间持久保存的对话记忆 |
| Tracing (追踪) | "Spans (跨度)" | 覆盖 LLM + 工具 + handoff (交接) + guardrail (护栏) 的内置可观测性 |
| Blocking guardrail (阻塞护栏) | "Sequential check (顺序检查)" | Guardrail (护栏) 先运行；触发时不浪费 token (令牌) |
| Parallel guardrail (并行护栏) | "Concurrent check (并发检查)" | Guardrail (护栏) 与主调用并行运行；延迟更低，触发时浪费 token (令牌) |

## Further Reading

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 原语、handoff (交接)、guardrail (护栏)、tracing (追踪)
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 风格的对标产品
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 何时该使用 handoff (交接)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Agents SDK span (跨度) 映射到的标准
