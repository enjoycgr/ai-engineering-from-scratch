# 工具使用与函数调用

> Toolformer (Schick 等, 2023) 开创了自监督工具标注 (self-supervised tool annotation)。Berkeley Function Calling Leaderboard V4 (Patil 等, 2025) 设定了 2026 年的基准：40% agentic (智能体轨迹)、30% multi-turn (多轮交互)、10% live (真实场景)、10% non-live (非真实场景)、10% hallucination (幻觉检测)。单轮调用 (single-turn) 已基本解决。记忆、动态决策和长程工具链仍是未解难题。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 分钟

## 学习目标

- 解释 Toolformer 的自监督训练信号：仅当工具执行能降低 next-token loss (下一词元损失) 时才保留工具标注。
- 说出 BFCL V4 (Berkeley Function Calling Leaderboard V4) 的五个评估类别及其衡量目标。
- 用标准库实现一个生产级 tool registry (工具注册表)，包含 schema validation (模式校验)、argument coercion (参数强制转换) 和 execution sandboxing (执行沙盒)。
- 诊断三个 2026 年仍未解决的难题：long-horizon tool chaining (长程工具链)、dynamic decision-making (动态决策) 和 memory (记忆)。

## 问题背景

早期的工具使用只问：模型能否预测一次正确的 function calling (函数调用)？现代工具使用问的是：模型能否在 40 步内串联工具，并携带记忆，面对部分可观测状态，从工具失败中恢复，且不对不存在的工具产生幻觉 (hallucination)？

Toolformer 奠定了基线：模型可以通过自监督学习何时调用工具。BFCL V4 定义了 2026 年的评估目标。二者之间的差距，就是生产级智能体所处的空间。

## 核心概念

### Toolformer (Schick 等, NeurIPS 2023)

核心思想：让模型在自己的预训练语料上标注候选 API 调用。对每个候选，执行它。仅当包含工具结果能降低 surrounding text 的 next-token loss (下一词元损失) 时，才保留该标注。在过滤后的语料上微调。

涵盖工具：calculator (计算器)、QA system (问答系统)、search engines (搜索引擎)、translator (翻译器)、calendar (日历)。自监督信号 purely about whether the tool helps predict text — 无需人工标签。

规模结果：tool use (工具使用) 随规模涌现。小模型被工具标注拖累；大模型受益。这就是 2026 年 frontier models (前沿模型) 内建强工具使用能力，而大多数 7B 模型需要显式 tool-use fine-tuning (工具使用微调) 才能可靠的原因。

### Berkeley Function Calling Leaderboard V4 (Patil 等, ICML 2025)

BFCL 是 2026 年事实上的评估标准。V4 构成：

- **Agentic (40%)** — 完整智能体轨迹：记忆、多轮、动态决策。
- **Multi-Turn (30%)** — 交互式对话中的工具链。
- **Live (10%)** — 用户提交的真实 prompt（更难分布）。
- **Non-Live (10%)** — 合成测试用例。
- **Hallucination (10%)** — 检测何时不应调用工具。

V3 引入了 state-based evaluation (基于状态的评估)：在一串工具调用之后，检查 API 的实际状态（例如“文件是否已创建？”），而非匹配工具调用的 AST。
V4 增加了 web search (网页搜索)、memory (记忆) 和 format sensitivity (格式敏感性) 类别。

2026 年的关键发现：single-turn function calling (单轮函数调用) 已接近解决。失败集中在 memory (跨轮次携带上下文)、dynamic decision-making (基于先前结果选择工具)、long-horizon chains (20 步以上后的漂移) 和 hallucination detection (无适配工具时拒绝调用)。

### Tool schema (工具模式)

每个 provider (供应商) 都有自己的 tool schema (工具模式)。细节不同，但形状相同：

```
name: string
description: string (功能说明、何时使用)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。二者都接受 JSON Schema。
Descriptions are load-bearing — 模型通过它们来选择正确的工具。糟糕的工具描述是 wrong-tool-picked (选错工具) 失败的第一大根因。

### Argument validation (参数验证)

不要信任任何工具调用。需验证：

1. **Type coercion (类型强制转换)。** 模型可能在 schema 要求 int 的地方返回字符串 `"5"`。如果无歧义则强制转换；否则拒绝。
2. **Enum validation (枚举验证)。** 如果 schema 规定 `status in {"open", "closed"}`，而模型输出了 `"in_progress"`，则返回描述性错误并拒绝。
3. **Required fields (必填字段)。** 缺失必填字段 -> 立即返回结构化 observation (观察) 给模型，而不是崩溃。
4. **Format validation (格式验证)。** 日期、邮箱、URL — 用具体解析器验证，而非正则表达式。

每次验证失败都应返回结构化 observation，以便模型用正确形状重试。

### Parallel tool calls (并行工具调用)

现代 provider 支持在一次 assistant turn 中并行调用多个工具。流程：

1. 模型发出 3 个带不同 `tool_use_id` 的工具调用。
2. Runtime 执行它们（如果独立则并行）。
3. 每个结果通过 `tool_use_id` 关联作为 `tool_result` 块返回。

工程原则：将 correlation ID (关联标识符) 视为 load-bearing。交换它们会导致工具结果路由到错误的调用。

### Sandboxing (沙盒隔离)

Tool execution (工具执行) 是沙盒边界。详见第 09 课。简而言之：每个工具都应指定 read/write surface (读写表面)、network access (网络访问)、timeout (超时)、memory cap (内存上限)。泛型的 `run_shell(cmd)` 是 red flag (危险信号)；具体的 `git_status()` 更安全。

## 动手实现

`code/main.py` 实现了一个生产级 tool registry (工具注册表)：

- JSON Schema 子集校验器（仅标准库）。
- 工具注册，附带 description、input schema、timeout 和 executor。
- 参数强制转换与枚举验证。
- 带 correlation ID 的并行工具分发。
- 错误 observation 以结构化字符串形式返回。

运行：

```
python3 code/main.py
```

trace 展示了一个迷你智能体在一轮内调用三个工具，其中一个故意格式错误，被附带描述性错误拒绝，模型可据此采取行动。

## 如何使用

每个 provider 都有自己的 tool schema — Anthropic、OpenAI、Gemini、Bedrock。如需跨 provider，使用翻译层（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是参考基准 — 如果工具使用是产品核心，上线前先用它测试你的智能体。

## 产物输出

`outputs/skill-tool-registry.md` 为给定任务域生成工具目录、模式和注册表。包含 description quality check (描述质量检查)：每个工具的描述是否告诉模型何时使用它？

## 练习

1. 添加一个 "no-op" 工具，让模型显式拒绝使用任何其他工具。在类 BFCL 的 hallucination (幻觉) 测试上测量。
2. 实现 int-as-string 和 float-as-string 的参数强制转换。强制转换从何时开始掩盖真正的 bug？
3. 为每个工具添加超时和断路器（连续 3 次失败后 60 秒内拒绝该工具）。这会如何改变模型的恢复行为？
4. 阅读 BFCL V4 描述。挑选一个类别（例如 "multi-turn"），让智能体跑 10 个示例 prompt。报告通过率。
5. 将标准库校验器移植到 Pydantic 或 Zod。Pydantic/Zod 捕获了哪些玩具实现遗漏的问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Function calling | "Tool use" | 带校验 schema 的结构化输出工具调用 |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 — 仅当工具调用结果降低 next-token loss 时才保留 |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 基准：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | "Function signature for the model" | name、description、JSON Schema 参数 |
| tool_use_id | "Correlation ID" | 将工具调用与结果关联；并行分发必需 |
| Hallucination detection | "Know when not to call" | V4 类别：无适配工具时拒绝调用 |
| Argument coercion | "String-to-int repair" | 对可预测 schema 不匹配做窄修复；有歧义则拒绝 |
| Sandboxing | "Tool execution boundary" | 每个工具的读写表面、网络、超时、内存上限 |

## 延伸阅读

- [Schick 等, Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) — 自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) — 2026 评估基准
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Agent SDK 中的生产级工具模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — function tool type 和 Guardrails
