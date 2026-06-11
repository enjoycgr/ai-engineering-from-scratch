# Structured Outputs & Constrained Decoding (结构化输出与约束解码)

> 让 LLM 返回 JSON。大多数时候能得到 JSON。在生产环境中，"大多数时候" 就是问题所在。Constrained decoding (约束解码) 通过在采样前编辑 logits，把 "大多数时候" 变成 "总是"。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 分钟

## The Problem

一个分类器提示 LLM："Return one of {positive, negative, neutral}." 模型返回 "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."。解析器崩溃。分类器的 F1 为 0.0。

Free-form generation (自由形式生成) 不是契约。它只是建议。生产系统需要契约。

2026 年存在三个层次。

1. **Prompting (提示).** 礼貌请求。"Return only the JSON object." 在前沿模型上大约 80% 有效，小模型上更低。
2. **Native structured output APIs (原生结构化输出 API).** OpenAI `response_format`、Anthropic tool use、Gemini JSON mode。在支持的 schema 上可靠。供应商锁定。
3. **Constrained decoding (约束解码).** 在每一步生成中修改 logits，使模型*无法*发出无效 token。按构造保证 100% 有效。适用于任何本地模型。

本课建立对三者的直觉，并指明何时选用哪一个。

## The Concept

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Constrained decoding 的工作原理。** 在每一步生成中，LLM 在整个词表（~100k token）上产生一个 logit (对数几率) 向量。一个 logit processor (对数几率处理器) 位于模型和采样器之间。它根据目标语法 —— JSON Schema、正则表达式、context-free grammar (上下文无关语法) —— 计算当前位置哪些 token 是有效的，并将所有无效 token 的 logits 设为负无穷。对剩余 logits 的 softmax (软最大值) 只将概率质量放在有效的后续 token 上。

2026 年的实现：

- **Outlines.** 将 JSON Schema 或正则表达式编译为 finite-state machine (有限状态机, FSM)。每个 token 获得 O(1) 的 valid-next-token (下一有效 token) 查找。基于 FSM，因此递归 schema 需要展平。
- **XGrammar / llguidance.** Context-free grammar (上下文无关语法) 引擎。处理递归 JSON Schema。解码开销接近零。OpenAI 在 2025 年的结构化输出实现中致谢了 llguidance。
- **vLLM guided decoding.** 内置 `guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`，通过 Outlines、XGrammar 或 lm-format-enforcer 后端。
- **Instructor.** 基于 Pydantic 的 LLM 包装器。在验证失败时重试。跨供应商，但不修改 logits —— 它依赖重试 + 结构化输出感知提示。

### The counterintuitive result

Constrained decoding 通常*比*无约束生成*更快*。两个原因。首先，它缩小了 next-token search space (下一 token 搜索空间)。其次，聪明的实现跳过对强制 token 的生成（如 `{"name": "` 这样的脚手架 —— 每个字节都是确定的）。

### The pitfall that costs you

Field order (字段顺序) 很重要。把 `answer` 放在 `reasoning` 之前，模型在思考前就承诺了答案。JSON 是有效的。答案是错的。没有任何验证能捕获它。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema 字段顺序是逻辑，不是格式。

## Build It

### Step 1: 从零实现正则约束生成

参见 `code/main.py` 中的独立 FSM 实现。核心思想用 30 行表达：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM 跟踪到目前为止已满足语法的哪些部分。`valid_tokens(state, tokenizer)` 计算哪些词汇 token 可以在不离开接受路径的情况下推进 FSM。

### Step 2: 用于 JSON Schema 的 Outlines

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远。FSM 使无效输出不可达。

### Step 3: 用于跨供应商 Pydantic 的 Instructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

不同机制。Instructor 不触碰 logits。它将 schema 格式化进 prompt，解析输出，并在验证失败时重试（默认 3 次）。适用于任何供应商。重试增加延迟和成本。跨供应商可移植性是卖点。

### Step 4: 原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务端 constrained decoding。对支持的 schema 可靠性与 Outlines 相当。无需本地模型管理。将你锁定到供应商。

## Pitfalls

- **Recursive schemas (递归 schema).** Outlines 将递归展平到固定深度。树结构输出（嵌套评论、AST）需要 XGrammar 或 llguidance（基于 CFG）。
- **Huge enums (巨大枚举).** 10,000 选项的枚举编译缓慢或超时。切换到检索器：先预测 top-k 候选，然后约束到这些。
- **Grammar too strict (语法过严).** 强制 `date: "YYYY-MM-DD"` 正则，模型就无法为缺失日期输出 `"unknown"`。模型通过编造日期来补偿。允许 `null` 或哨兵值。
- **Premature commitment (过早承诺).** 参见上面的字段顺序陷阱。始终将 reasoning 放在前面。
- **Vendor JSON mode without schema (无 schema 的供应商 JSON mode).** 纯 JSON mode 只保证有效 JSON，不保证对你的*用例*有效。始终提供完整 schema。

## Use It

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| OpenAI/Anthropic/Google 模型，简单 schema | Native vendor structured output (原生供应商结构化输出) |
| 任何供应商，Pydantic 工作流，可容忍重试 | Instructor |
| 本地模型，需要 100% 有效性，扁平 schema | Outlines (FSM) |
| 本地模型，递归 schema | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM guided decoding |
| 批处理，可接受重试 | Instructor + 最便宜的模型 |

## Ship It

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: 选择结构化输出方法、schema 设计和验证计划。
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

给定用例（供应商、延迟预算、schema 复杂度、失败容忍度），输出：

1. Mechanism (机制). Native vendor structured output、Instructor retries、Outlines FSM 或 XGrammar CFG。一句话理由。
2. Schema design (Schema 设计). 字段顺序（reasoning 在前，answer 在后）、"unknown" 的可空字段、enum vs regex、必填字段。
3. Failure strategy (失败策略). 最大重试次数、回退模型、优雅的 `null` 处理、分布外拒绝。
4. Validation plan (验证计划). Schema compliance rate (目标 100%)、semantic validity (语义有效性, LLM-judge)、字段覆盖率、latency p50/p99。

拒绝任何将 `answer` 或 `decision` 放在 reasoning 字段之前的设计。拒绝在没有 schema 的情况下使用裸 JSON mode。标记仅支持 FSM 的库背后的递归 schema。
```

## Exercises

1. **Easy.** 在没有 constrained decoding 的情况下提示一个小型开放权重模型（例如 Llama-3.2-3B）生成 `Review(sentiment, confidence, evidence_span)`。测量在 100 条评论中有多少比例能解析为有效 JSON。
2. **Medium.** 对同一语料使用 Outlines JSON mode。比较 compliance rate (合规率)、延迟和语义准确率。
3. **Hard.** 从零实现一个正则约束解码器，用于电话号码 (`\d{3}-\d{3}-\d{4}`)。验证在 1000 个样本中 0 个无效输出。

## Key Terms

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Constrained decoding | 强制有效输出 | 在每一步生成中屏蔽无效 token 的 logits。 |
| Logit processor | 约束的东西 | 函数：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示；O(1) 查找下一有效 token。 |
| CFG | 上下文无关语法 | 处理递归的语法；比 FSM 慢但更具表达力。 |
| Schema field order | 重要吗？ | 重要 —— 第一个字段会承诺；始终将 reasoning 放在 answer 之前。 |
| Guided decoding | vLLM 的叫法 | 相同概念，集成到推理服务器中。 |
| JSON mode | OpenAI 的早期版本 | 保证 JSON 语法；不保证 schema 匹配。 |

## Further Reading

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines 论文。
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — 快速基于 CFG 的 constrained decoding。
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成。
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API 参考 + 陷阱。
- [Instructor library](https://python.useinstructor.com/) — 跨供应商的 Pydantic + 重试。
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 对 6 个 constrained decoding 框架的基准测试。
