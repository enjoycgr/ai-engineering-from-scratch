# Prompt Caching 和 Context Caching

> 你的 system prompt 有 4,000 tokens。你的 RAG 上下文有 20,000 tokens。你每次请求都发送两者。你也每次都要为两者付费。Prompt caching 让提供商在服务端保留该前缀，复用时只按正常费率的 10% 计费。正确使用可将推理成本降低 50–90%，首 token 延迟降低 40–85%。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 11 · 01（Prompt Engineering），Phase 11 · 05（Context Engineering），Phase 11 · 11（Caching and Cost）
**时间：** ~60 分钟

## 问题

一个 coding agent 在对话的每一轮都向 Claude 发送相同的 15,000-token system prompt。20 轮对话，按 $3/M input tokens 计算，仅 input 成本就达 $0.90 —— 还没算用户的实际消息。乘以 10,000 次日度对话，账单达到 $9,000/天，而这些文本从未改变。

你无法在不损害质量的前提下缩短 prompt。你也无法避免发送它 —— 模型每一轮都需要它。唯一的办法是停止为提供商已经见过的前缀支付全价。

这个办法就是 prompt caching。Anthropic 于 2024 年 8 月推出（2025 年推出了 1 小时延长 TTL 的变体），OpenAI 同年晚些时候实现了自动化，Google 在 Gemini 1.5 旁边推出了显式的 context caching，现在三家都在其 frontier model 上将其作为一等功能提供。

## 概念

![Prompt caching：一次写入，廉价读取](../assets/prompt-caching.svg)

**机制。** 当请求的前缀与最近请求的前缀匹配时，提供商从之前的运行中提供 KV-cache，而不是重新编码 tokens。你首次支付少量写入溢价，之后每次享受大幅读取折扣。

**2026 年的三种提供商风格。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在 content blocks 上显式放置 `cache_control` 标记 | input 费用 90% off | 25% 附加费 | 5 分钟（可延长至 1 小时） | 1,024 tokens（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | input 费用 50% off | 无 | 最长 1 小时（尽力而为） | 1,024 tokens |
| Google (Gemini) | 显式 `CachedContent` API | 按存储计费；读取约正常费率的 25% | 按 token·小时 收取存储费 | 用户设置（默认 1 小时） | 4,096 tokens（Flash），32,768（Pro） |

**不变量。** 三家都只缓存前缀。如果请求之间的任何 token 不同，从第一个不同 token 开始的所有内容都是 miss。将*稳定*部分放在顶部，*可变*部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反顺序 —— 将 user message 放在 system prompt 之上，或在 few-shots 之间穿插动态检索 —— 缓存永远不会命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着缓存块至少需要被读取两次才能净省钱。1 次写入 + 1 次读取平均每次请求成本为 0.675x（节省 32%）；1 次写入 + 10 次读取平均为 0.205x（节省 80%）。经验法则：缓存任何你预计在 TTL 内至少复用 3 次的内容。

## 构建

### Step 1：Anthropic prompt caching，使用显式标记

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 将块存储 5 分钟。在该窗口内复用命中；过期后重新写入。

**响应 usage 字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中检查两个字段 —— 如果 `cache_read_input_tokens` 在跨请求时始终为零，说明你的缓存 key 在漂移。

### Step 2：一小时延长 TTL

对于长时间运行的批处理作业，5 分钟默认值会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价为 2x（比基线高 50% 而非 25%），但只要在批处理中复用前缀超过 5 次就能快速回本。

### Step 3：OpenAI 自动缓存

OpenAI 无需你配置任何内容。任何超过 1,024 tokens 且与最近请求匹配的前缀自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样的缓存友好布局规则适用。有两件事会杀死 OpenAI 的缓存而不会影响 Anthropic 的：更改 `user` 字段（用作缓存 key 组件）和重新排序 tools。

### Step 4：Gemini 显式 context caching

Gemini 将缓存视为你可以创建和命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 按 token·小时 收取缓存存活期间的存储费，读取费用约为正常 input 费率的 ~25%。当你跨多天在多会话中复用相同的巨大 prompt 时，这种形状最合适。

### Step 5：在生产环境中测量命中率

参见 `code/main.py` 了解一个模拟的三提供商成本核算器，它追踪 write/read/miss 计数并计算每 1K 请求的混合成本。以目标命中率作为部署门槛 —— 大多数生产环境的 Anthropic 设置在预热后应看到 >80% 的读取比例。

## 2026 年仍在发生的陷阱

- **顶部的动态时间戳。** `"Current time: 2026-04-22 15:30:02"` 放在 system prompt 顶部。每个请求都 miss。将时间戳移到缓存断点下方。
- **Tool 重新排序。** 以稳定顺序序列化 tools —— 部署之间的 dict 重排会破坏每次命中。
- **自由文本近似重复。** "You are helpful." vs "You are a helpful assistant." —— 一个字节差异 = 完全 miss。
- **块太小。** Anthropic 强制执行 1,024-token 下限（Haiku 为 2,048）。更小的块静默不缓存。
- **盲目的成本仪表盘。** 将 "input tokens" 拆分为 cached vs uncached。否则流量下降看起来像是缓存胜利。

## 使用

2026 年的缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| Agent 有稳定的 10k+ system prompt，多轮对话 | Anthropic `cache_control`，5 分钟 TTL |
| 批处理作业复用前缀 30+ 分钟 | Anthropic，`ttl: "1h"` |
| GPT-5 的无服务器端点，无自定义基础设施 | OpenAI 自动（只需让前缀稳定且足够长） |
| 跨多天复用巨大的代码/文档语料库 | Gemini 显式 `CachedContent` |
| 跨提供商降级 | 保持可缓存前缀布局在所有提供商间一致，以便任何命中都有效 |

与 semantic caching（Phase 11 · 11）结合用于 user-message 层：prompt caching 处理*token 完全相同*的复用，semantic caching 处理*意思相同*的复用。

## 交付

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: 设计缓存友好的 prompt 布局并选择正确的提供商缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定一个 prompt（system + tools + few-shot + retrieval + history + user）和一个使用概况（每小时请求数、所需 TTL、提供商），输出：

1. 布局。重新排序的段落，标记单个缓存断点；解释哪些段落稳定，哪些易变。
2. 提供商模式。Anthropic cache_control、OpenAI 自动或 Gemini CachedContent。从 TTL 和复用模式论证。
3. 盈亏平衡。TTL 内每次写入的预期读取次数；与无缓存的净成本对比，附数学计算。
4. 验证计划。CI 断言：第二次相同请求上 cache_read_input_tokens > 0；仪表盘按 cached vs uncached tokens 拆分。
5. 故障模式。列出此设置中缓存 miss 的三个最可能原因（动态时间戳、tool 重排、近似重复文本）及预防措施。

拒绝交付将动态字段放在断点上方的缓存计划。拒绝在未达到使 2x 写入溢价回本的复用次数时启用 1h TTL。
```

## 练习

1. **简单。** 针对 Claude 进行一次 10 轮对话，system prompt 为 5,000 tokens。先不带 `cache_control` 运行，然后带它运行。报告每种情况的 input-token 账单。
2. **中等。** 编写一个测试 harness，给定一个 prompt template 和一个请求日志，计算每个提供商的预期命中率和美元节省（Anthropic 5m、Anthropic 1h、OpenAI 自动、Gemini 显式）。
3. **困难。** 构建一个布局优化器：给定一个 prompt 和一个标记为 `stable=True/False` 的字段列表，重写 prompt 以将单个缓存断点放在最大缓存友好位置且不丢失信息。在真实的 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Prompt caching | "让长 prompt 变便宜" | 复用提供商端的 KV-cache 以匹配前缀；重复 input tokens 享受 50-90% 折扣。 |
| `cache_control` | "Anthropic 的标记" | 声明"到此为止的所有内容都可缓存"的 content-block 属性；`{"type": "ephemeral"}`。 |
| Cache write | "支付溢价" | 首次填充缓存的请求；Anthropic 按 ~1.25x input 费率计费，OpenAI 免费。 |
| Cache read | "折扣" | 后续匹配前缀的请求；按 10%（Anthropic）、50%（OpenAI）、~25%（Gemini）计费。 |
| TTL | "存活多久" | 缓存保持温热的时间；Anthropic 默认 5 分钟（可延长 1 小时），OpenAI 尽力而为最长 1 小时，Gemini 用户设置。 |
| Extended TTL | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；2x 写入溢价，但批处理复用回本快。 |
| Prefix match | "为什么我的缓存没命中" | 只有从开头到断点的每个 token 都字节相同时，缓存才会命中。 |
| Context caching (Gemini) | "显式那个" | Google 的命名、按存储计费的缓存对象；最适合跨多天复用大型语料库。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— `cache_control`、1h TTL、盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) —— 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) —— `CachedContent` API 和存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) —— 原始发布帖子，含延迟数据。
- Phase 11 · 05（Context Engineering）—— 在哪里切分 prompt 以便缓存可以落地。
- Phase 11 · 11（Caching and Cost）—— 将 prompt caching 与 user messages 上的 semantic cache 配对。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) —— prompt caching 向用户暴露的 KV-cache 内存模型；解释为什么缓存前缀的重新读取成本比重新编码低约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) —— prefill 是 prompt caching 跳过的阶段；本文解释了为什么 TTFT 在缓存命中时大幅下降而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) —— prompt caching 与 speculative decoding、Flash Attention 和 MQA/GQA 并列为弯曲推理成本曲线的杠杆；阅读本文了解其他三个。
