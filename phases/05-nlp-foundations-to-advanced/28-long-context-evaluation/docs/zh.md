# 长上下文评估 —— NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称支持 10M token 的上下文。在 1M token 时，8-needle MRCR 降至 26.3%。Advertised（宣称值）≠ usable（可用值）。长上下文评估告诉你正在部署的模型的实际容量。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 分钟

## 问题所在

你有一份 200 页的合同。模型宣称支持 1M token 的上下文。你把合同贴进去问："终止条款是什么？" 模型回答了 —— 但答案来自封面页，因为终止条款位于 120k token 深处，超出了模型实际关注（attend）的范围。

这就是 2026 年的 context-capacity gap（上下文容量差距）。规格表说 1M 或 10M。现实是其中 60-70% 可用，而"可用"取决于任务。

- **Retrieval（检索，单 needle in haystack）：** 在 frontier models（前沿模型）上，接近完美，直到宣称的最大值。
- **Multi-hop / aggregation（多跳 / 聚合）：** 在大多数模型上，超过 ~128k 后急剧下降。
- **Reasoning over dispersed facts（对分散事实的推理）：** 最先失败的任务。

长上下文评估衡量这些维度。本课介绍各个 benchmark（基准测试）的名称、每个实际衡量什么，以及如何为你的领域构建自定义 needle test（针测试）。

## 核心概念

![NIAH 基线、RULER 多任务、LongBench 整体评估](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023)。** 将一个事实（"the magic word is pineapple"）放在长上下文中的可控深度。让模型检索它。Sweep（扫描）depth × length。最初的长上下文基准测试。Frontier models 现在已经饱和（saturate）这个测试；它是必要但不充分的基线。

**RULER (Nvidia, 2024)。** 4 个类别共 13 种任务类型：retrieval（检索，单 key / 多 key / 多 value）、multi-hop tracing（多跳追踪，变量跟踪）、aggregation（聚合，常见词频）、QA。可配置上下文长度（4k 到 128k+）。揭示那些饱和 NIAH 但在 multi-hop 上失败的模型。在 2024 年的发布中，17 个宣称 32k+ 上下文的模型中只有一半在 32k 时保持了质量。

**LongBench v2 (2024)。** 503 道多选题，8k-2M 词上下文，六个任务类别：single-doc QA（单文档问答）、multi-doc QA（多文档问答）、long in-context learning（长上下文学习）、long dialogue（长对话）、code repo（代码仓库）、long structured data（长结构化数据）。真实世界长上下文行为的生产级基准测试。

**MRCR (Multi-Round Coreference Resolution，多轮共指消解)。** 大规模多轮共指。8-needle、24-needle、100-needle 变体。暴露模型在 attention（注意力）退化前能同时处理多少事实。

**NoLiMa。** "Non-lexical needle（非词汇 needle）"。Needle 和 query 没有字面重叠；检索需要一步 semantic reasoning（语义推理）。比 NIAH 更难。

**HELMET。** 拼接多个文档，从任意一个中提问。测试 selective attention（选择性注意力）。

**BABILong。** 将 bAbI reasoning chains（推理链）嵌入无关的 haystack 中。测试 reasoning-in-a-haystack（草堆中的推理），而不仅仅是 retrieval（检索）。

### 实际应报告什么

- **Advertised context window（宣称的上下文窗口）。** 规格表上的数字。
- **Effective retrieval length（有效检索长度）。** NIAH 在某个阈值（例如 90%）下的通过率。
- **Effective reasoning length（有效推理长度）。** Multi-hop 或 aggregation 在该阈值下的通过率。
- **Degradation curve（退化曲线）。** 每种任务类型的准确率 vs 上下文长度曲线。

你的规格表应有两个数字：retrieval-effective 和 reasoning-effective。通常 reasoning-effective 是宣称窗口的 25-50%。

## 动手构建

### 步骤 1：为你的领域构建自定义 NIAH

见 `code/main.py`。骨架如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # 重复 filler 直到足够长以填充 haystack 主体。
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Sweep（扫描）`depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}。绘制 heatmap（热力图）。这就是你的目标模型的 NIAH 报告卡。

### 步骤 2：multi-needle 变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

像"三个魔法词是什么？"这样的问题需要全部检索出来。Single-needle 成功不能预测 multi-needle 成功。

### 步骤 3：multi-hop variable tracing（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

答案需要链式连接三个赋值。在 128k 时，frontier models 在此类任务上的准确率常降至 50-70%。

### 步骤 4：在你的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。Aggregate scores（聚合分数）会隐藏巨大的任务级差异。

## 常见陷阱

- **仅 NIAH 评估。** 在 1M token 上通过 NIAH 对 multi-hop 毫无意义。始终运行 RULER 或自定义 multi-hop 测试。
- **Uniform depth sampling（均匀深度采样）。** 许多实现只测试 depth=0.5。测试 depth=0, 0.25, 0.5, 0.75, 1.0 —— "lost in the middle" 效应是真实存在的。
- **Needle 与 filler 的词汇重叠。** 如果 needle 与 filler 共享关键词，检索就变得过于简单。使用 NoLiMa 风格的非重叠 needle。
- **忽略延迟。** 1M token 的 prompt prefill（预填充）需要 30-120 秒。测量 time-to-first-token（首 token 时间） alongside 准确率。
- **Vendor-self-reported numbers（厂商自报数字）。** OpenAI、Google、Anthropic 都发布自己的分数。始终在你的使用场景上独立重新运行。

## 如何使用

2026 年技术栈：

| 场景 | Benchmark |
|-----------|-----------|
| 快速 sanity check（健全性检查） | 自定义 NIAH，3 个深度 × 3 个长度 |
| 生产环境模型选择 | RULER（13 个任务），在你目标长度上运行 |
| 真实世界 QA 质量 | LongBench v2 single-doc-QA 子集 |
| Multi-hop reasoning（多跳推理） | BABILong 或自定义 variable-tracing |
| 对话 / dialogue | MRCR 8-needle，在你目标长度上运行 |
| 模型升级回归测试 | 固定的内部 NIAH + RULER 测试框架，每个新模型都跑 |

生产经验法则：永远不要信任上下文窗口，除非你已在目标长度上跑过 NIAH + 1 个推理任务。

## 交付物

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习

1. **简单。** 构建一个 NIAH，3 个深度（0.25, 0.5, 0.75）× 3 个长度（1k, 4k, 16k）。在任意模型上运行。将通过率绘制为 3×3 heatmap。
2. **中等。** 添加一个 3-needle 变体。在每个长度上测量全部 3 个的检索率。与相同长度下的 single-needle 通过率比较。
3. **困难。** 构建一个 variable-tracing 任务（X1 → X2 → X3，共 3 跳），嵌入 64k 的 filler 中。在 3 个 frontier models 上测量准确率。报告每个模型的 effective reasoning length（有效推理长度）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| NIAH | Needle in haystack | 在 filler 中植入一个事实，让模型检索它。 |
| RULER | NIAH 的加强版 | 13 种任务类型，涵盖 retrieval / multi-hop / aggregation / QA。 |
| Effective context | 真实容量 | 准确率仍高于阈值的上下文长度。 |
| Lost in the middle | 深度偏见 | 模型对长输入中间部分的内容关注不足。 |
| Multi-needle | 多个事实同时存在 | 多次植入；测试注意力 juggling（调度），而非仅 retrieval。 |
| MRCR | Multi-round coref（多轮共指） | 8、24 或 100-needle 共指；暴露 attention saturation（注意力饱和）。 |
| NoLiMa | Non-lexical needle | Needle 和 query 没有字面 token 重叠；需要推理。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) —— 原始 NIAH 仓库。
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) —— 多任务基准测试。
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) —— 真实世界长上下文评估。
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) —— 更难的 needle。
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) —— reasoning-in-haystack。
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) —— 深度偏见论文。
