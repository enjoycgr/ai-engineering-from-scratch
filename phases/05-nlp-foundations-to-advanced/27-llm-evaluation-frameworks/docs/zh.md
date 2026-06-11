# LLM 评估 —— RAGAS、DeepEval、G-Eval

> Exact Match 和 F1 会遗漏语义等价性。人工审核无法扩展。LLM-as-judge（大语言模型作为评判者）是生产环境中的答案 —— 前提是经过充分校准，让你能信任这个数字。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 分钟

## 问题所在

你的 RAG 系统回答："June 29th, 2007"。
黄金参考答案是："June 29, 2007"。
Exact Match 得分为 0。F1 得分约为 75%。而人类会给出 100% 的分数。

现在将这种情况乘以 10,000 个测试用例。再乘以检索器、分块、提示词或模型的每一次变更。你需要一个能理解语义、能低成本大规模运行、不会隐瞒回归问题、并能暴露正确失败模式的评估器。

2026 年有三个框架主导了这个领域。

- **RAGAS。** Retrieval-Augmented Generation ASsessment（检索增强生成评估）。四个 RAG 指标（faithfulness（忠实度）、answer-relevance（答案相关性）、context-precision（上下文精确率）、context-recall（上下文召回率）），后端使用 NLI + LLM-judge。有研究背书，轻量级。
- **DeepEval。** Pytest for LLMs（面向 LLM 的 Pytest）。G-Eval、task-completion（任务完成度）、hallucination（幻觉）、bias（偏见）指标。原生支持 CI/CD。
- **G-Eval。** 一种方法（也是 DeepEval 的一个指标）：LLM-as-judge（大语言模型作为评判者），配合 chain-of-thought（思维链）、自定义标准、0-1 分数。

三者都依赖 LLM-as-judge。本课将建立对该方法及其信任层（trust layer）的直觉。

## 核心概念

![四个评估维度，LLM-as-judge 架构](../assets/llm-evaluation.svg)

**LLM-as-judge（大语言模型作为评判者）。** 用一个 LLM 替代静态指标，根据评分标准（rubric）为输出打分。给定 `(query, context, answer)`，向 judge LLM 提示："在忠实度上打 0-1 分。" 返回分数。

为何有效：LLM 以极低成本近似人类判断。GPT-4o-mini 每个评分用例约 $0.003，1000 个样本的回归评估运行成本不到 $5。

为何静默失效：

1. **Judge bias（评判者偏见）。** 评判者偏好更长的答案、来自同一模型家族的答案、与提示风格匹配的回答。
2. **JSON 解析失败。** 错误的 JSON → NaN 分数 → 被静默排除在聚合统计之外。RAGAS 用户深知此痛。用 try/except + 显式失败模式做防护。
3. **模型版本漂移。** 升级评判者会改变所有指标。冻结 judge model + version（评判模型及其版本）。

**RAG 四大指标。**

| 指标 | 问题 | 后端 |
|------|------|------|
| Faithfulness（忠实度） | 答案中的每个 claim（断言）是否都来自检索到的上下文？ | 基于 NLI 的 entailment（蕴含） |
| Answer relevance（答案相关性） | 答案是否回答了问题？ | 从答案生成假设问题；与真实问题比较 |
| Context precision（上下文精确率） | 在检索到的 chunk（块）中，有多少比例是相关的？ | LLM-judge |
| Context recall（上下文召回率） | 检索是否返回了所有需要的内容？ | LLM-judge 对比黄金答案 |

**G-Eval。** 定义一个自定义标准："答案是否引用了正确的来源？" 框架自动将其扩展为 chain-of-thought（思维链）评估步骤，然后给出 0-1 分。适用于 RAGAS 未覆盖的领域特定质量维度。

**Calibration（校准）。** 永远不要信任原始评判分数，除非你已验证过它与人工标签的相关性。运行 100 个手工标注的示例。绘制评判者 vs 人类的散点图。计算 Spearman rho。如果 rho < 0.7，你的评判标准需要改进。

## 动手构建

### 步骤 1：使用 NLI 计算忠实度（RAGAS 风格）

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` 是任意可调用对象：prompt str -> generated str。
# 示例：llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为 atomic claims（原子断言）。用 NLI 逐一检查每个 claim 与检索到的上下文之间的 entailment（蕴含关系）。Faithfulness（忠实度）= 被支持 claim 的比例。

### 步骤 2：答案相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: 任何实现了 .encode(texts, normalize_embeddings=True) -> ndarray 的模型
# 例如 encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果答案暗示的问题与所问问题不同，相关性就会下降。

### 步骤 3：G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

evaluation steps（评估步骤）就是评分标准（rubric）。显式步骤比隐式的 "score 0-1" 提示更稳定。

### 步骤 4：CI 门禁

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

作为 pytest 文件发布。每次 PR 都运行。回归问题阻塞合并。

### 步骤 5：从零开始的玩具级评估

见 `code/main.py`。仅使用标准库对 faithfulness（答案 claim 与上下文的重叠）和 relevance（答案 token 与问题 token 的重叠）进行近似。非生产级。展示的是评估框架的骨架。

## 常见陷阱

- **未做校准。** 与人工标签相关性仅为 0.3 的评判者就是噪声。在上线前要求完成校准运行。
- **自评（self-evaluation）。** 使用同一个 LLM 生成和评判会使分数虚高 10-20%。评判者应使用不同的模型家族。
- **成对评判中的位置偏见。** 评判者偏好第一个呈现的选项。始终随机化顺序并双向运行。
- **原始聚合值掩盖失败。** 平均分 0.85 往往掩盖了 5% 的灾难性失败。始终检查底部 quantile（分位数）。
- **黄金数据集腐化。** 未版本化的评估集随时间漂移，破坏纵向比较。每次变更都要给数据集打 tag。
- **LLM 成本。** 大规模时，评判调用占成本主导。使用满足校准阈值的最便宜模型。GPT-4o-mini、Claude Haiku、Mistral-small。

## 如何使用

2026 年技术栈：

| 使用场景 | 框架 |
|---------|------|
| RAG 质量监控 | RAGAS（4 个指标） |
| CI/CD 回归门禁 | DeepEval + pytest |
| 自定义领域标准 | DeepEval 内的 G-Eval |
| 在线实时流量监控 | RAGAS reference-free 模式 |
| 人工抽检 | LangSmith 或 Phoenix（带标注 UI） |
| 红队测试 / 安全评估 | Promptfoo + DeepEval |

典型栈：RAGAS 用于监控，DeepEval 用于 CI，G-Eval 用于新维度。三者都跑；它们的分歧本身就有信息量。

## 交付物

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习

1. **简单。** 在 10 个已知存在幻觉的 RAG 示例上使用 RAGAS。验证 faithfulness 指标能否捕获每一个。
2. **中等。** 手工标注 50 个 QA 答案的正确性（0-1 分）。用 G-Eval 打分。测量评判者与人类之间的 Spearman rho。
3. **困难。** 用 DeepEval 构建一个 pytest CI 门禁。故意让检索器退化。验证门禁是否失败。通过对最低 10% 的阈值检查添加 bottom-quantile 告警。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| LLM-as-judge | 用 LLM 打分 | 向 judge model 提供评分标准，让其为输出打 0-1 分。 |
| RAGAS | RAG 指标库 | 开源评估框架，包含 4 个 reference-free（无参考）RAG 指标。 |
| Faithfulness | 答案是否 grounded？ | 答案中的 claim 被检索上下文 entail（蕴含）的比例。 |
| Context precision | 检索到的 chunk 是否相关？ | 前 K 个 chunk 中真正起作用的比例。 |
| Context recall | 检索是否找全了？ | 黄金答案中的 claim 被检索 chunk 支持的比例。 |
| G-Eval | 自定义 LLM 评判 | 评分标准 + chain-of-thought（思维链）评估步骤 + 0-1 分。 |
| Calibration | 信任但验证 | 评判者分数与人类分数之间的 Spearman 相关性。 |

## 延伸阅读

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) —— RAGAS 论文。
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) —— G-Eval 论文。
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) —— 开源生产栈。
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) —— 偏见、校准与局限。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) —— 整合 RAGAS、DeepEval、Phoenix 的统一框架。
