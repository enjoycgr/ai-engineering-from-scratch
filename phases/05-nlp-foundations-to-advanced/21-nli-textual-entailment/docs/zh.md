# 自然语言推理 —— 文本蕴含

> "t entails h" 意味着阅读 t 的人会得出 h 为真的结论。NLI 是预测蕴含 / 矛盾 / 中立的任务。表面上看起来枯燥，但在生产环境中至关重要。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 分钟

## 问题背景

你构建了一个摘要生成器。它生成了摘要。你如何知道摘要中不包含幻觉（hallucination）？

你构建了一个聊天机器人。它回答了"是"。你如何知道该答案是否被检索到的段落所支持？

你需要对 10,000 篇新闻文章按主题分类。你没有训练标签。你能复用某个模型吗？

这三个问题都可以归结为自然语言推理（Natural Language Inference, NLI）。NLI 问的是：给定前提 `t` 和假设 `h`，`h` 是被 `t` 蕴含（entailment）、矛盾（contradiction），还是中立（neutral，无关）？

- **幻觉检测：** `t` = 源文档，`h` = 摘要中的主张。非蕴含 = 幻觉。
- **有依据的问答（Grounded QA）：** `t` = 检索到的段落，`h` = 生成的答案。非蕴含 = 捏造。
- **零样本分类（Zero-shot classification）：** `t` = 文档，`h` = 语言化的标签（"This is about sports"）。蕴含 = 预测标签。

一个任务，三种生产用途。这就是为什么每个 RAG 评估框架背后都内置了一个 NLI 模型。

## 核心概念

![NLI: 三分类，前提 vs 假设](../assets/nli.svg)

**三个标签。**

- **蕴含（Entailment）。** `t` → `h`。"The cat is on the mat" 蕴含 "There is a cat."
- **矛盾（Contradiction）。** `t` → ¬`h`。"The cat is on the mat" 矛盾于 "There is no cat."
- **中立（Neutral）。** 无法推断任何方向。"The cat is on the mat" 对 "The cat is hungry." 是中立的。

**不是逻辑蕴含。** NLI 是*自然*语言推理——典型的人类读者会推断出什么，而不是严格的逻辑。"John walked his dog" 在 NLI 中蕴含 "John has a dog"，但严格的一阶逻辑只有在将拥有关系公理化后才会承认这一点。

**数据集。**

- **SNLI** (2015)。570k 人工标注对，以图像描述作为前提。领域较窄。
- **MultiNLI** (2017)。433k 对，涵盖 10 个领域。2026 年的标准训练语料。
- **ANLI** (2019)。对抗性 NLI。人类专门编写用来攻破现有模型的例子。更难。
- **DocNLI, ConTRoL** (2020–21)。文档级前提。测试多跳（multi-hop）和长距离推理。

**架构。** 一个 transformer 编码器（BERT, RoBERTa, DeBERTa）读取 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 表示输入到一个 3 分类的 softmax (软最大值)。在 MNLI 上训练，在留出基准上评估，在分布内对子上可达到 90%+ 准确率。

**通过 NLI 实现零样本分类。** 给定一个文档和候选标签，将每个标签转化为假设（"This text is about sports"）。计算每个标签的蕴含概率。选择最大值。这就是 Hugging Face `zero-shot-classification` pipeline 背后的机制。

## 动手实践

### 步骤 1：运行预训练 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

对于生产环境的 NLI，`facebook/bart-large-mnli` 和 `microsoft/deberta-v3-large-mnli` 是默认的开源选择。DeBERTa-v3 在排行榜上名列前茅。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板是 "This example is about {label}." 可通过 `hypothesis_template` 自定义。无需训练数据。无需 fine-tuning (微调)。开箱即用。

### 步骤 3：RAG 的忠实度检查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这是 RAGAS 忠实度（faithfulness）的核心。将生成的答案拆分为原子级主张。针对检索到的上下文逐一检查。报告蕴含主张的比例。

### 步骤 4：手写的 NLI 分类器（概念性）

参见 `code/main.py` 中的纯标准库玩具示例：前提和假设通过词汇重叠 + 否定检测进行比较。无法与 transformer 模型竞争——但它展示了任务的本质：两个文本输入，3 分类输出，loss (损失函数) = 在 `{entail, contradict, neutral}` 上的 cross-entropy (交叉熵)。

## 常见陷阱

- **仅假设捷径（Hypothesis-only shortcuts）。** 模型仅通过假设本身就能以约 60% 的准确率预测 SNLI 的标签，因为 "not"、"nobody"、"never" 与矛盾相关。这是检测标签泄露的有力基线。
- **词汇重叠启发式。** 子序列启发式（"每个子序列都被蕴含"）能通过 SNLI，但在 HANS/ANLI 上失败。请使用对抗性基准。
- **文档长度退化。** 单句 NLI 模型在文档级前提上 F1 下降 20+。长上下文请使用 DocNLI 训练的模型。
- **零样本模板敏感性。** "This example is about {label}" 与 "{label}" 与 "The topic is {label}" 之间的准确率可能相差 10+ 个百分点。请调优模板。
- **领域不匹配。** MNLI 训练于通用英语。法律、医学和科学文本需要领域特定的 NLI 模型（例如 SciNLI, MedNLI）。

## 实际应用

2026 年技术栈：

| 使用场景 | 模型 |
|---------|-------|
| 通用 NLI | `microsoft/deberta-v3-large-mnli` |
| 快速 / 边缘部署 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量） | `facebook/bart-large-mnli` |
| 文档级 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻觉检测 | RAGAS / DeepEval 中的 NLI 层 |

2026 年的元模式：NLI 是文本理解的万能胶带。每当你需要 "A 是否支持 B？" 或 "A 是否矛盾于 B？" —— 先尝试 NLI，再考虑额外的 LLM 调用。

## 交付物

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习

1. **简单。** 在 20 个手工构建的 (premise, hypothesis, label) 三元组上运行 `facebook/bart-large-mnli`，涵盖所有三个类别。测量准确率。添加对抗性 "子序列启发式" 陷阱（"I did not eat the cake" vs "I ate the cake"），观察模型是否失效。
2. **中等。** 在 100 条 AG News 标题上比较零样本模板 `"This text is about {label}"` 与 `"The topic is {label}"` 和 `"{label}"`。报告准确率波动。
3. **困难。** 构建一个 RAG 忠实度检查器：原子级主张分解 + 每个主张的 NLI 检查。在 50 个带有 gold context 的 RAG 生成答案上评估。测量与手工标注相比的假阳性率和假阴性率。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| NLI | Natural Language Inference | 前提-假设关系的三分类。 |
| RTE | Recognizing Textual Entailment | NLI 的旧称；同一任务。 |
| Entailment | "t implies h" | 典型读者会得出给定 t 则 h 为真的结论。 |
| Contradiction | "t rules out h" | 典型读者会得出给定 t 则 h 为假的结论。 |
| Neutral | "undecided" | 从 t 到 h 无法推断任何方向。 |
| Zero-shot classification | NLI as classifier | 将标签语言化为假设，选择最大蕴含。 |
| Faithfulness | Is the answer supported? | 在（检索到的上下文，生成的答案）上应用 NLI。 |

## 延伸阅读

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI 基准。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年的 NLI 主力模型。
