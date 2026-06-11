# Text Summarization（文本摘要）

> Extractive systems tell you what the document said. Abstractive systems tell you what the author meant. Different tasks, different pitfalls.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 分钟

## The Problem（问题）

一篇 2,000 字的新闻文章出现在你的信息流中。你需要用 120 字概括它。你可以从文章中挑选最重要的三句话（extractive summarization，抽取式摘要），或者用自己的话重写内容（abstractive summarization，生成式摘要）。两者都叫 summarization（摘要）。它们是完全不同的问题。

Extractive summarization 是一个 ranking problem（排序问题）。给每个句子打分，返回 top-`k`。输出总是语法正确的，因为它是逐字提取的。风险在于可能遗漏分散在文章各处的内容。

Abstractive summarization 是一个 generation problem（生成问题）。Transformer 根据输入生成新文本。输出流畅且压缩度高，但可能 hallucinate（幻觉）源文本中不存在的事实。风险在于自信的捏造。

本课将构建两种方法，并分析每种方法各自的 failure mode（失效模式）。

## The Concept（概念）

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Extractive（抽取式）。** 将文章视为图，节点是句子，边是相似度。在图上运行 PageRank（或类似算法），根据句子与其他所有内容的连接程度来打分。得分最高的句子构成摘要。经典实现是 **TextRank** (Mihalcea and Tarau, 2004)。

**Abstractive（生成式）。** 在文档-摘要对上 fine-tune（微调）transformer encoder-decoder（BART, T5, Pegasus）。在 inference（推理）时，模型通过 cross-attention（交叉注意力）读取文档并逐 token 生成摘要。Pegasus 尤其使用 gap-sentence pretraining objective（间隙句子预训练目标），使其在 summarization 上表现优异，无需太多 fine-tuning。

使用 **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) 进行评估。ROUGE-1 和 ROUGE-2 分别评分 unigram 和 bigram 重叠。ROUGE-L 评分 longest common subsequence（最长公共子序列）。越高越好，但 40 ROUGE-L 是"良好"，50 是"卓越"。每篇论文都会报告这三个指标。使用 `rouge-score` 包。

## Build It（动手实现）

### Step 1: TextRank (extractive)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两件事值得说明。Similarity function 使用 log-normalized word overlap，这是原始 TextRank 的变体。TF-IDF 向量的余弦相似度也能工作。Damping factor（阻尼系数）0.85 和迭代次数是 PageRank 的默认值。

### Step 2: abstractive with BART（使用 BART 的生成式摘要）

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 在 CNN/DailyMail 语料库上进行了 fine-tune。它能直接生成新闻风格的摘要。对于其他领域（科学论文、对话、法律），使用对应的 Pegasus checkpoint 或在目标数据上 fine-tune。

### Step 3: ROUGE evaluation（ROUGE 评估）

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用 stemming（词干提取）。没有它，"running" 和 "run" 会被算作不同的词，ROUGE 会低估重叠。

### Beyond ROUGE（超越 ROUGE，2026 年的摘要评估）

ROUGE 二十年来一直是主导的 summarization metric（摘要指标），但在 2026 年单独使用已不足够。一项对 NLG 论文的大规模 meta-analysis（元分析）显示：

- **BERTScore** (contextual embedding similarity) 在 2023 年后逐渐普及，现在大多数 summarization 论文都会与 ROUGE 一起报告。
- **BARTScore** 将评估视为 generation（生成）：通过预训练 BART 给定的 source 对 summary 赋予的概率来评分。
- **MoverScore** (Earth Mover's Distance over contextual embeddings) 在 2025 年的 summarization benchmarks 中登顶，因为它比 ROUGE 更好地捕捉语义重叠。
- **FactCC** 和 **QA-based faithfulness** 在 2021-2023 年很常见，现在常被 **G-Eval** 取代（一个 GPT-4 prompt chain，通过 chain-of-thought reasoning 评分 coherence、consistency、fluency、relevance）。
- **G-Eval** 和类似的 LLM-judge 方法在评分标准设计良好时，与人类判断一致率约为 80%。

Production recommendation（生产建议）：报告 ROUGE-L 用于 legacy comparison（遗留对比），BERTScore 用于 semantic overlap（语义重叠），G-Eval 用于 coherence 和 factuality（事实性）。在将其用于生产数据之前，先用 50-100 条人工标注的摘要进行校准。

### Step 4: the factuality problem（事实性问题）

Abstractive summaries 容易产生 hallucination（幻觉）。Extractive summaries 的 hallucination risk（幻觉风险）低得多，因为输出是逐字从源文本提取的，尽管如果源句子被 decontextualized（去语境化）、过时或顺序错乱，仍可能产生误导。这是生产系统仍然偏好 extractive methods（抽取式方法）用于 compliance-adjacent content（合规相关内容）的最大原因。

需要命名的 hallucination 类型：

- **Entity swap（实体替换）。** 源文本说 "John Smith"，摘要说 "John Brown"。
- **Number drift（数字漂移）。** 源文本说 "25,000"，摘要说 "25 million"。
- **Polarity flip（极性翻转）。** 源文本说 "rejected the offer"，摘要说 "accepted the offer"。
- **Fact invention（事实捏造）。** 源文本未提及 CEO，摘要说 CEO 批准了。

有效的评估方法：

- **FactCC。** 在源句子和摘要句子之间训练的二元分类器，基于 entailment（蕴含关系）。预测 factual / not-factual。
- **QA-based factuality（基于问答的事实性）。** 向 QA 模型提问，答案应在源文本中。如果摘要支持不同的答案，则标记。
- **Entity-level F1。** 比较源文本和摘要中的 named entities。仅出现在摘要中的实体值得怀疑。

对于任何面向用户且 factuality 重要的场景（新闻、医疗、法律、金融），extractive 是更安全的默认选择。Abstractive 需要在循环中加入 factuality check（事实性检查）。

## Use It（使用）

2026 年的 stack：

| Use case | Recommended |
|---------|-------------|
| News, 3-5 sentence summary, English | `facebook/bart-large-cnn` |
| Scientific papers | `google/pegasus-pubmed` or a tuned T5 |
| Multi-document, long-form | Any LLM with 32k+ context, prompted |
| Dialog summarization | `philschmid/bart-large-cnn-samsum` |
| Extractive, low hallucination risk by construction | TextRank or `sumy`'s LSA / LexRank |

在计算资源不受限时，长上下文 LLM 在 2026 年常常超越 specialized models（专用模型）。权衡在于成本和 reproducibility（可复现性）；专用模型给出更一致的输出。

## Ship It（部署）

Save as `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Exercises（练习）

1. **Easy.** 在 5 篇新闻文章上运行 TextRank。将 top-3 句子与 reference summary（参考摘要）比较。测量 ROUGE-L。在 CNN/DailyMail 风格的文章上，你应该能看到 30-45 的 ROUGE-L。
2. **Medium.** 实现 entity-level factuality（实体级别事实性）：从源文本和摘要中提取 named entities（使用 spaCy），计算源实体在摘要中的 recall 和摘要实体相对于源文本的 precision。Precision 高而 recall 低意味着安全但简短；precision 低意味着幻觉实体。
3. **Hard.** 在 50 篇 CNN/DailyMail 文章上比较 BART-large-CNN 与 LLM（Claude 或 GPT-4）。报告 ROUGE-L、factuality（通过 entity F1）和每条摘要的成本。记录各自在哪些方面胜出。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Extractive | Pick sentences | Return sentences verbatim from the source. Never hallucinates. |
| Abstractive | Rewrite | Generate new text conditioned on source. Can hallucinate. |
| ROUGE | Summary metric | N-gram / LCS overlap between system output and reference. |
| TextRank | Graph-based extractive | PageRank over sentence similarity graph. |
| Factuality | Is it right | Whether summary claims are supported by the source. |
| Hallucination | Made-up content | Content in the summary that the source does not support. |

## Further Reading（延伸阅读）

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 抽取式摘要的经典论文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 论文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 与 gap-sentence objective。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 论文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — factuality 领域综述论文。
