# Transformer 之前的文本生成 — N-gram 语言模型

> 如果一个词令人意外，模型就很差。困惑度（perplexity）把“意外”变成了数字。平滑（smoothing）让它保持有限。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 分钟

## 问题背景

在 transformer 之前，在 RNN 之前，在词嵌入（word embeddings）之前，语言模型通过统计一个词在前 `n-1` 个词之后出现的次数来预测下一个词。统计 "the cat" → "sat" 出现了 47 次，"the cat" → "jumped" 出现了 12 次，"the cat" → "refrigerator" 出现了 0 次。归一化后得到概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年，它驱动了每一个语音识别器、每一个拼写检查器和每一个基于短语的机器翻译系统。当你需要低延迟的本地语言建模时，它至今仍在运行。

有趣的问题是如何处理未见的 n-gram。原始基于计数的模型对任何未见过的序列都赋予零概率，这是灾难性的，因为句子很长，几乎每个长句都包含至少一个未见的序列。五十年的平滑（smoothing）研究解决了这个问题。Kneser-Ney 平滑是这一研究的结晶，而现代深度学习继承了它的经验传统。

## 核心概念

![N-gram 模型：计数、平滑、生成](../assets/ngram.svg)

**N-gram 概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常为 3 表示 trigram，4 表示 4-gram）。从计数中计算：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练中未出现的 n-gram 都会得到零概率。2007 年一项针对 Brown 语料库的研究发现，即使 4-gram 模型也有 30% 的 held-out 4-gram 在训练中未出现。没有平滑，你无法在任何真实文本上进行评估。

**平滑方法，按复杂程度排序：**

1. **Laplace（加一平滑）。** 给每个计数加 1。简单，但对罕见事件效果很差。
2. **Good-Turing。** 基于频率的频率（frequency-of-frequencies），将概率质量从高频事件重新分配给未见事件。
3. **Interpolation（插值）。** 将 n-gram、(n-1)-gram 等估计值用可调的权重组合。
4. **Backoff（回退）。** 如果 n-gram 计数为零，回退到 (n-1)-gram。Katz backoff 对此进行了归一化。
5. **Absolute discounting（绝对折扣）。** 从所有计数中减去一个固定的折扣 `D`，重新分配给未见事件。
6. **Kneser-Ney。** 绝对折扣加上对低阶模型的巧妙选择：使用*接续概率（continuation probability）*（一个词出现在多少个不同上下文中）而不是原始频率。

Kneser-Ney 的洞察很深刻。"San Francisco" 是一个常见的 bigram。Unigram "Francisco" 几乎只出现在 "San" 之后。朴素绝对折扣会给 "Francisco" 很高的 unigram 概率（因为计数很高）。Kneser-Ney 注意到 "Francisco" 只出现在一个上下文中，因此降低了它的接续概率。结果：一个以 "Francisco" 结尾的新 bigram 得到了适当的低概率。

**评估：困惑度（perplexity）。** 在 held-out 测试集上每个词的平均负对数似然的指数。越低越好。困惑度为 100 意味着模型的困惑程度相当于在 100 个词中均匀选择。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 动手实现

### 第一步：trigram 计数

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入是分词后的句子列表。输出是 n-gram 计数和上下文计数。`<s>` 和 `</s>` 是句子边界标记。

### 第二步：Laplace 平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

给每个计数加 1。虽然进行了平滑，但会向未见事件过度分配概率质量，同时损害已知的罕见事件。

### 第三步：Kneser-Ney（bigram，插值形式）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个关键部分。`continuation_prob` 捕捉“这个词出现在多少个不同的上下文中？”（Kneser-Ney 的创新）。`lambda_prev` 是折扣释放出的概率质量，用于加权回退（backoff）。最终概率是折扣后的主项加上加权的接续项。

### 第四步：用采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每次种子不同，输出也不同。如果需要类似 beam search 的输出，可以在每一步选择 argmax（贪婪），并加入一个小的随机性旋钮（temperature）。

### 第五步：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。对于 Brown 语料库，一个调优良好的 4-gram KN 模型困惑度约为 140。Transformer 语言模型在同一测试集上达到 15-30。差距约 10 倍。这就是领域继续前进的原因。

## 应用场景

- **经典 NLP 教学。** 接触平滑、最大似然估计（MLE）和困惑度的最清晰方式。
- **KenLM。** 生产级 n-gram 库。在延迟敏感的语音和机器翻译系统中用作重打分器（rescorer）。
- **本地自动补全。** 键盘中的 trigram 模型。至今仍在使用。
- **基线。** 在宣称你的神经语言模型很好之前，总是先计算一个 n-gram LM 的困惑度。如果你的 transformer 没有大幅超过 KN，那一定有问题。

## 交付物

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: 在训练神经语言模型之前，构建一个可复现的 n-gram 语言模型基线。
phase: 5
lesson: 16
---

给定语料库和目标用途（下一个词预测、重打分、困惑度基线），输出：

1. N-gram 阶数。通用英语用 trigram，大语料库用 4-gram，语音重打分用 5-gram。
2. 平滑方法。默认使用 Modified Kneser-Ney；教学场景可用 Laplace。
3. 库选择。生产环境用 `kenlm`，教学用 `nltk.lm`，仅为了学习数学原理时才自己实现。
4. 评估。使用 held-out 困惑度，训练集和测试集之间保持一致的 tokenization。

拒绝报告在不同 tokenization 系统之间计算的困惑度 —— 困惑度数字只有在相同 tokenization 下才可比较。标记测试集中的 OOV 比例；KN 对 OOV 处理不佳，除非你在训练期间预留了特殊的 `<UNK>` 词元。
```

## 练习

1. **简单。** 在一个 1,000 句的莎士比亚语料库上训练一个 trigram LM。生成 20 个句子。它们会在局部上合理，但在全局上不连贯。这是经典的演示。
2. **中等。** 在 held-out 的莎士比亚分割上为你的 KN 模型实现困惑度。与 Laplace 比较。你应该看到 KN 将困惑度降低 30-50%。
3. **困难。** 构建一个 trigram 拼写纠正器：给定一个拼写错误的词及其上下文，生成候选纠正并按 LM 下的上下文概率排序。在 Birkbeck 拼写语料库（公开）上评估。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| N-gram | 词序列 | `n` 个连续 token 的序列。 |
| Smoothing（平滑） | 避免零概率 | 重新分配概率质量，使未见事件获得非零概率。 |
| Perplexity（困惑度） | 语言模型质量指标 | 在 held-out 数据上的 `exp(-平均对数概率)`。越低越好。 |
| Backoff（回退） | 回退到更短的上下文 | 如果 trigram 计数为零，使用 bigram。Katz backoff 将其形式化。 |
| Kneser-Ney | 最佳 n-gram 平滑 | 绝对折扣 + 对低阶模型使用接续概率。 |
| Continuation probability（接续概率） | KN 特有 | 按词出现的上下文数量加权的 `P(w)`，而非原始计数。 |

## 延伸阅读

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM 和平滑的经典教材。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 确立 Kneser-Ney 为最佳 n-gram 平滑方法的论文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — KN 的原始论文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速生产级 n-gram LM，2026 年仍在延迟敏感应用中使用。
