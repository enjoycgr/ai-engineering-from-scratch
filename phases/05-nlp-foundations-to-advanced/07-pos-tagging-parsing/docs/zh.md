# POS 标注与句法分析

> 语法曾一度不受欢迎。后来每个大语言模型（LLM）流水线都需要验证结构化抽取，于是它又回来了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 分钟

## The Problem

第 01 课曾提到，词形还原（lemmatization）需要词性标签。如果不知道 `running` 是动词，词形还原器就无法把它还原为 `run`；如果不知道 `better` 是形容词，也无法还原为 `good`。

这个承诺背后隐藏了一整个子领域。词性标注（POS tagging）为每个词分配语法类别。句法分析（syntactic parsing）还原句子的树形结构：哪个词修饰哪个词，哪个动词支配哪些论元。传统 NLP 花了二十年打磨这两者。随后深度学习把它们压缩成基于预训练 transformer 的词分类任务，研究社区便转向了别处。

但应用社区没有。每个结构化抽取流水线仍在底层使用 POS 和依存树。大语言模型生成的 JSON 需要根据语法约束进行验证。问答系统利用依存分析来分解查询。机器翻译质量评估器检查分析树的对齐。

值得了解。本课介绍标签集、基线，以及你应该停止从零实现、转而调用 spaCy 的那个临界点。

## The Concept

**POS 标注**为每个词标注一个语法类别。Penn Treebank (PTB) 标签集是英语的默认标准，包含 36 个标签，其中一些区分对普通读者来说显得过于琐碎：`NN` 单数名词、`NNS` 复数名词、`NNP` 专有名词单数、`VBD` 动词过去式、`VBZ` 动词第三人称单数现在时，等等。**Universal Dependencies (UD)** 标签集更粗粒度（17 个标签）且语言无关；它已成为跨语言工作的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析**生成一棵树。主要有两种风格：

- **成分分析（Constituency parsing）。** 名词短语、动词短语、介词短语相互嵌套。输出是一棵由非终结类别（NP、VP、PP）构成的树，词作为叶子节点。
- **依存分析（Dependency parsing）。** 每个词都有一个它依附的中心词（head），并标注一条语法关系。输出是一棵树，其中每条边都是一个（head, dependent, relation）三元组。

依存分析在 2010 年代胜出，因为它能干净地跨语言泛化，尤其适用于自由语序的语言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Build It

### Step 1: 最频繁标签基线

最傻但有效的 POS 标注器。对每个词，预测它在训练集中出现次数最多的标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 语料库上，这个基线能达到约 85% 的准确率。不算好，但它是任何严肃模型都不应跌破的底线。

### Step 2: 二元 HMM 标注器

对序列的联合概率进行建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两张表：转移概率（给定前一个标签的标签）和发射概率（给定标签的词）。两者都通过带 Laplace 平滑的计数来估计。用 Viterbi 算法（在标签格子上进行动态规划）进行解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

在 Brown 语料库上，二元 HMM 的准确率约为 93%。从 85% 到 93% 的提升主要归功于转移概率——模型学会了 `DET NOUN` 很常见，而 `NOUN DET` 很罕见。

### Step 3: 为什么现代标注器能超越它

转移概率 + 发射概率都是局部的。它们无法捕捉 `saw` 在 "I bought a saw" 中是名词，而在 "I saw the movie" 中是动词。一个带有任意特征（后缀、词形、前后词、词本身）的 CRF 能达到约 97%。BiLSTM-CRF 或 transformer 能达到 98% 以上。

这项任务的天花板由标注者分歧决定。人类标注者在 Penn Treebank 上的一致性约为 97%。超过 98% 的模型很可能是在过拟合测试集。

### Step 4: 依存分析概述

完整的依存分析从零实现超出了本课范围；经典教材处理见 Jurafsky 和 Martin。需要了解的两大经典家族：

- **基于转移（Transition-based）** 的分析器（arc-eager、arc-standard）行为类似移进-归约分析器：读取词，把它们移进栈，然后应用创建弧的归约动作。贪心解码速度快。经典实现是 MaltParser。现代神经网络版本：Chen and Manning 的基于转移分析器。
- **基于图（Graph-based）** 的分析器（Eisner 算法、Dozat-Manning biaffine）为每个可能的 head-dependent 边打分，然后选择最大生成树。更慢但更准确。

对于大多数应用工作，直接调用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上读 `dep` 列，句子的语法结构就一目了然。

## Use It

每个生产级 NLP 库都把 POS 和依存分析器作为标准流水线的一部分提供。

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`)。快速、准确，与分词 + NER + 词形还原集成。`token.tag_` (PTB)、`token.pos_` (UD)、`token.dep_` (依存关系)。
- **Stanford NLP (stanza)**。Stanford 对 CoreNLP 的继任者。在 60+ 种语言上达到最先进水平。
- **trankit**。基于 transformer，UD 准确率高。
- **NLTK**。`pos_tag`。可用，慢，较老。适合教学。

### 在 2026 年这为什么仍然重要

- **词形还原。** 第 01 课需要 POS 才能正确还原词形。永远如此。
- **从 LLM 输出中进行结构化抽取。** 验证生成的句子是否遵守语法约束（例如主谓一致、必需的修饰语）。
- **基于方面的情感分析。** 依存分析告诉你哪个形容词修饰哪个名词。
- **查询理解。** "movies directed by Wes Anderson starring Bill Murray" 通过分析可分解为结构化约束。
- **跨语言迁移。** UD 标签和依存关系是语言无关的，使得对新语言进行零样本结构化分析成为可能。
- **低算力流水线。** 如果你无法部署 transformer，POS + 依存分析 + 地名词典（gazetteer）能让你走得更远。

## Ship It

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: 为下游 NLP 任务设计经典的 POS + 依存分析流水线。
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

给定一个下游任务（信息抽取、改写验证、查询分解、词形还原），输出：

1. 要使用的标签集。纯英语遗留流水线用 Penn Treebank，多语言或跨语言用 Universal Dependencies。
2. 库。大多数生产环境用 spaCy，学术级多语言用 stanza，最高 UD 准确率用 trankit。给出具体模型 ID。
3. 集成模式。展示调用库并消费所需属性（`.pos_`、`.dep_`、`.head`）的 3-5 行代码。
4. 需要测试的失效模式。名词-动词歧义（`saw`、`book`、`can`）和介词短语附着歧义是经典陷阱。采样 20 条输出并人工检查。

拒绝推荐从零自建分析器。从零构建分析器是一个研究项目，不是应用任务。标记任何不处理大小写变体就消费 POS 标签的流水线为脆弱。
```

## Exercises

1. **Easy.** 在一个小型标注语料库（例如 NLTK 的 Brown 子集）上使用最频繁标签基线，在留出句子上测量准确率。验证约 85% 的结果。
2. **Medium.** 训练上面的二元 HMM 并报告每个标签的精确率/召回率。HMM 最容易混淆哪些标签？
3. **Hard.** 使用 spaCy 的依存分析从 1000 句样本中提取主-谓-宾三元组。在 50 条人工标注的三元组上评估。记录提取失败的地方（通常是被动语态、并列结构和省略主语）。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| POS tag | Word's type | 语法类别。PTB 有 36 个；UD 有 17 个。 |
| Penn Treebank | Standard tagset | 英语专用。对动词时态和名词数有更细粒度的区分。 |
| Universal Dependencies | Multilingual tagset | 比 PTB 更粗粒度；语言无关；跨语言工作的默认选择。 |
| Dependency parse | Sentence tree | 每个词有一个中心词，每条边有一个语法关系。 |
| Viterbi | Dynamic programming | 给定发射概率和转移概率，找出概率最高的标签序列。 |

## Further Reading

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS 和句法分析的经典教材处理。
- [Universal Dependencies project](https://universaldependencies.org/) — 跨语言标签集和树库集合，每个多语言分析器都在使用。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每个暴露属性的实用参考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 把神经网络分析器带入主流的论文。
