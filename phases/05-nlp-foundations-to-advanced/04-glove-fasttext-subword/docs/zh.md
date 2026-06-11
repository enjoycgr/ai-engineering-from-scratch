# GloVe、FastText 与 Subword Embeddings

> Word2Vec 为每个词训练一个 embedding。GloVe 分解了共现矩阵。FastText 嵌入了词的片段。BPE 架起了通往 transformer 的桥梁。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 分钟

## The Problem

Word2Vec 留下了两个未解问题。

第一，存在一条平行研究路线直接分解共现矩阵（LSA、HAL），而不是做在线 skip-gram 更新。Word2Vec 的迭代方法本质上更好，还是差异只是两种方法处理计数方式不同造成的artifact？**GloVe** 回答了这个问题：通过精心选择 loss function (损失函数) 的矩阵分解能够匹配或击败 Word2Vec，且训练成本更低。

第二，两种方法对从未见过的词都没有应对策略。`Zoomer-approved`、`dogecoin`、上周创造的任何专有名词、罕见词根的每一种屈折形式。**FastText** 通过嵌入字符 n-gram 修复了这一点：一个词是其部分之和，包括词素，因此即使是 OOV (out-of-vocabulary) 词也能获得合理的向量。

第三，transformer 出现后，问题再次转移。词级词汇表上限约一百万条目；真实语言比那更开放。**Byte-pair encoding (BPE)** 及其变体通过学习频繁子词单元的词汇表解决了这个问题，覆盖所有内容。每个现代 LLM 的 tokenizer 都是 subword tokenizer (子词分词器)。

本课涵盖以上三者，然后解释何时该选用哪一种。

## The Concept

**GloVe (Global Vectors).** 构建词-词共现矩阵 `X`，其中 `X[i][j]` 是词 `j` 出现在词 `i` 上下文中的次数。训练向量使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对 loss 加权，使高频词对不会主导训练。完成。

**FastText.** 一个词是其字符 n-gram 之和加上词本身。`where` 变成 `<wh, whe, her, ere, re>, <where>`。词向量是这些组成部分向量之和。像 Word2Vec 一样训练。好处：未见过的词（`whereupon`）可以从已知的 n-gram 组合而成。

**BPE (Byte-Pair Encoding).** 从单个字节（或字符）的词汇表开始。统计语料库中每个相邻词对的出现次数。将最频繁的词对合并为一个新 token。重复 `k` 次。结果：一个包含 `k + 256` 个 token 的词汇表，其中频繁序列（`ing`、`tion`、`the`）是单个 token，罕见词被拆成熟悉的片段。每个句子都能被分词成某种东西。

## Build It

### GloVe：分解共现矩阵

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个值得指出的细节。权重函数 `f(x) = (x/x_max)^alpha` 对非常频繁的词对（如 `(the, and)`）降权，使它们不会主导 loss。最终 embedding 是 `W`（中心）和 `W_tilde`（上下文）两个表之和。将两者相加是一个已发表的技巧，通常比只用一个表现更好。

### FastText：子词感知嵌入

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个词由其 n-gram 集合表示（通常是 3 到 6 个字符）。词 embedding 是其 n-gram embedding 之和。对于 skip-gram 训练，在 Word2Vec 使用单个向量的地方替换为这种表示。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于未见过的词，只要它的一些 n-gram 是已知的，你仍然能得到一个向量。`whereupon` 与 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，因此两者在空间中彼此靠近。

### BPE：学习的子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合并最常见的相邻词对。经过足够多次迭代后，频繁的子串（`low`、`est`、`tion`）成为单个 token，罕见词被干净地拆分。

真实的 GPT / BERT / T5 tokenizer 学习 30k-100k 次合并。结果：任何文本都能被分词成有限长度的已知 ID 序列，永远不会出现 OOV。

## Use It

实践中，你很少自己训练这些模型。你会加载预训练检查点。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

对于 transformer 时代的 BPE 风格子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前缀标记词边界（GPT-2 的约定）。每个现代 tokenizer 都是 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）。

### 何时选择哪一种

| 场景 | 选择 |
|-----------|------|
| 预训练通用词向量，不需要 OOV 容忍 | GloVe 300d |
| 预训练通用词向量，必须处理拼写错误 / 新造词 / 形态丰富语言 | FastText |
| 任何进入 transformer 的场景（训练或 inference） | 使用模型自带的 tokenizer。永远不要替换。 |
| 从零训练自己的语言模型 | 先在语料库上训练 BPE 或 SentencePiece tokenizer |
| 生产环境文本分类使用线性模型 | 仍然使用 TF-IDF。第 02 课。 |

## Ship It

保存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定任务和数据集描述，你输出：

1. Tokenization strategy (分词策略)（word-level、BPE、WordPiece、SentencePiece、byte-level）。一句话说明理由。
2. 词汇表大小目标（例如，仅英语 LM 用 32k，多语言用 64k-100k）。
3. 包含精确训练命令的库调用。指明库名。引用参数。
4. 一个可复现性陷阱。Tokenizer-model mismatch 是最常见的静默生产 bug；指出哪一对必须一起使用。

当用户 fine-tuning (微调) 预训练 LLM 时，拒绝推荐训练自定义 tokenizer。拒绝为任何面向生产 inference 的模型推荐 word-level tokenization。将非英语 / 多文字语料标记为需要带 byte fallback 的 SentencePiece。
```

## Exercises

1. **Easy.** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`。计算两个 n-gram 集合的 Jaccard 重叠。你应该看到大量共享片段（`pla`、`lay`、`play`），这就是 FastText 在形态变体间迁移良好的原因。
2. **Medium.** 扩展 `learn_bpe` 以追踪词汇表增长。绘制 tokens-per-corpus-character 随合并次数变化的函数图。你应该看到初期快速压缩，渐近趋近 ~2-3 字符每 token。
3. **Hard.** 在莎士比亚全集上训练 1k-merge BPE。比较常见词与罕见专有名词的分词结果。测量前后平均每词的 token 数。写下让你惊讶的发现。

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Co-occurrence matrix (共现矩阵) | 词-词频率表 | `X[i][j]` = 词 `j` 出现在词 `i` 窗口中的次数。 |
| Subword (子词) | 词的片段 | 字符 n-gram（FastText）或学习的 token（BPE/WordPiece/SentencePiece）。 |
| BPE | Byte-pair encoding | 迭代合并最频繁的相邻词对，直到词汇表达到目标大小。 |
| OOV (out-of-vocabulary) | 词汇表外 | 模型从未见过的词。Word2Vec/GloVe 失败。FastText 和 BPE 能处理。 |
| Byte-level BPE | 原始字节上的 BPE | GPT-2 的方案。词汇表从 256 字节开始，因此永远不会出现 OOV。 |

## Further Reading

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) —— GloVe 论文，七页，仍然是 loss 推导的最佳来源。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) —— FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) —— 将 BPE 引入现代 NLP 的论文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) —— BPE、WordPiece 和 SentencePiece 在实际中的差异。
