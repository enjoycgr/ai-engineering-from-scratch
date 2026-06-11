# Bag of Words、TF-IDF 与文本表示

> 先计数，再思考。在定义明确的任务上，TF-IDF 在 2026 年仍然能击败 embedding。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 分钟

## 问题所在

模型需要数字。你有字符串。

每个 NLP 流程都必须回答同一个问题。如何将可变长度的 token 流转换为分类器可以消费的固定大小向量。该领域找到的第一个答案是最笨但有效的那个：数单词。组成向量。

这个向量支撑的生产级 NLP 比任何 embedding 模型都多。垃圾邮件过滤器、主题分类器、日志异常检测、搜索排序（BM25 之前）、第一波情感分析、第一个十年的学术 NLP 基准测试。2026 年的从业者在狭窄分类任务上仍然首先想到它。它快速、可解释，而且在单词存在性就是关键的任务上，往往与 4 亿参数的 embedding 模型无法区分。

本节课从零构建 bag of words，然后是 TF-IDF。然后展示 scikit-learn 用三行代码完成同样的事。最后指出让你转向 embedding 的失效模式。

## 核心概念

**Bag of Words (BoW)** 丢弃顺序。对于每个文档，统计每个词汇单词出现的次数。向量长度是词汇表大小。位置 `i` 是单词 `i` 的计数。

**TF-IDF** 对 BoW 重新加权。在每个文档中都出现的单词没有信息量，所以缩小其权重。在语料库中罕见但在单个文档中频繁的单词是信号，所以放大其权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中 `TF` 是文档中的词频，`df` 是文档频率（包含该单词的文档数），`N` 是总文档数。`log` 使 ubiquitous 词的权重有界。

关键属性：两者都产生具有可解释轴的稀疏向量。你可以查看训练好的分类器的权重，读出哪些单词推动文档朝向每个类别。你无法对 768 维的 BERT embedding 做到这一点。

## 动手实现

### 第一步：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何词级分词器都可以；本节课的 `code/main.py` 使用简化的小写变体）。输出：`{word: index}` 字典。稳定的插入顺序意味着单词索引 0 是第一个文档中看到的第一个单词。约定各不相同；scikit-learn 按字母顺序排序。

### 第二步：bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行是文档。列是词汇索引。条目 `[i][j]` 是"单词 `j` 在文档 `i` 中出现多少次"。文档 1 有 `cat` 两次，因为它确实如此。文档 0 有 `ran` 零次，因为它没有。

### 第三步：词频和文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

两个值得指出的平滑技巧。`(n+1)/(d+1)` 避免了 `log(x/0)`。末尾的 `+1` 确保在每个文档中都出现的单词仍然有 IDF 1（而不是 0），与 scikit-learn 的默认值匹配。其他实现使用原始 `log(N/df)`。两者都有效；平滑版本更友好。

### 第四步：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三个文档，五个词汇词（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三个中，所以其 IDF 低。`dog` 出现在一个中，所以其 IDF 高。向量是稀疏的（大多数条目很小），有区分度的单词凸显出来。

### 第五步：L2 归一化行

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

没有归一化，更长的文档会得到更大的向量并在相似度分数中占主导。L2 归一化将每个文档放到单位超球面上。行之间的 cosine similarity 现在就是点积。

## 使用现有工具

scikit-learn 提供了生产级版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 在一个调用中完成分词、词汇表构建和 BoW。`TfidfVectorizer` 添加 IDF 加权和 L2 归一化。两者都返回稀疏矩阵。对于 10 万文档，密集版本无法装入内存；在分类器需要密集矩阵之前保持稀疏。

改变一切的参数：

| 参数 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 包含 bigram。通常提升分类效果。 |
| `min_df=2` | 丢弃在少于 2 个文档中出现的词。在噪声数据上修剪词汇表。 |
| `max_df=0.95` | 丢弃在超过 95% 文档中出现的词。近似停用词移除，无需硬编码列表。 |
| `stop_words="english"` | scikit-learn 内置的停用词列表。任务相关 —— 情感分析不应该丢弃否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 代替原始 `tf`。帮助当一个词在单个文档中重复多次时。 |

### TF-IDF 仍然获胜的场景（截至 2026 年）

- 垃圾邮件检测、主题标注、日志异常标记。单词存在性就是关键；语义细微差别不重要。
- 低数据场景（数百个标注样本）。TF-IDF 加 logistic regression 没有预训练成本。
- 任何 latency 重要的地方。TF-IDF 加线性模型在微秒内回答。通过 transformer embedding 一个文档需要 10-100 毫秒。
- 必须解释其预测的系统。检查分类器的系数。最正面的单词就是原因。

### TF-IDF 失效的场景

语义盲失效。考虑这两个文档：

- "The movie was not good at all."
- "The movie was excellent."

一个是负面评价。一个是正面。它们的 TF-IDF 重叠恰好是 `{the, movie, was}`。bag-of-words 分类器必须记住 `not` 在 `good` 附近会翻转标签。它可以在足够数据上学会这一点，但永远不如理解句法的模型优雅。

另一个失效：推理时的 out-of-vocabulary 词。在 IMDb 评论上训练的 BoW 模型不知道如何处理 `Zoomer-approved`，如果该 token 从未在训练中出现。Subword embedding（第 04 课）处理这个。TF-IDF 不能。

### 混合方案：TF-IDF 加权 embedding

2026 年中等数据分类的务实默认方案：使用 TF-IDF 权重作为 word embedding 的注意力。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你从 embedding 获得语义能力，从 TF-IDF 获得罕见词强调。分类器在池化向量上训练。在情感、主题和意图分类上，对于低于约 5 万标注样本的情况，这优于单独使用任何一种方法。

## 交付物

保存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: 给定文本分类任务，推荐 BoW、TF-IDF、embedding 或混合方案。
phase: 5
lesson: 02
---

你推荐文本向量化策略。给定任务描述，输出：

1. 表示方法（BoW、TF-IDF、transformer embedding 或混合方案）。用一句话解释原因。
2. 具体的 vectorizer 配置。命名库。引用参数（`ngram_range`、`min_df`、`max_df`、`sublinear_tf`、`stop_words`）。
3. 交付前应测试的一个失效模式。

当用户有少于 500 个标注样本时，拒绝推荐 embedding，除非他们展示了 TF-IDF 基线上的语义失效证据。拒绝为情感分析移除停用词（否定词携带信号）。标记类别不平衡需要的不只是 vectorizer 的改变。

示例输入："将 3 万条客户支持工单分类到 12 个类别。大多数工单 2-3 句话。仅英语。需要为审计日志提供可解释性。"

示例输出：

- 表示方法：TF-IDF。3 万样本不算小；可解释性要求排除了密集 embedding。
- 配置：`TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`。保留停用词，因为类别关键词有时就是停用词（"not working" vs "working"）。
- 需测试的失效：验证 `min_df=3` 不会丢弃罕见的类别关键词。按类别过滤 `get_feature_names_out` 并肉眼检查。
```

## 练习

1. **简单。** 在 L2 归一化的 TF-IDF 输出上实现 `cosine_similarity(doc_vec_a, doc_vec_b)`。验证相同文档得分为 1.0，词汇完全不重叠的文档得分为 0.0。
2. **中等。** 为 `bag_of_words` 添加 `n-gram` 支持。参数 `n` 产生 `n`-gram 的计数。测试 `n=2` 在 `["the", "cat", "sat"]` 上产生 `["the cat", "cat sat"]` 的 bigram 计数。
3. **困难。** 使用 GloVe 100d 向量构建上述 TF-IDF 加权 embedding 混合方案（下载一次，缓存）。在 20 Newsgroups 数据集上比较与纯 TF-IDF 和纯 mean-pooled embedding 的分类准确率。报告哪种在何处获胜。

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|-------------|-----------------------|
| BoW | 词频向量 | 单个文档中词汇单词的计数。丢弃顺序。 |
| TF | 词频 (Term frequency) | 一个单词在文档中的计数，可选择按文档长度归一化。 |
| DF | 文档频率 (Document frequency) | 至少包含该单词一次的文档计数。 |
| IDF | 逆文档频率 (Inverse document frequency) | `log(N / df)` 平滑后。降低 ubiquitous 词的权重。 |
| Sparse vector | 大部分是零 | 词汇表通常为 1 万-10 万词；大多数不会出现在任何给定文档中。 |
| Cosine similarity | 向量夹角 | L2 归一化向量的点积。1 表示相同，0 表示正交。 |

## 延伸阅读

- [scikit-learn —— 文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— 权威 API 参考，加上每个参数的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) —— 让 TF-IDF 成为十年默认方法的论文。
- ["Why TF-IDF Still Beats Embeddings" —— Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) —— 2026 年关于旧方法何时获胜以及为什么的见解。
