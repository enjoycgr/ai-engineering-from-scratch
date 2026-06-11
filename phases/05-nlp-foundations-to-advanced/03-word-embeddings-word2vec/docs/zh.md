# 词嵌入 —— 从零实现 Word2Vec

> 一个词由它的邻居定义。在一个浅层网络上训练这个想法，几何结构就会自然涌现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 分钟

## The Problem

TF-IDF 知道 `dog` 和 `puppy` 是不同的词，但它不知道它们意思几乎相同。一个在 `dog` 上训练好的分类器无法泛化到关于 `puppy` 的评论。你可以通过列出同义词来掩盖这个问题，但这在罕见词、领域术语以及每一种你未预料到的语言上都会失败。

你想要一种表示方式，让 `dog` 和 `puppy` 在空间中彼此靠近。让 `king - man + woman` 落在 `queen` 附近。让一个在 `dog` 上训练的模型能免费将部分信号迁移到 `puppy` 上。

Word2Vec 给了我们这样的空间。两层神经网络，万亿 token 的训练规模，2013 年发表。架构简单到几乎令人尴尬。结果却重塑了 NLP 长达十年。

## The Concept

**Distributional hypothesis (分布假说)**（Firth, 1957）："You shall know a word by the company it keeps." 如果两个词出现在相似的上下文中，它们可能意思相近。

Word2Vec 有两种形式，都利用了上述思想。

- **Skip-gram.** 给定中心词，预测周围的词。`cat -> (the, sat, on)`，窗口大小为 2。
- **CBOW (continuous bag of words).** 给定周围的词，预测中心词。`(the, sat, on) -> cat`。

Skip-gram 训练更慢，但对罕见词效果更好。它成为了默认选择。

网络有一个隐藏层，没有非线性。输入是词汇表上的 one-hot 向量。输出是词汇表上的 softmax。训练完成后，丢弃输出层。隐藏层的权重就是 embedding (嵌入)。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

关键技巧：对 10 万词做 softmax 计算量巨大到无法承受。Word2Vec 使用 **negative sampling (负采样)** 将其转化为二分类任务：预测"这个上下文词是否出现在该中心词附近，是或否"。对每个正样本对，采样少量负样本（非共现词），而不是在整个词汇表上计算 softmax。

## Build It

### Step 1: 从语料库生成训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口内的每一个 (center, context) 对都是一个正训练样本。

### Step 2: 嵌入表

两个矩阵。`W` 是中心词 embedding 表（保留的那个）。`W'` 是上下文词表（通常被丢弃，有时与 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小的随机初始化。词汇量 10k、维度 100 是真实场景的规模；教学演示用 50 词 x 16 维就足以观察几何结构。

### Step 3: 负采样目标函数

对每个正样本对 `(center, context)`，从词汇表中采样 `k` 个随机词作为负样本。训练模型使得正样本的 `W[center] · W'[context]` 点积高，负样本的点积低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

核心公式：正样本对上的 logistic loss（希望 sigmoid 接近 1）加上负样本对上的 logistic loss（希望 sigmoid 接近 0）。梯度同时流向两个表。完整推导见原始论文；如果你想真正掌握，建议用纸笔亲自推导一遍。

### Step 4: 在玩具语料库上训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

经过足够多的 epoch 和大型语料库训练后，共享上下文的词会具有相似的中心 embedding。在玩具语料库上效果微弱，在数十亿 token 上效果惊人。

### Step 5: 类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在预训练的 300 维 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因为模型知道什么是王室，而是因为向量 `(king - man)` 捕获了类似"royal"的方向，将其加到 `woman` 上就会落在 royal-female 区域附近。

## Use It

从零写 Word2Vec 是为了教学。生产 NLP 使用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

实际工作中，你几乎不会自己训练 Word2Vec。你会下载预训练向量。

- **GloVe** —— Stanford 的共现矩阵分解方法。50d、100d、200d、300d 检查点。通用覆盖好。第 04 课专门讲 GloVe。
- **fastText** —— Facebook 对 Word2Vec 的扩展，嵌入字符 n-gram。通过组合子词来处理 OOV 词。第 04 课。
- **Pretrained Word2Vec on Google News** —— 300d，300 万词词汇表，2013 年发布。至今仍被每天下载。

### 2026 年 Word2Vec 仍然胜出的场景

- 轻量级领域特定检索。在笔记本电脑上花一小时在医学摘要上训练，获得通用模型无法捕获的专业向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。从其他词中减去它，得到性别中性轴。仍在公平性研究中使用。
- 可解释性。100 维足够小，可以通过 PCA 或 t-SNE 可视化并实际观察聚类形成。
- 任何必须在无 GPU 设备上运行 inference (推理) 的场景。Word2Vec 的 lookup 只是一次行读取。

### Word2Vec 的失败之处

多义词壁垒。`bank` 只有一个向量。`river bank` 和 `financial bank` 共享它。`table`（电子表格 vs. 家具）共享它。下游分类器无法从向量中区分不同含义。

Contextual embeddings (上下文嵌入)（ELMo、BERT 及之后的所有 transformer）通过根据周围上下文为每个词的出现生成不同向量来解决这个问题。这就是从 Word2Vec 到 BERT 的飞跃：从静态到上下文。第 7 阶段涵盖 transformer 部分。

OOV (out-of-vocabulary) 问题是另一个失败点。如果 `Zoomer-approved` 不在训练数据中，Word2Vec 从未见过它，没有回退机制。fastText 通过子词组合修复了这一点（第 04 课）。

## Ship It

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

你探查训练好的词嵌入以验证它们是否正常工作。给定一个 `gensim.models.KeyedVectors` 对象和词汇表，你运行：

1. 三个经典类比测试。`king : man :: queen : woman`。`paris : france :: tokyo : japan`。`walking : walked :: swimming : ?`。报告 top-1 结果及其余弦值。
2. 五个用户提供的领域特定词的最近邻测试。打印 top-5 邻居及其余弦值。
3. 一个对称性检查。`similarity(a, b) == similarity(b, a)` 在浮点精度范围内成立。
4. 一个退化检查。如果任何 embedding 的范数低于 0.01 或高于 100，说明模型存在训练 bug。标记它。

不要仅凭类比准确率就判定模型是好的。类比基准是可操纵的，无法迁移到下游任务。建议同时使用 intrinsic + downstream evaluation (内在评估 + 下游评估)。
```

## Exercises

1. **Easy.** 在微型语料库（20 句关于猫和狗的句子）上运行训练循环。200 个 epoch 后，验证 `nearest(vocab, W, W[vocab["cat"]])` 的 top 3 中是否包含 `dog`。如果没有，增加 epoch 或词汇量。
2. **Medium.** 添加高频词的 subsampling (子采样)。频率高于 `10^-5` 的词以与其频率成正比的概率从训练对中丢弃。测量对罕见词相似度的影响。
3. **Hard.** 在 20 Newsgroups 语料库上训练模型。计算两个偏置轴：`he - she` 和 `doctor - nurse`。将职业词投影到两个轴上。报告哪些职业具有最大的偏置差距。这是公平性研究者使用的探针类型。

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Word embedding (词嵌入) | 词作为向量 | 一种从上下文中学习到的密集、低维（通常 100-300 维）表示。 |
| Skip-gram | Word2Vec 技巧 | 从中心词预测上下文词。比 CBOW 慢，但对罕见词更好。 |
| Negative sampling (负采样) | 训练捷径 | 用对 `k` 个随机词的二分类替代对整个词汇表的 softmax。 |
| Static embedding (静态嵌入) | 一词一向量 | 无论上下文如何，同一个词总是同一个向量。在多义词上失败。 |
| Contextual embedding (上下文嵌入) | 上下文敏感向量 | 根据周围词为每个出现生成不同向量。这是 transformer 的输出。 |
| OOV (out-of-vocabulary) | 词汇表外 | 训练时未见过的词。Word2Vec 无法为它们生成向量。 |

## Further Reading

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) —— 负采样论文。简短易读。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) —— 最清晰的梯度推导，如果原始论文的数学让你感到密集。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) —— 生产环境中真正有效的训练设置。
