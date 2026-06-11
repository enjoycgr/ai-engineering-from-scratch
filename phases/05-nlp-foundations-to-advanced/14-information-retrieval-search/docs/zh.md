# 信息检索与搜索 (Information Retrieval and Search)

> BM25 精确但脆弱。Dense retrieval（稠密检索）撒网广但会漏掉关键词。Hybrid（混合检索）是 2026 年的默认方案。其余都是调优。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 分钟

## 问题 (The Problem)

用户输入 "what happens if someone lies to get money"，期望找到真正涵盖此问题的法条："Section 420 IPC." 关键词搜索完全错过（没有共享词汇）。如果 embedding（嵌入 / 词嵌入）未在法律文本上训练，语义搜索也会错过。真正的搜索必须同时处理两者。

IR（信息检索）是每个 RAG 系统、每个搜索栏、每个文档站点模糊查找背后的流水线。2026 年能在生产环境落地的架构不是单一方法，而是一系列互补方法的链条，每个方法都能捕捉前一个方法的失败。

本课构建每个部分，并指出每种方法捕捉了哪些失败。

## 概念 (The Concept)

![混合检索：BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四层。按需选取。

1. **Sparse retrieval（稀疏检索）(BM25)。** 快速，精确匹配表现好，语义方面差。在倒排索引 (inverted index) 上运行。百万级文档上每次查询低于 10 毫秒。能正确找到法条引用、产品代码、错误信息、命名实体。
2. **Dense retrieval（稠密检索）。** 将查询和文档编码为向量。最近邻搜索。捕捉改写 (paraphrase) 和语义相似性。会漏掉仅差一个字符的精确关键词匹配。使用 FAISS 或向量数据库时每次查询 50-200 毫秒。
3. **Fusion（融合）。** 合并 sparse 和 dense 的排序列表。Reciprocal Rank Fusion (RRF) 是简单的默认选择，因为它忽略原始分数（处于不同尺度）而仅使用排名位置。当你知道某个信号在特定领域占主导时，加权融合 (weighted fusion) 也是一种选择。
4. **Cross-encoder rerank（交叉编码器重排）。** 从融合结果中取 top-30。运行 cross-encoder（将 query + document 一起输入，为每对打分）。保留 top-5。Cross-encoder 比 bi-encoder 每对更慢，但精确得多。你只需在 top-30 上运行它们，从而摊平成本。

三路检索 (BM25 + dense + learned-sparse 如 SPLADE) 在 2026 年基准测试中优于两路，但需要 learned-sparse 索引的基础设施。对于大多数团队，两路 + cross-encoder rerank 是最佳平衡点。

## 动手构建 (Build It)

### 步骤 1：从零实现 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

两个值得了解的参数。`k1=1.5` 控制词频 (term-frequency) 饱和度；越高意味着对词重复赋予更大权重。`b=0.75` 控制长度归一化 (length normalization)；0 忽略文档长度，1 完全归一化。默认值来自原始论文中 Robertson 的建议，很少需要调整。

### 步骤 2：使用 bi-encoder（双编码器）进行 dense retrieval

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对 embedding（嵌入 / 词嵌入）进行 L2 归一化，使点积等于余弦相似度。`all-MiniLM-L6-v2` 是 384 维，速度快，对大多数英文检索足够强。多语言工作请用 `paraphrase-multilingual-MiniLM-L12-v2`。追求最高精度请用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：Reciprocal Rank Fusion（倒数排名融合）

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 常数来自原始 RRF 论文。更高的 `k` 会拉平排名差异的贡献；更低的 `k` 使高排名主导。60 是发表的默认值，很少需要调整。

### 步骤 4：混合搜索 + 重排

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三个阶段组合。BM25 找到词汇匹配。Dense 找到语义匹配。RRF 合并两个排序，无需分数校准。Cross-encoder 使用 query-document 对一起对 top-30 重新打分，捕捉到 bi-encoder 遗漏的细粒度相关性。保留 top-5。

### 步骤 5：评估

| 指标 | 含义 |
|--------|---------|
| Recall@k | 在正确答案存在的查询中，有多少比例能在 top-k 中找到？ |
| MRR (Mean Reciprocal Rank) | 首个相关文档排名的倒数平均值。 |
| nDCG@k | 考虑相关性分级，而非仅二元的相关/不相关。 |

对于 RAG 来说，检索器的 **Recall@k** 是最重要的数字。如果正确的段落不在检索集合中，你的阅读器就无法回答。

调试技巧：对于失败的查询，对比 sparse 和 dense 的排序结果。如果其中一个找到了正确文档而另一个没有，说明存在词汇不匹配（修复：补上缺失的一半）或语义歧义（修复：更好的 embedding 或重排器）。

## 使用 (Use It)

2026 年技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k 文档 | 内存 BM25 + `all-MiniLM-L6-v2` embedding + RRF。无需单独数据库。 |
| 100k-10M 文档 | FAISS 或 pgvector 用于 dense + Elasticsearch / OpenSearch 用于 BM25。并行运行。 |
| 10M+ 文档 | Qdrant / Weaviate / Vespa / Milvus 支持混合检索。Cross-encoder 对 top-30 重排。 |
| 最佳质量前沿 | 三路 (BM25 + dense + SPLADE) + ColBERT late-interaction reranking |

无论选择什么，都要为评估预留预算。在基准测试端到端 RAG 准确率之前，先基准测试检索召回率。阅读器无法修复检索器遗漏的内容。

### 2026 年生产 RAG 的宝贵经验

- **80% 的 RAG 失败源于摄入 (ingestion) 和分块 (chunking)，而非模型。** 团队花数周时间更换 LLM 和调整提示词，而检索器每三个查询就悄悄返回错误上下文。先修复分块。
- **分块策略比分块大小更重要。** 固定大小切分会破坏表格、代码和嵌套标题。句子感知 (sentence-aware) 是默认选择；语义分块或基于 LLM 的分块对技术文档和产品手册有回报。
- **Parent-doc 模式。** 检索小的 "子" 片段以获得精确度。当同一父节的多个子片段出现时，替换为父块以保留上下文。这能在不重新训练的情况下持续提升答案质量。
- **k_rerank=3 通常最优。** 超过这个数量的每个额外片段都会增加 token 成本和生成延迟，而不会提升答案质量。如果 k=8 对你来说仍然比 k=3 更好，说明重排器表现不佳。
- **HyDE / 查询扩展 (query expansion)。** 从查询生成假设答案，嵌入该答案，然后检索。弥合短问题与长文档之间的措辞差距。无需训练即可免费提升精确度。
- **上下文预算低于 8K token。** 持续达到该限制意味着重排器阈值太宽松。
- **版本化一切。** 提示词、分块规则、embedding 模型、重排器。任何漂移都会悄悄破坏答案质量。在忠实度 (faithfulness)、上下文精确度和未回答问题率上的 CI 门禁 (CI gates) 能在用户看到之前阻止退化。
- **三路检索 (BM25 + dense + learned-sparse 如 SPLADE) 优于两路** 在 2026 年基准测试中，尤其对混合专有名词和语义的查询。当基础设施支持 SPLADE 索引时，请部署它。

根据 2026 年行业测量，合理的检索设计可将幻觉率降低 70-90%。大多数 RAG 性能提升来自更好的检索，而非模型微调 (fine-tuning)。

## 交付 (Ship It)

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习 (Exercises)

1. **简单。** 在 500 份文档的语料库上实现上述 `hybrid_search`。测试 20 个查询。比较 BM25-only、dense-only 和 hybrid 的 recall@5。
2. **中等。** 添加 MRR 计算。对于每个已知正确文档的测试查询，找出该文档在 BM25、dense 和 hybrid 排序中的排名。报告各自的 MRR。
3. **困难。** 使用 MultipleNegativesRankingLoss（Sentence Transformers）在您的领域上微调 (fine-tune) 一个 dense encoder。从 500 个 query-document 对构建训练集。比较微调前后的召回率。

## 关键术语 (Key Terms)

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BM25 | 关键词搜索 | Okapi BM25。通过词频 (term frequency)、IDF 和长度对文档打分。 |
| Dense retrieval | 向量搜索 | 将 query + doc 编码为向量，找最近邻。 |
| Bi-encoder | Embedding 模型 | 独立编码 query 和 doc。查询时速度快。 |
| Cross-encoder | 重排器模型 | 将 query + doc 一起编码。慢但精确。 |
| RRF | 排名融合 | 通过求和 `1/(k + rank)` 合并两个排序。 |
| Recall@k | 检索指标 | 相关文档在 top-k 中的查询比例。 |

## 延伸阅读 (Further Reading)

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — 最权威的 BM25 论述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，经典 bi-encoder。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 缩小与 dense 差距的 learned-sparse 检索器。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 论文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — late-interaction 检索。
