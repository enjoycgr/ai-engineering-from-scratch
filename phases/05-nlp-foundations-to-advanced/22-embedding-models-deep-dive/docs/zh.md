# Embedding Models —— 2026 深度解析

> Word2Vec 为每个词给出一个向量。现代 embedding（嵌入）模型为每个段落给出一个向量，支持跨语言，并提供稀疏、密集和多向量视图，尺寸可根据索引需求调整。选错了，你的 RAG 就会检索到错误的内容。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 分钟

## 问题背景

你的 RAG 系统 40% 的时间检索到了错误的段落。罪魁祸首很少是向量数据库或 prompt。而是 embedding 模型。

在 2026 年选择 embedding 意味着在五个维度上做选择：

1. **Dense vs sparse vs multi-vector（多向量）。** 每个段落一个向量，或每个 token 一个向量，或一个稀疏的加权词袋。
2. **语言覆盖。** 单语英语模型在纯英语任务上仍然领先。多语言模型在语料混合时胜出。
3. **上下文长度。** 512 tokens vs 8,192 vs 32,768 —— 实际有效容量通常只有标称最大值的 60-70%。
4. **维度预算。** 3,072 维全精度浮点数 = 每个向量 12 KB。1 亿向量时，存储成本为 $1,300/月。Matryoshka（俄罗斯套娃式）截断可将此成本降低 4 倍。
5. **开源 vs 托管。** 开源权重意味着你控制整个技术栈和数据。托管意味着用控制权换取始终最新的模型。

本课程将阐明这些权衡，让你能够基于证据而非上季度的流行度来做选择。

## 核心概念

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embeddings（密集嵌入）。** 每个段落一个向量（通常 384-3,072 维）。Cosine similarity（余弦相似度）按语义接近度对段落排序。OpenAI `text-embedding-3-large`、BGE-M3 dense 模式、Voyage-3。默认选择。

**Sparse embeddings（稀疏嵌入）。** SPLADE 风格。Transformer 为每个词表 token 预测一个权重，然后将大部分置零。结果是一个大小为 |vocab| 的稀疏向量。捕获词汇匹配（类似 BM25）但具有学习的词权重。在关键词密集型查询上表现强劲。

**Multi-vector (late interaction)（多向量 / 迟交互）。** ColBERTv2、Jina-ColBERT。每个 token 一个向量。使用 MaxSim 评分：对每个查询 token，找到最相似的文档 token，累加分数。存储和评分更昂贵，但在长查询和领域特定语料上表现更好。

**BGE-M3：三者兼备。** 单个模型同时输出 dense、sparse 和 multi-vector 表示。每种都可以独立查询；分数通过加权求和融合。当你希望从一个 checkpoint 获得灵活性时，这是 2026 年的默认选择。

**Matryoshka Representation Learning（俄罗斯套娃表示学习）。** 训练目标是使向量的前 N 维本身构成一个有用的独立 embedding。将 1,536 维向量截断到 256 维，以约 1% 的准确率损失换取 6 倍存储节省。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5+ 均支持。

### MTEB 排行榜只讲了一部分故事

Massive Text Embedding Benchmark（大规模文本嵌入基准）—— 发布时（2022）涵盖 8 种任务类型的 56 个任务，在 MTEB v2 中扩展到 100+ 任务。2026 年初，Gemini Embedding 2 在检索任务上领先（67.71 MTEB-R）。Cohere embed-v4 在通用任务上领先（65.2 MTEB）。BGE-M3 在开源多语言模型中领先（63.0）。排行榜是必要的但不充分的——始终要在你的领域上做基准测试。

### 三层模式

| 使用场景 | 模式 |
|----------|------|
| 快速初筛 | Dense bi-encoder（BGE-M3, text-3-small） |
| 召回提升 | Sparse（SPLADE, BGE-M3 sparse）+ RRF 融合 |
| Top-50 精确度 | Multi-vector（ColBERTv2）或 cross-encoder reranker |

大多数生产环境技术栈会同时使用这三种。

## 动手实践

### 步骤 1：基线 —— 使用 Sentence-BERT 的 dense embeddings

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 使 dot product（点积）等于 cosine similarity（余弦相似度）。务必开启。

### 步骤 2：Matryoshka 截断

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截断后重新归一化。Nomic v1.5、OpenAI text-3 和 Voyage-4 经过训练，使得前几层截断几乎无损。非 Matryoshka 模型（原始 Sentence-BERT）截断后会急剧退化。

### 步骤 3：BGE-M3 多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

一次 inference（推理），三个索引。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的领域上调优权重。

### 步骤 4：在自定义任务上进行 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*代表性*子集上运行候选模型。不要只信任排行榜排名——你的领域很重要。

### 步骤 5：从零手写 cosine

参见 `code/main.py`。平均 Hashing Trick embeddings（纯标准库）。无法与 transformer embeddings 竞争，但展示了基本形态：tokenize → vector → normalize → dot product。

## 常见陷阱

- **查询和文档使用同一模型。** 某些模型（Voyage, Jina-ColBERT）使用 asymmetric encoding（非对称编码）—— 查询和文档经过不同的路径。务必查看模型卡。
- **遗漏前缀。** `bge-*` 模型需要在查询前添加 `"Represent this sentence for searching relevant passages: "`。遗漏会导致 3-5 个点的 recall（召回率）差距。
- **过度截断 Matryoshka。** 1,536 → 256 通常是安全的。1,536 → 64 则不是。在你的评估集上验证。
- **上下文截断。** 大多数模型会静默截断超过最大长度的输入。长文档需要分块（参见第 23 课）。
- **忽略延迟尾部。** MTEB 分数隐藏了 p99 latency（延迟）。一个 600M 模型可能比 335M 模型高 2 分，但每查询成本高出 3 倍。

## 实际应用

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 仅英语、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、英语 | `BAAI/bge-large-en-v1.5` |
| 开源权重、多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文 (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| 仅 CPU 部署 | Nomic Embed v2 (137M 参数, MoE) |
| 存储受限 | Matryoshka-truncated + int8 quantization |
| 关键词密集型查询 | 添加 SPLADE sparse，通过 RRF 与 dense 融合 |

2026 年模式：从 BGE-M3 或 text-3-large 开始，用 MTEB 在你的领域上评估，如果某个领域特定模型领先超过 3 分，则替换。

## 交付物

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习

1. **简单。** 用 `bge-small-en-v1.5` 以全维度 (384) 编码 100 个句子，然后以 Matryoshka 128 编码。在 10 个查询上测量 MRR 下降。
2. **中等。** 在你的领域 500 个段落上比较 BGE-M3 dense、sparse 和 colbert。哪个在 recall@10 上胜出？RRF 融合是否击败最佳单一模式？
3. **困难。** 在你的前 2 个领域任务上，对三个候选模型运行 MTEB。报告 MTEB 分数、100 查询批次的 p99 latency、以及 $/1M queries。选择 Pareto-optimal（帕累托最优）的模型。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Dense embedding | 向量 | 每个文本一个固定大小的向量。用 cosine similarity（余弦相似度）排序。 |
| Sparse embedding | 学习的 BM25 | 每个词表 token 一个权重；大部分为零；端到端训练。 |
| Multi-vector | ColBERT 风格 | 每个 token 一个向量；MaxSim 评分；索引更大，recall（召回率）更好。 |
| Matryoshka | 俄罗斯套娃技巧 | 前 N 维本身构成一个有效的更小 embedding。 |
| MTEB | 基准 | Massive Text Embedding Benchmark —— 发布时 56 个任务，v2 中 100+。 |
| BEIR | 检索基准 | 18 个零样本检索任务；常被引用以证明跨领域鲁棒性。 |
| Asymmetric encoding | Query ≠ doc path | 模型对查询和文档使用不同的投影。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) —— bi-encoder 论文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) —— 排行榜论文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) —— 统一三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) —— 维度阶梯训练目标。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) —— 生产环境中的迟交互。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) —— 实时排名。
