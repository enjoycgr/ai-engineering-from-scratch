# 高级 RAG（Advanced RAG）：分块、重排序、混合搜索

> 基本 RAG 检索最相似的 top-k 个 chunk。这对简单问题有效。对于多跳推理、模糊查询和大型语料库，它就崩溃了。高级 RAG 是 demo 在 10 份文档上有效与系统在 1000 万份文档上有效之间的区别。

**类型：** Build
**语言：** Python
**前置知识：** Phase 11，第 06 课（RAG）
**时间：** ~90 分钟
**相关：** Phase 5 · 23（RAG 的分块策略）涵盖所有六种分块算法 —— 递归、语义、句子、父文档、延迟分块、上下文检索 —— 以及 Vectara/Anthropic 基准测试。本课在此基础上构建：混合搜索、重排序、查询转换。

## 学习目标

- 实现高级分块策略（语义、递归、父子）以保留文档结构和上下文
- 构建结合 BM25 关键词匹配与语义向量搜索的混合搜索流水线，并使用 cross-encoder 重排序器
- 应用查询转换技术（HyDE、多查询、step-back）以改善模糊或复杂问题的检索效果
- 诊断和修复常见 RAG 故障：检索到错误 chunk、答案不在上下文中、多跳推理崩溃

## 问题背景

你在第 06 课构建了一个基本 RAG 流水线。它对小型语料库上的直接问题有效。现在试试这些：

**模糊查询**："What was revenue last quarter?" 语义搜索返回关于 revenue strategy、revenue projections 和 CFO 对 revenue growth 想法的 chunk。所有都与"revenue"一词语义相似。没有一个包含实际数字。正确的 chunk 说"$47.2M in Q3 2025"但使用了"earnings"而不是"revenue"。嵌入模型认为"revenue strategy"比"Q3 earnings were $47.2M"更接近查询。

**多跳问题**："Which team had the highest customer satisfaction score improvement?" 这需要找到每个团队的满意度分数、比较它们，并确定最大值。没有单个 chunk 包含答案。信息分散在团队报告中。

**大型语料库问题**：你有 200 万个 chunk。正确答案在 chunk #1,847,293。你的 top-5 检索拉取了 chunk #14、#89,201、#1,200,000、#44 和 #901,333。在嵌入空间中接近，但没有一个包含答案。在这个规模下，近似最近邻搜索引入的误差足以将相关结果推出 top-k。

基本 RAG 失败，因为向量相似度不等于相关性。一个 chunk 可以在语义上与查询相似，但对回答它没有用处。高级 RAG 通过四种技术解决这个问题：混合搜索（添加关键词匹配）、重排序（更仔细地评分候选）、查询转换（搜索前修复查询）和更好的分块（以正确的粒度检索）。

## 核心概念

### 混合搜索：语义 + 关键词

语义搜索（向量相似度）擅长理解含义。"How do I cancel my subscription?" 匹配 "Steps to terminate your plan"，即使它们没有共享任何词。但它错过精确匹配。"Error code E-4021" 可能不会匹配包含 "E-4021" 的 chunk，如果嵌入模型将其视为噪声。

关键词搜索（BM25）正好相反。它擅长精确匹配。"E-4021" 完美匹配。但 "cancel my subscription" 如果文档说 "terminate your plan" 则返回零结果。

混合搜索同时运行两者，然后合并结果。

**BM25**（Best Matching 25）是标准的关键词搜索算法。自 1990 年代以来一直是搜索引擎的支柱。公式：

```
BM25(q, d) = sum over terms t in q:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中 tf(t,d) 是词 t 在文档 d 中的词频，IDF(t) 是逆文档频率，|d| 是文档长度，avgdl 是平均文档长度，k1 控制词频饱和度（默认 1.2），b 控制长度归一化（默认 0.75）。

简单来说：BM25 对包含查询词（尤其是稀有词）的文档评分更高，但重复词的收益递减。一个包含 "revenue" 50 次的文档并不比包含一次的文档相关 50 倍。

### 倒数排名融合（Reciprocal Rank Fusion, RRF）

你有两个排名列表：一个来自向量搜索，一个来自 BM25。如何合并它们？倒数排名融合是标准方法。

```
RRF_score(d) = sum over rankings R:
    1 / (k + rank_R(d))
```

其中 k 是一个常数（通常为 60），防止排名最高的结果主导。

一个在向量搜索中排名第 1、在 BM25 中排名第 5 的文档获得：1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

一个在向量搜索中排名第 3、在 BM25 中排名第 2 的文档获得：1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

RRF 自然地平衡两个信号。在两个列表中都排名高的文档获得最佳分数。在一个列表中排名第 1 但在另一个列表中不存在的文档获得中等分数。这很稳健，因为它使用排名而非原始分数，所以两个系统之间分数分布的差异无关紧要。

### 重排序（Reranking）

检索（无论是向量、关键词还是混合）快速但不精确。它使用双编码器（bi-encoders）：查询和每个文档独立嵌入，然后比较。嵌入计算一次并缓存。这可以扩展到数百万份文档。

重排序使用交叉编码器（cross-encoders）：查询和候选文档一起输入一个输出相关性分数的模型。模型同时看到两个文本，可以捕捉它们之间的细粒度交互。Cross-encoder 可以理解 "What were Q3 earnings?" 与包含 "$47.2M in Q3" 的 chunk 高度相关，即使 bi-encoder 错过了这种联系。

权衡：cross-encoders 比 bi-encoders 慢 100-1000 倍，因为它们联合处理查询-文档对。你无法为百万份文档预计算 cross-encoder 分数。解决方案：检索更大的候选集（混合搜索的 top-50），然后用 cross-encoder 重排序以获得最终的 top-5。

```mermaid
graph LR
    Q["Query"] --> H["Hybrid Search"]
    H --> C50["Top 50 candidates"]
    C50 --> RR["Cross-Encoder Reranker"]
    RR --> C5["Top 5 final results"]
    C5 --> P["Build prompt"]
    P --> LLM["Generate answer"]
```

常见重排序模型（2026 阵容）：
- Cohere Rerank 3.5：托管 API，多语言，混合语料库上最佳召回提升
- Voyage rerank-2.5：托管 API，延迟最低的托管选项
- Jina-Reranker-v2 Multilingual：开源权重，100+ 语言
- bge-reranker-v2-m3：开源权重，强基线
- cross-encoder/ms-marco-MiniLM-L-6-v2：开源权重，CPU 原型设计
- ColBERTv2 / Jina-ColBERT-v2：延迟交互多向量重排序器 —— 评分时间是 O(tokens) 而非 O(docs)

### 查询转换

有时问题不在检索，而在查询本身。"What was that thing about the new policy change?" 是一个糟糕的搜索查询。它没有具体词汇。嵌入很模糊。没有检索系统能从这个查询找到正确的文档。

**查询重写（Query rewriting）**：将用户查询改写为更好的搜索查询。LLM 可以做这个：

```
User: "What was that thing about the new policy change?"
Rewritten: "Recent policy changes and updates"
```

**HyDE（Hypothetical Document Embeddings，假设文档嵌入）**：不是用查询搜索，而是生成一个假设答案，嵌入它，然后搜索相似的真实文档。

```
Query: "What is the refund policy for enterprise?"
Hypothetical answer: "Enterprise customers are eligible for a full refund
within 60 days of purchase. Refunds are pro-rated based on the remaining
subscription period and processed within 5-7 business days."
```

嵌入假设答案并搜索与其相似的真实文档。直觉：假设答案在嵌入空间中比原始问题更接近真实答案。问题和答案有不同的语言结构。通过生成假设答案，你弥合了嵌入中"问题空间"和"答案空间"之间的差距。

HyDE 在检索前增加一次 LLM 调用。这增加 500-2000ms 延迟。当原始查询的检索质量差时值得使用。

### 父子分块（Parent-Child Chunking）

标准分块迫使一个权衡：小 chunk 用于精确检索，大 chunk 用于足够上下文。父子分块消除了这个权衡。

索引小 chunk（128 tokens）用于检索。当小 chunk 被检索到时，返回其父 chunk（512 tokens）用于提示。小 chunk 精确匹配查询。父 chunk 为 LLM 生成好答案提供足够的上下文。

```mermaid
graph TD
    P["Parent chunk (512 tokens)<br/>Full section about refund policy"]
    C1["Child chunk (128 tokens)<br/>Standard plan: 30-day refund"]
    C2["Child chunk (128 tokens)<br/>Enterprise: 60-day pro-rated"]
    C3["Child chunk (128 tokens)<br/>Processing time: 5-7 days"]
    C4["Child chunk (128 tokens)<br/>How to submit a request"]

    P --> C1
    P --> C2
    P --> C3
    P --> C4

    Q["Query: enterprise refund?"] -.->|"matches child"| C2
    C2 -.->|"return parent"| P
```

查询 "enterprise refund?" 精确匹配子 chunk C2。但提示收到完整的父 chunk P，其中包含关于处理时间和提交流程的周围上下文。

### 元数据过滤（Metadata Filtering）

在运行向量搜索之前，按元数据过滤语料库：日期、来源、类别、作者、语言。这减少搜索空间并防止不相关的结果。

"What changed in the security policy last month?" 应该只搜索过去 30 天内安全类别的文档。没有元数据过滤，你搜索整个语料库，可能检索到两年前恰好语义相似的安全文档。

生产 RAG 系统为每个 chunk 存储元数据：源文档、创建日期、类别、作者、版本。向量数据库支持在相似度搜索之前按元数据预过滤，这对大规模性能至关重要。

### 评估

你构建了一个 RAG 系统。如何知道它是否有效？三个指标：

**检索相关性（Recall@k）**：对于一组带有已知相关文档的测试问题，相关文档中有多少百分比出现在 top-k 结果中？如果问题的答案在 chunk #47，chunk #47 是否出现在 top-5 中？

**忠实度（Faithfulness）**：生成的回答是否基于检索到的文档？如果检索到的 chunk 说"60-day refund window"而模型说"90-day refund window"，那就是忠实度失败。模型尽管有正确的上下文还是幻觉了。

**回答正确性（Answer correctness）**：生成的回答是否与预期答案匹配？这是端到端指标。它结合了检索质量和生成质量。

一个简单的忠实度检查：取生成回答中的每个声明，并验证它（在实质上）出现在检索到的 chunk 中。如果回答包含任何检索到的 chunk 中没有的事实，它很可能是幻觉的。

```mermaid
graph TD
    subgraph "评估框架"
        Q["测试问题<br/>+ 预期答案<br/>+ 相关文档 ID"]
        Q --> Ret["检索评估<br/>Recall@k: 正确<br/>文档是否被检索？"]
        Q --> Faith["忠实度评估<br/>回答是否 grounding<br/>在检索到的文档中？"]
        Q --> Correct["正确性评估<br/>回答是否匹配<br/>预期答案？"]
    end
```

## 动手实现

### 步骤 1：BM25 实现

```python
import math
from collections import Counter

class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = {}
        self.n_docs = 0

    def index(self, documents):
        self.docs = documents
        self.n_docs = len(documents)
        self.doc_lengths = []
        self.doc_freqs = {}

        for doc in documents:
            words = doc.lower().split()
            self.doc_lengths.append(len(words))
            unique_words = set(words)
            for word in unique_words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.avg_dl = sum(self.doc_lengths) / self.n_docs if self.n_docs else 1

    def score(self, query, doc_idx):
        query_words = query.lower().split()
        doc_words = self.docs[doc_idx].lower().split()
        doc_len = self.doc_lengths[doc_idx]
        word_counts = Counter(doc_words)
        score = 0.0

        for term in query_words:
            if term not in word_counts:
                continue
            tf = word_counts[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * numerator / denominator

        return score

    def search(self, query, top_k=10):
        scores = [(i, self.score(query, i)) for i in range(self.n_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 步骤 2：倒数排名融合

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
```

### 步骤 3：混合搜索流水线

```python
def hybrid_search(query, chunks, vector_embeddings, vocab, idf, bm25_index, top_k=5, fusion_k=60):
    query_emb = tfidf_embed(query, vocab, idf)
    vector_results = search(query_emb, vector_embeddings, top_k=top_k * 3)
    bm25_results = bm25_index.search(query, top_k=top_k * 3)
    fused = reciprocal_rank_fusion([vector_results, bm25_results], k=fusion_k)
    return fused[:top_k]
```

### 步骤 4：简单重排序器

在生产环境中，你会使用 cross-encoder 模型。这里我们构建一个重排序器，使用词重叠、词重要性和短语匹配来评分查询-文档相关性。

```python
def rerank(query, candidates, chunks):
    query_words = set(query.lower().split())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how",
                  "why", "when", "where", "do", "does", "for", "of", "in", "to",
                  "and", "or", "on", "at", "by", "it", "its", "this", "that",
                  "with", "from", "be", "has", "have", "had", "not", "but"}
    query_terms = query_words - stop_words

    scored = []
    for doc_id, initial_score in candidates:
        chunk = chunks[doc_id].lower()
        chunk_words = set(chunk.split())

        term_overlap = len(query_terms & chunk_words)

        query_bigrams = set()
        q_list = [w for w in query.lower().split() if w not in stop_words]
        for i in range(len(q_list) - 1):
            query_bigrams.add(q_list[i] + " " + q_list[i + 1])
        bigram_matches = sum(1 for bg in query_bigrams if bg in chunk)

        position_boost = 0
        for term in query_terms:
            pos = chunk.find(term)
            if pos != -1 and pos < len(chunk) // 3:
                position_boost += 0.5

        rerank_score = (
            term_overlap * 1.0
            + bigram_matches * 2.0
            + position_boost
            + initial_score * 5.0
        )
        scored.append((doc_id, rerank_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

### 步骤 5：HyDE（假设文档嵌入）

```python
def hyde_generate_hypothesis(query):
    templates = {
        "what": "The answer to '{query}' is as follows: Based on our documentation, {topic} involves specific policies and procedures that define how the process works.",
        "how": "To address '{query}': The process involves several steps. First, you need to initiate the request. Then, the system processes it according to the defined rules.",
        "default": "Regarding '{query}': Our records indicate specific details and policies related to this topic that provide a comprehensive answer."
    }
    query_lower = query.lower()
    if query_lower.startswith("what"):
        template = templates["what"]
    elif query_lower.startswith("how"):
        template = templates["how"]
    else:
        template = templates["default"]

    topic_words = [w for w in query.lower().split()
                   if w not in {"what", "is", "the", "how", "do", "does", "a", "an",
                                "for", "of", "to", "in", "on", "at", "by", "and", "or"}]
    topic = " ".join(topic_words) if topic_words else "this topic"

    return template.format(query=query, topic=topic)


def hyde_search(query, chunks, vector_embeddings, vocab, idf, top_k=5):
    hypothesis = hyde_generate_hypothesis(query)
    hypothesis_emb = tfidf_embed(hypothesis, vocab, idf)
    results = search(hypothesis_emb, vector_embeddings, top_k)
    return results, hypothesis
```

### 步骤 6：父子分块

```python
def create_parent_child_chunks(text, parent_size=200, child_size=50):
    words = text.split()
    parents = []
    children = []
    child_to_parent = {}

    parent_idx = 0
    start = 0
    while start < len(words):
        parent_end = min(start + parent_size, len(words))
        parent_text = " ".join(words[start:parent_end])
        parents.append(parent_text)

        child_start = start
        while child_start < parent_end:
            child_end = min(child_start + child_size, parent_end)
            child_text = " ".join(words[child_start:child_end])
            child_idx = len(children)
            children.append(child_text)
            child_to_parent[child_idx] = parent_idx
            child_start += child_size

        parent_idx += 1
        start += parent_size

    return parents, children, child_to_parent
```

### 步骤 7：忠实度评估

```python
def evaluate_faithfulness(answer, retrieved_chunks):
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not answer_sentences:
        return 1.0, []

    grounded = 0
    ungrounded = []
    context = " ".join(retrieved_chunks).lower()

    for sentence in answer_sentences:
        words = set(sentence.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or",
                      "to", "of", "in", "for", "on", "at", "by", "it", "this", "that"}
        content_words = words - stop_words
        if not content_words:
            grounded += 1
            continue

        matched = sum(1 for w in content_words if w in context)
        ratio = matched / len(content_words) if content_words else 0

        if ratio >= 0.5:
            grounded += 1
        else:
            ungrounded.append(sentence)

    score = grounded / len(answer_sentences) if answer_sentences else 1.0
    return score, ungrounded


def evaluate_retrieval_recall(queries_with_relevant, retrieval_fn, k=5):
    total_recall = 0.0
    results = []

    for query, relevant_indices in queries_with_relevant:
        retrieved = retrieval_fn(query, k)
        retrieved_indices = set(idx for idx, _ in retrieved)
        relevant_set = set(relevant_indices)
        hits = len(retrieved_indices & relevant_set)
        recall = hits / len(relevant_set) if relevant_set else 1.0
        total_recall += recall
        results.append({
            "query": query,
            "recall": recall,
            "hits": hits,
            "total_relevant": len(relevant_set)
        })

    avg_recall = total_recall / len(queries_with_relevant) if queries_with_relevant else 0
    return avg_recall, results
```

## 实际应用

使用真实的 cross-encoder 进行重排序：

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, candidates, chunks, top_k=5):
    pairs = [(query, chunks[doc_id]) for doc_id, _ in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip([doc_id for doc_id, _ in candidates], scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

使用 Cohere 的托管重排序器：

```python
import cohere

co = cohere.Client()

def rerank_with_cohere(query, candidates, chunks, top_k=5):
    docs = [chunks[doc_id] for doc_id, _ in candidates]
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=top_k
    )
    return [(candidates[r.index][0], r.relevance_score) for r in response.results]
```

使用真实 LLM 的 HyDE：

```python
import anthropic

client = anthropic.Anthropic()

def hyde_with_llm(query):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a short paragraph that would be a good answer to this question. Do not say you don't know. Just write what the answer would look like.\n\nQuestion: {query}"
        }]
    )
    return response.content[0].text
```

使用 Weaviate 的生产环境混合搜索：

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.get("Documents")
response = collection.query.hybrid(
    query="enterprise refund policy",
    alpha=0.5,
    limit=10
)
```

alpha 参数控制平衡：0.0 = 纯关键词（BM25），1.0 = 纯向量，0.5 = 等权重。大多数生产系统使用 0.3 到 0.7 之间的 alpha。

## 产出物

本课产出：
- `outputs/prompt-advanced-rag-debugger.md` —— 一个用于诊断和修复 RAG 质量问题的提示
- `outputs/skill-advanced-rag.md` —— 一个用于构建生产级 RAG（混合搜索和重排序）的技能

## 练习

1. 在样本文档上比较 BM25 vs 向量搜索 vs 混合搜索。对于 5 个测试查询中的每一个，记录哪种方法在位置 #1 返回最相关的 chunk。混合搜索应该至少在 5 个中的 3 个上获胜。

2. 实现一个元数据过滤器。为每个文档添加一个"category"字段（security、billing、api、product）。在运行向量搜索之前，只过滤到相关类别的 chunk。用"What encryption is used?"测试，并验证它只搜索 security 类别的 chunk。

3. 使用第 06 课的 simple generate 函数构建完整的 HyDE 流水线。在所有 5 个测试查询上比较直接查询搜索和 HyDE 搜索的检索质量（top-3 相关性）。HyDE 应该改善模糊查询的结果。

4. 在样本文档上实现父子分块策略。使用 child_size=30 和 parent_size=100。用子 chunk 搜索但将父 chunk 返回给提示。将生成的答案与 chunk_size=50 的标准分块进行比较。

5. 创建一个评估数据集：10 个带有已知答案 chunk 的问题。测量 (a) 仅向量搜索、(b) 仅 BM25、(c) 混合搜索、(d) 混合 + 重排序的 Recall@3、Recall@5 和 Recall@10。绘制结果并识别重排序最有帮助的地方。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| BM25 | "关键词搜索" | 一种概率排名算法，通过词频、逆文档频率和文档长度归一化对文档评分 |
| Hybrid search（混合搜索） | "两全其美" | 并行运行语义（向量）和关键词（BM25）搜索，然后用排名融合合并结果 |
| Reciprocal Rank Fusion（倒数排名融合） | "合并排名列表" | 通过对每个文档在所有列表中的 1/(k + rank) 求和来合并多个排名列表 |
| Reranking（重排序） | "第二遍评分" | 使用更昂贵的 cross-encoder 模型对初始检索的候选集重新评分 |
| Cross-encoder（交叉编码器） | "联合查询-文档模型" | 将查询和文档作为单个输入，产生相关性分数的模型；比 bi-encoders 更准确，但对完整语料库搜索太慢 |
| Bi-encoder（双编码器） | "独立嵌入模型" | 独立嵌入查询和文档的模型；快因为嵌入是预计算的，但比 cross-encoders 不太准确 |
| HyDE | "用假答案搜索" | 生成查询的假设答案，嵌入它，并搜索与其相似的真实文档 |
| Parent-child chunking（父子分块） | "小搜索，大上下文" | 索引小 chunk 以精确检索，但返回更大的父 chunk 以提供足够的上下文 |
| Metadata filtering（元数据过滤） | "搜索前缩小范围" | 在运行向量搜索之前按属性（日期、来源、类别）过滤文档，以减少搜索空间 |
| Faithfulness（忠实度） | "是否保持 grounding" | 生成的回答是否得到检索到的文档支持，而非从模型的训练数据中幻觉出来 |

## 延伸阅读

- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond" (2009) -- BM25 的权威参考，解释公式背后的概率基础
- Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods" (2009) -- 原始 RRF 论文，证明它击败更复杂的融合方法
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) -- HyDE 论文，证明假设文档嵌入无需任何训练数据即可改善检索
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) -- 证明在 BM25 之上的 cross-encoder 重排序显著提升检索质量
- [Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (2023)](https://arxiv.org/abs/2310.03714) -- 将提示构建和权重选择视为检索流水线的优化问题；阅读以了解"编程 LLM"而非"提示 LLM"
- [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft Research 2024)](https://arxiv.org/abs/2404.16130) -- GraphRAG 论文：实体-关系提取 + Leiden 社区检测用于查询聚焦摘要；全局 vs 局部检索的区别
- [Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (ICLR 2024)](https://arxiv.org/abs/2310.11511) -- 带反思 token 的自我评估 RAG；超越静态 retrieve-then-generate 的智能前沿
- [LangChain Query Construction blog](https://blog.langchain.dev/query-construction/) -- 如何将自然语言查询转换为结构化数据库查询（Text-to-SQL、Cypher）作为检索前步骤
