# 问答系统 (Question Answering Systems)

> 三种系统塑造了现代问答。抽取式 (extractive) 找到答案片段。检索增强式 (retrieval-augmented) 将答案锚定在文档中。生成式 (generative) 直接产出答案。每个现代 AI 助手都是三者的混合体。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 分钟

## 问题 (The Problem)

用户输入 "When did the first iPhone launch?"，期望得到 "June 29, 2007." 而不是 "Apple's history is long and varied." 也不是孤零零的 "2007" 没有上下文。要的是直接、有依据、正确的答案。

过去十年中，三种架构主导了问答领域。

- **抽取式 QA (Extractive QA)。** 给定一个问题和一段已知包含答案的文本，找出答案片段在文本中的起始和结束索引。SQuAD 是经典基准。
- **开放域 QA (Open-domain QA)。** 不给出文本。先检索相关段落，再抽取或生成答案。这是当今每个 RAG 流水线的基石。
- **生成式 / 闭卷 QA (Generative / Closed-book QA)。** 大型语言模型从其参数记忆中回答。无检索。推理最快，事实可靠性最低。

2026 年的趋势是混合式：先检索最佳的几段文本，再提示生成式模型基于这些文本作答。这就是 RAG，第 14 课将深入讲解检索部分。本课构建的是 QA 部分。

## 概念 (The Concept)

![QA 架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式 (Extractive)。** 使用 transformer（BERT 家族）将问题和段落一起编码。训练两个头分别预测答案的起始和结束 token 索引。损失是对有效位置的交叉熵 (cross-entropy)。输出是段落中的一个片段。按构造不会幻觉 (hallucinate)，按构造无法处理段落无法回答的问题。

**检索增强式 (RAG)。** 两个阶段。首先，检索器从语料库中找到 top-k 段落。其次，阅读器（抽取式或生成式）利用这些段落生成答案。检索器-阅读器的拆分使两者可以独立训练和评估。现代 RAG 通常会在它们之间加入一个重排器 (reranker)。

**生成式 (Generative)。** 仅解码器的 LLM（GPT、Claude、Llama）从学习到的权重中回答。无检索步骤。在常识问题上表现出色，在罕见或最新事实上表现灾难性。幻觉率与预训练数据中事实出现频率成反比。

## 动手构建 (Build It)

### 步骤 1：使用预训练模型进行抽取式 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 在 SQuAD 2.0 上训练，其中包含无法回答的问题。默认情况下，`question-answering` pipeline 即使模型的 null score 更高，也会返回得分最高的片段——它**不会**自动返回空答案。要获得显式的 "no answer" 行为，请在 pipeline 调用中传入 `handle_impossible_answer=True`：此时 pipeline 仅当 null score 超过所有片段得分时才返回空答案。无论哪种方式，都要检查 `score` 字段。

### 步骤 2：检索增强流水线（草图）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段流水线。稠密检索器 (dense retriever)（Sentence-BERT）通过语义相似度找到相关段落。抽取式阅读器 (extractive reader)（RoBERTa-SQuAD）从合并后的 top 段落中抽取答案片段。适用于小型语料库。对于百万级文档语料库，请使用 FAISS 或向量数据库。

### 步骤 3：使用 RAG 进行生成式问答

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示词 (prompt) 模式很重要。明确告诉模型要基于上下文作答，并在上下文不足时返回 "I don't know"，与朴素提示相比，可将幻觉率降低 40-60%。更复杂的模式还会添加引用、置信分数和结构化抽取。

### 步骤 4：反映真实世界的评估

SQuAD 使用 **Exact Match (EM)** 和 **token-level F1**。EM 是归一化（小写、去除标点、去掉冠词）后的严格匹配——预测与参考答案完全一致才得 1 分，否则为 0。F1 基于预测与参考答案之间的 token 重叠计算，给予部分分数。两者都对改写 (paraphrase) 欠公平："June 29, 2007" 与 "June 29th, 2007" 通常 EM 为 0（序数词打破了归一化），但重叠 token 仍能带来可观的 F1 分数。

生产环境 QA：

- **答案准确率 (Answer accuracy)**（由 LLM 或人工评判，因为指标无法捕捉语义等价性）。
- **引用准确率 (Citation accuracy)。** 引用的段落是否真的支持答案？通过生成引用与检索段落之间的字符串匹配即可轻松自动检查。
- **拒答校准 (Refusal calibration)。** 当答案不在检索到的段落中时，系统能否正确地说 "I don't know"？测量错误自信率 (false confidence rate)。
- **检索召回率 (Retrieval recall)。** 在评估阅读器之前，先测量检索器是否将正确段落放入 top-k。阅读器无法修复缺失的段落。

### RAGAS：2026 年生产环境评估框架

`RAGAS` 专为 RAG 系统设计，是 2026 年的默认选择。它在无需参考答案的情况下，从四个维度打分：

- **Faithfulness（忠实度）。** 答案中的每个主张是否都来自检索到的上下文？通过基于 NLI 的蕴含 (entailment) 测量。这是你的主要幻觉指标。
- **Answer relevance（答案相关性）。** 答案是否回答了问题？通过从答案生成假设问题并与真实问题比较来测量。
- **Context precision（上下文精确度）。** 检索到的片段中，实际相关的占多少？低精确度 = 提示中的噪声。
- **Context recall（上下文召回率）。** 检索到的集合是否包含了所有需要的信息？低召回率 = 阅读器无法成功。

无需参考的评分让你可以在实时生产流量上评估，无需精心策划的标准答案。对于开放式问题（精确匹配指标无用），再叠加 LLM-as-judge。

`pip install ragas`。接入你的检索器 + 阅读器。每个查询得到四个标量。对退化 (regression) 发出告警。

## 使用 (Use It)

2026 年技术栈。

| 使用场景 | 推荐方案 |
|---------|-------------|
| 给定段落，找出答案片段 | `deepset/roberta-base-squad2` |
| 固定语料库，闭卷不可接受 | RAG：稠密检索器 + LLM 阅读器 |
| 文档存储上的实时查询 | RAG + 混合检索 (BM25 + dense) + 重排器（第 14 课） |
| 对话式 QA（追问） | LLM + 对话历史 + 每轮 RAG |
| 高度事实性、受监管领域 | 基于权威语料库的抽取式；绝不单独使用生成式 |

抽取式 QA 在 2026 年已不时髦，因为基于 LLM 的 RAG 能处理更多场景。但在需要逐字引用的场景中仍有应用：法律研究、监管合规、审计工具。

## 交付 (Ship It)

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习 (Exercises)

1. **简单。** 在上述 SQuAD 抽取式流水线上测试 10 段 Wikipedia 文本。手工编写 10 个问题。测量答案正确的频率。如果文本和问题都清晰，你应该能看到 7-9 个正确。
2. **中等。** 添加拒答分类器。当 top retrieval score 低于阈值（比如 0.3 cosine）时，返回 "I don't know" 而不是调用阅读器。在留出集 (held-out set) 上调优阈值。
3. **困难。** 在你选择的 10,000 份文档语料库上构建 RAG 流水线。实现混合检索 (BM25 + dense) 配合 RRF 融合（见第 14 课）。测量有无混合步骤时的答案准确率。记录哪种问题类型受益最大。

## 关键术语 (Key Terms)

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Extractive QA | 找答案片段 | 预测给定段落中答案的起始和结束索引。 |
| Open-domain QA | 语料库上的 QA | 不给定段落；必须先检索再回答。 |
| RAG | 检索再生成 | Retrieval-augmented generation。检索器 + 阅读器流水线。 |
| SQuAD | 经典基准 | Stanford Question Answering Dataset。EM + F1 指标。 |
| Hallucination | 编造答案 | 阅读器输出未被检索上下文支持。 |
| Refusal calibration | 知道何时闭嘴 | 系统无法在无法回答时正确地说 "I don't know"。 |

## 延伸阅读 (Further Reading)

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 基准论文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 的经典稠密检索器。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的论文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 全面的 RAG 综述。
