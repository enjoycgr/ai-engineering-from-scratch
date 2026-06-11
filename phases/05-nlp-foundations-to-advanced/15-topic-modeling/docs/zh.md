# 主题建模 — LDA 与 BERTopic

> LDA：文档是主题的混合，主题是词上的分布。BERTopic：文档在嵌入空间中聚类，聚类即主题。目标相同，分解方式不同。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 分钟

## 问题背景

你有 10,000 条客服工单、50,000 篇新闻文章，或 200,000 条推文。你想在不逐篇阅读的情况下，知道这批数据主要讲什么。你没有标注好的类别，甚至不知道存在多少类别。

主题建模（topic modeling）可以在无监督的情况下回答这个问题。给它一个语料库，它会返回一组数量不多的、语义连贯的主题，以及每篇文档在这些主题上的分布。

两大算法家族占主导地位。LDA（2003）将每篇文档视为潜在主题（latent topics）的混合，每个主题是词上的概率分布。推断采用贝叶斯方法。在需要混合成员（mixed-membership）主题分配和可解释的词级概率分布的生产环境中，LDA 至今仍在使用。

BERTopic（2020）使用 BERT 对文档进行编码，用 UMAP 降维，再用 HDBSCAN 聚类，最后通过基于类别的 TF-IDF 提取主题词。它在短文本、社交媒体以及语义相似性比词重叠更重要的场景下表现更好。每篇文档只分配一个主题，这对长内容来说是一个局限。

本节课将建立对两者的直觉，并说明针对给定语料库应如何选择。

## 核心概念

![LDA 混合模型 vs BERTopic 聚类](../assets/topic-modeling.svg)

**LDA 生成故事（generative story）。** 每个主题是词上的分布。每篇文档是主题的混合。要生成文档中的一个词，先从文档的主题混合中采样一个主题，再从该主题的词分布中采样一个词。推断则反过来：给定观测到的词，推断每篇文档的主题分布和每个主题的词分布。折叠吉布斯采样（collapsed Gibbs sampling）或变分贝叶斯（variational Bayes）负责具体的数学计算。

LDA 的关键输出：

- `doc_topic`：矩阵 `(n_docs, n_topics)`，每行求和为 1（文档的主题混合）。
- `topic_word`：矩阵 `(n_topics, vocab_size)`，每行求和为 1（主题的词分布）。

**BERTopic 流程。**

1. 用句子 Transformer（如 `all-MiniLM-L6-v2`）对每篇文档编码，得到 384 维向量。
2. 用 UMAP 将维度降至约 5 维。BERT 嵌入维度过高，直接聚类效果差。
3. 用 HDBSCAN 聚类。基于密度，产生大小不一的簇和一个“离群”标签。
4. 对每个簇，在其文档上计算基于类别的 TF-IDF，提取 top 主题词。

输出是每篇文档一个主题（外加 -1 的离群标签）。可选地，通过 HDBSCAN 的概率向量获得软成员关系（soft membership）。

## 动手实现

### 第一步：通过 scikit-learn 实现 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：已去除停用词（stopwords），min_df 和 max_df 过滤掉罕见词和过于常见的词，使用 CountVectorizer（而非 TfidfVectorizer），因为 LDA 期望原始词频计数。

### 第二步：BERTopic（生产环境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

对 `Topic != -1` 的过滤会丢弃 BERTopic 的离群桶（outlier bucket，即 HDBSCAN 无法聚类的文档）。`min_topic_size` 控制 HDBSCAN 的最小簇大小；BERTopic 库的默认值是 10。本例显式设为 15，以适应课程规模的数据。对于超过 10,000 篇文档的语料库，建议提高到 50 或 100。

### 第三步：评估

两种方法都会输出主题词。问题是这些词是否语义连贯。

- **主题连贯性（topic coherence, c_v）。** 结合滑动窗口上下文（sliding-window contexts）中 top 主题词对的 NPMI（normalized pointwise mutual information，归一化点互信息），将分数聚合成主题向量，再通过余弦相似度比较。越高越好。使用 `gensim.models.CoherenceModel` 并设置 `coherence="c_v"`。
- **主题多样性（topic diversity）。** 所有主题 top 词中唯一词的比例。越高越好（主题之间不重叠）。
- **定性检查。** 阅读每个主题的 top 词。它们是否指代一个真实的事物？人工判断仍是最后一道防线。

## 如何选择

| 场景 | 选择 |
|-----------|------|
| 短文本（推文、评论、标题） | BERTopic |
| 长文档且存在主题混合 | LDA |
| 无 GPU / 计算资源有限 | LDA 或 NMF |
| 需要文档级别的多主题分布 | LDA |
| 与 LLM 集成进行主题标注 | BERTopic（直接支持） |
| 资源受限的边缘部署 | LDA |
| 最大化语义连贯性 | BERTopic |

最大的实际考量是文档长度。BERT 嵌入会截断（truncate）；LDA 的计数方式对任何长度都有效。如果文档长度超过嵌入模型的上下文限制，要么进行分块 + 聚合，要么使用 LDA。

## 应用场景

2026 年的技术栈：

- **BERTopic。** 短文本和任何语义重要的场景的默认选择。
- **`gensim.models.LdaModel`。** 经典 LDA，生产环境成熟、久经考验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 实验阶段最简便的 LDA。
- **NMF。** 非负矩阵分解（Non-negative matrix factorization）。LDA 的快速替代方案，在短文本上质量相当。
- **Top2Vec。** 设计与 BERTopic 类似。社区较小，但在部分基准测试中表现不错。
- **FASTopic。** 较新，在超大规模语料库上比 BERTopic 更快。
- **基于 LLM 的标注。** 运行任意聚类后，再 prompt 一个模型为每个簇命名。

## 交付物

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: 为语料库选择 LDA 或 BERTopic。指定库、参数和评估方式。
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

给定语料库描述（文档数量、平均长度、领域、语言、计算预算），输出：

1. 算法。LDA / NMF / BERTopic / Top2Vec / FASTopic。一句话说明理由。
2. 配置。主题数量：推荐 `max(5, round(sqrt(n_docs)))`，40,000 篇以下文档的语料库上限为 200；仅当语料库确实很大（>40k）时才允许超过 200，并注明计算成本增加。`min_df` / `max_df` 过滤器和神经方法的嵌入模型也放在这里。
3. 评估。通过 `gensim.models.CoherenceModel` 计算主题连贯性（c_v），主题多样性，以及 20 个样本的人工阅读。
4. 需要排查的失效模式。LDA 的“垃圾主题”吸收停用词和极常见词；BERTopic 的 -1 离群簇吞掉模糊文档。

如果文档长度超过嵌入模型的上下文窗口且没有分块策略，拒绝使用 BERTopic。如果文本极短（推文、少于 10 个 token 的评论），拒绝使用 LDA，因为连贯性会崩溃。任何低于 5 的 n_topics 选择都标记为可能错误；40k 以下语料库选择 >200 标记为可能过度拆分。
```

## 练习

1. **简单。** 在 20 Newsgroups 数据集上拟合 5 个主题的 LDA。打印每个主题的 top 10 词。手工为每个主题打标签。算法是否找到了真实的类别？
2. **中等。** 在相同的 20 Newsgroups 子集上拟合 BERTopic。比较找到的主题数量、top 词和定性连贯性与 LDA 的差异。哪种方法更清晰地呈现了真实类别？
3. **困难。** 在你的语料库上为 LDA 和 BERTopic 都计算 c_v 连贯性。分别用 5、10、20、50 个主题运行。绘制连贯性 vs 主题数量的曲线。报告哪种方法在不同主题数量下更稳定。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Topic（主题） | 语料库讲的内容 | 词上的概率分布（LDA）或相似文档的簇（BERTopic）。 |
| Mixed membership（混合成员） | 文档属于多个主题 | LDA 为每篇文档分配一个覆盖所有主题的分布。 |
| UMAP | 降维 | 保留局部结构的流形学习；用于 BERTopic。 |
| HDBSCAN | 密度聚类 | 发现大小不一的簇；为离群点生成“噪声”标签（-1）。 |
| c_v coherence | 主题质量指标 | top 主题词在滑动窗口内的平均逐点互信息。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 论文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 论文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 等指标的论文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 生产参考文档，示例丰富。
