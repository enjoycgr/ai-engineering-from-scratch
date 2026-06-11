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
3. 评估。通过 `gensim.models.CoherenceModel` 计算主题连贯性（topic coherence, c_v），主题多样性（topic diversity），以及 20 个样本的人工阅读。
4. 需要排查的失效模式。LDA 的“垃圾主题”吸收停用词和频繁词。BERTopic 的 -1 离群簇吞掉模糊文档。

如果文档长度超过嵌入模型的上下文窗口且没有分块策略，拒绝使用 BERTopic。如果文本极短（推文、少于 10 个 token 的评论），拒绝使用 LDA，因为连贯性会崩溃。任何低于 5 的 n_topics 选择都标记为可能错误；40k 以下语料库选择 >200 标记为可能过度拆分。
