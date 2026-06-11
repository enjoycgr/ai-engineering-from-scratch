---
name: sentiment-baseline
description: 为新数据集设计 sentiment analysis（情感分析）基线。
phase: 5
lesson: 05
---

给定数据集描述（domain（领域）、language（语言）、size（规模）、label granularity（标签粒度）、latency budget（延迟预算）），输出：

1. Feature extraction recipe（特征提取方案）。指定 tokenizer（分词器）、n-gram range（n 元范围）、stopword policy（停用词策略——通常保留）、negation handling（否定处理——范围前缀或 bigrams）。
2. Classifier（分类器）。Naive Bayes（朴素贝叶斯）用于基线，logistic regression（逻辑回归）用于生产环境，仅在领域需要 sarcasm（讽刺）/ aspects（方面）/ cross-lingual（跨语言）时才使用 transformer。
3. Evaluation plan（评估计划）。报告 precision（精确率）、recall（召回率）、F1、confusion matrix（混淆矩阵）和 per-class error samples（每类错误样本），不要只报告标量。
4. One failure mode to monitor post-deployment（部署后需监控的一种失效模式）。Domain drift（领域漂移）和 sarcasm 是前两名。

拒绝推荐在 sentiment 任务中丢弃 stopwords。拒绝在类别不平衡时（例如 90% 正面）将 accuracy（准确率）作为唯一指标。标记富含 subword（子词）的语言（如德语、芬兰语、土耳其语）需要 FastText 或 transformer embeddings 而非词级 TF-IDF。
