---
name: lm-baseline
description: 在训练神经语言模型之前，构建一个可复现的 n-gram 语言模型基线。
phase: 5
lesson: 16
---

给定语料库和目标用途（下一个词预测、重打分、困惑度基线），输出：

1. N-gram 阶数。通用英语用 trigram，大语料库用 4-gram，语音重打分用 5-gram。
2. 平滑方法。默认使用 Modified Kneser-Ney；教学场景可用 Laplace。
3. 库选择。生产环境用 `kenlm`，教学用 `nltk.lm`，仅为了学习数学原理时才自己实现。
4. 评估。使用 held-out 困惑度（perplexity），训练集和测试集之间保持一致的 tokenization。

拒绝报告在不同 tokenization 系统之间计算的困惑度 —— 困惑度数字只有在相同 tokenization 下才可比较。标记测试集中的 OOV 比例；KN 对 OOV 处理不佳，除非你在训练期间预留了特殊的 `<UNK>` 词元。
