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
