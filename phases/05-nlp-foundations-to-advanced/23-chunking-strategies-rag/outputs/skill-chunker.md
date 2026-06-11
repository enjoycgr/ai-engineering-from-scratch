---
name: chunker
description: 为给定语料库和查询分布选择分块策略、大小和重叠。
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

给定语料库（文档类型、平均长度、领域）和查询分布（factoid / analytical / multi-hop），输出：

1. 策略。Recursive / sentence / semantic / parent-document / late / contextual。理由。
2. Chunk 大小。Token 数量。与查询类型关联的理由。
3. 重叠。默认 0；如 >0 需说明理由。
4. 最小/最大限制。`min_tokens`、`max_tokens` 保护。
5. 评估计划。在 50-query stratified eval set（factoid、analytical、multi-hop）上测试 Recall@5。

拒绝任何没有最小/最大 chunk 大小强制执行的分块策略。拒绝没有 ablation 实验证明有帮助的 >20% 重叠。标记没有 min-token floor 的 semantic chunking 推荐。
