---
name: long-context-eval
description: 为给定模型和使用场景设计长上下文评估方案。
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

给定目标模型、目标上下文长度和使用场景，输出：

1. Tests（测试）。NIAH depth × length 网格；RULER multi-hop；自定义领域任务。
2. Sampling（采样）。在每个长度上测试深度 0, 0.25, 0.5, 0.75, 1.0。
3. Metrics（指标）。Retrieval pass rate（检索通过率）；reasoning pass rate（推理通过率）；time-to-first-token（首 token 时间）；cost-per-query（每次查询成本）。
4. Cutoff（截断值）。Effective retrieval length（有效检索长度，90% 通过）和 effective reasoning length（有效推理长度，70% 通过）。两者都报告。
5. Regression（回归测试）。固定测试框架，每次模型升级都重新运行，暴露差异。

拒绝仅信任 model card（模型卡）上的上下文窗口。拒绝任何 multi-hop 工作负载仅使用 NIAH 评估。拒绝将厂商自报的长上下文分数作为独立证据。
