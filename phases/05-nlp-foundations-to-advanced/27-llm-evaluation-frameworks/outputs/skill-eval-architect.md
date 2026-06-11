---
name: eval-architect
description: 设计带有校准评判者与 CI 门禁的 LLM 评估计划。
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

给定一个使用场景（RAG / agent / 生成任务），输出：

1. Metrics（指标）。Faithfulness / relevance / context-precision / context-recall + 任何带标准的自定义 G-Eval 指标。
2. Judge model（评判模型）。具名的模型 + 版本，成本与准确性的权衡理由。
3. Calibration（校准）。手工标注集大小，目标 Spearman rho vs 人类 > 0.7。
4. Dataset versioning（数据集版本控制）。Tag 策略、变更日志、分层（stratification）。
5. CI gate（CI 门禁）。每个指标的阈值、回归窗口逻辑、底部 quantile（分位数）告警。

拒绝依赖未在 ≥50 个人工标注示例上测试过的 judge。拒绝 self-evaluation（同一模型生成+评判）。拒绝没有暴露底部 10% 的仅聚合报告。标记任何没有并行基线评估就升级 judge 的流水线。
