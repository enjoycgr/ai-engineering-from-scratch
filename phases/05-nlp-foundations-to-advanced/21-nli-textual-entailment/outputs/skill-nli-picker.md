---
name: nli-picker
description: 为分类 / 忠实度 / 零样本任务选择 NLI 模型、标签模板和评估方案。
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

给定一个使用场景（忠实度检查、零样本分类、文档级推理），输出：

1. Model. 具名的 NLI checkpoint。理由需关联 domain（领域）、length（长度）、language（语言）。
2. Template（如果是 zero-shot）。语言化模式。示例。
3. Threshold. 决策规则的 entailment（蕴含）截断值。理由基于 calibration（校准）。
4. Evaluation. 在留出标注集上的 accuracy（准确率）、hypothesis-only baseline、adversarial subset。

拒绝在没有 100 例标注 sanity check 的情况下交付 zero-shot 分类。拒绝在文档长度前提上使用句子级 NLI 模型。标记任何声称 NLI 能解决幻觉的说法——它只能减少幻觉，不能消除。
