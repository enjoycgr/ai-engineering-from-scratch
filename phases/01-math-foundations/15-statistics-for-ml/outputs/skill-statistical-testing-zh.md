---
name: skill-statistical-testing
zh_name: 统计检验技能
description: Choose the right statistical test for comparing ML models and evaluating experiments (为比较 ML 模型和评估实验选择正确的统计检验)
version: 1.0.0
phase: 1
lesson: 15
tags: [statistics, hypothesis-testing, model-comparison]
---

# 机器学习中的统计检验

如何为比较模型、运行 A/B 实验或验证结果选择合适的检验方法。

## 决策检查清单

1. 你在比较什么？均值、比例、分布还是相关性？
2. 有多少组？一个样本与参考值、两组还是多组？
3. 观察值是成对的（同一测试集、同一折）还是独立的？
4. 数据是否正态分布？如果 n < 30 且不是明显正态，使用非参数检验。
5. 数据是连续、有序还是分类的？
6. 你进行了多少次检验？如果多于一次，应用校正。

## 决策树

```text
比较均值？
  两组？
    成对（相同数据划分）？--> 配对 t-test（若非正态则用 Wilcoxon signed-rank）
    独立？--> Welch's t-test（若非正态则用 Mann-Whitney U）
  多组？
    成对？--> 重复测量 ANOVA（或 Friedman test）
    独立？--> 单因素 ANOVA（或 Kruskal-Wallis）

比较比例？
  两组？--> Chi-squared test 或 Fisher's exact test（小 n）
  多组？--> Chi-squared test

比较分布？
  一个分布是参考？--> Kolmogorov-Smirnov test
  两个都是经验分布？--> 双样本 KS test

度量关联？
  两者连续且大致正态？--> Pearson correlation
  有序或非正态？--> Spearman rank correlation
  分类 x 分类？--> Chi-squared test of independence

进行多次检验？
  应用 Bonferroni correction：alpha_adjusted = alpha / number_of_tests
  或使用 Holm-Bonferroni（不那么保守，仍控制族错误率）
```

## 何时使用每种检验

| 检验 | 数据类型 | 假设 | ML 用例 |
|---|---|---|---|
| 配对 t-test | 连续，成对 | 正态差值 | 在同一 k-fold 划分上比较 2 个模型 |
| Wilcoxon signed-rank | 连续/有序，成对 | 无（非参数） | 比较 2 个模型，k 较小（5-10 折） |
| Welch's t-test | 连续，独立 | 大致正态 | 在两个独立数据集上比较模型 |
| Mann-Whitney U | 连续/有序，独立 | 无 | 比较延迟分布 |
| ANOVA | 连续，3+ 组 | 正态、等方差 | 比较多个模型架构 |
| Kruskal-Wallis | 连续/有序，3+ 组 | 无 | 比较多个模型，非正态指标 |
| Chi-squared | 分类计数 | 期望计数 >= 5 | 比较类别分布、混淆矩阵 |
| Fisher's exact | 分类计数 | 小样本 | 罕见事件比较 |
| KS test | 连续 | 无 | 检验预测是否遵循期望分布 |
| Bootstrap CI | 任意统计量 | 无 | 对 AUC、F1、任何指标构建置信区间 |
| McNemar's test | 成对二元 | 无 | 在同一测试集上比较两个分类器 |

## 模型比较方案

1. 在运行实验前定义指标和显著性水平（alpha = 0.05）。
2. 在相同的 k-fold 交叉验证划分（k = 5 或 10）上运行两个模型。
3. 收集成对分数：(a_1, b_1), (a_2, b_2), ..., (a_k, b_k)。
4. 计算差值：d_i = b_i - a_i。
5. 运行配对检验（k <= 10 用 Wilcoxon，k > 10 或差值正态用配对 t-test）。
6. 报告：p-value、均值差、95% confidence interval、effect size（Cohen's d）。
7. 如果 p < alpha 且 effect size 有意义，差异是真实的且值得采取行动。

## 常见错误

- 在成对数据上使用独立检验。如果两个模型在相同的测试折上评估，必须使用配对检验。独立检验抛弃了配对信息并损失统计功效。
- 报告 p < 0.05 但不报告 effect size。统计显著的 0.1% 准确率提升不值得部署。始终计算 Cohen's d 或原始均值差。
- 在不同测试集上比较模型。测试集对两个模型必须完全相同。不同测试集使比较毫无意义。
- 运行 20 次比较并报告最好的一次而不做 Bonferroni correction。在 alpha = 0.05 下检验 20 次，你期望有 1 次 false positive。
- 在不平衡数据上使用准确率。在 99% 多数类上，一个平凡分类器就能达到 99%。使用 F1、precision-recall AUC 或 Matthews correlation coefficient。
- 将交叉验证折视为独立样本。它们共享训练数据，违反了独立性假设。corrected resampled t-test 考虑了这一点。

## 快速参考：effect size 解释

| Cohen's d | 解释 |
|---|---|
| 0.2 | 小效应 |
| 0.5 | 中效应 |
| 0.8 | 大效应 |
| > 1.0 | 极大效应 |

| 应报告 | 原因 |
|---|---|
| p-value | 差异是否真实？ |
| Confidence interval | 差异可能有多大？ |
| Effect size (Cohen's d) | 差异是否重要？ |
| 样本量 (n 或 k 折数) | 我们能信任结果吗？ |
