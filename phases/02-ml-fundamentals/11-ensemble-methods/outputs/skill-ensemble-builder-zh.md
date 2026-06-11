---
name: skill-ensemble-builder
description: 为你的问题选择合适的集成方法并进行配置
version: 1.0.0
phase: 2
lesson: 11
tags: [ensemble, bagging, boosting, random-forest, xgboost, stacking]
---

# 集成方法选择指南

集成（ensemble）组合多个模型以产生比任何单个模型更好的预测。问题始终是：哪种集成，以及何时使用？

## 决策清单

1. 你当前模型的主要问题是什么？
   - 高方差（high variance，过拟合）：使用 bagging（Random Forest）
   - 高偏差（high bias，欠拟合）：使用 boosting（Gradient Boosting, XGBoost）
   - 两者都有，或你追求最大准确率：使用 stacking

2. 你有多少数据？
   - 少于 1,000 行：Random Forest（鲁棒，难以配置错误）
   - 1,000 到 100,000 行：XGBoost 或 LightGBM（表格数据整体最佳）
   - 超过 100,000 行：LightGBM（最快的梯度提升，大数据处理良好）

3. 你能投入多少调参时间？
   - 最少：Random Forest 默认参数（几乎总是有效）
   - 中等：XGBoost，learning_rate=0.1，用 early stopping 调 n_estimators
   - 最大：LightGBM 或 XGBoost，使用贝叶斯超参数搜索

4. 你需要可解释性吗？
   - 是：单棵决策树或带特征重要性的小型 Random Forest
   - 部分：带 SHAP 值的 gradient boosting
   - 否：stacking 或深度集成

5. 数据是否有很多噪声和异常值？
   - 是：Random Forest（bagging 对噪声具有鲁棒性）
   - 否：gradient boosting（在干净数据上可以进一步提升准确率）

## 何时使用每种方法

**Random Forest (Bagging)**：你的安全首选。在 bootstrap samples 上训练多棵树并平均。在不增加 bias 的情况下降低 variance。在中等数据上几乎不可能过拟合。调参需求最小：设置 n_estimators=100-500，其余用默认。

**AdaBoost**：带样本重新加权的顺序 boosting。与简单的基学习器（decision stumps）配合良好。对异常值和噪声标签敏感，因为它会提升被错误分类点的权重。在实践中 largely 被 gradient boosting 取代。

**Gradient Boosting**：将每棵新树拟合到当前集成的残差。降低 bias。表格数据最强大的方法。需要调参：learning_rate、n_estimators、max_depth、min_child_weight、subsample。

**XGBoost**：带 regularization（正则化）、二阶优化和系统级加速的 gradient boosting。原生处理缺失值。Kaggle 竞赛和生产级表格 ML 的默认选择。

**LightGBM**：带 leaf-wise（按叶子）生长（而非 level-wise 按层级）的 gradient boosting。大数据集上比 XGBoost 更快。使用直方图分裂。最适合 50k 行以上的数据集。

**CatBoost**：带原生类别特征处理的 gradient boosting。无需 one-hot 编码。当你有很多类别特征时效果很好。

**Stacking**：在多个多样化基模型的预测上训练 meta-learner。当你需要绝对最佳准确率且有充足算力时使用。始终通过 cross-validation 生成基模型预测以避免泄漏。

**Voting**：最简单的集成。Hard voting（多数类别）或 soft voting（平均概率）。快速组合 2-3 个多样化模型，无需 meta-learner。

## 常见错误

- 没有 early stopping 就使用 gradient boosting（如果运行太多轮会过拟合）
- learning_rate 设置过高（超过 0.3 通常会导致不稳定）
- 不为 gradient boosting 调 max_depth（默认无限制或非常深的树会过拟合）
- stacking 使用全部同类型的模型（多样性是 stacking 的要点）
- 在噪声数据上使用 AdaBoost（异常值每轮获得越来越高的权重）
- 期望 Random Forest 修复欠拟合（它降低 variance，不是 bias）

## 按方法的调参优先级

**Random Forest：**
1. n_estimators: 100-500（更多通常不会更差，只是更慢）
2. max_depth: None（让树完全生长）或限制在 10-20 以提速
3. max_features: 分类用 "sqrt"，回归用 "log2" 或 n/3

**XGBoost / LightGBM：**
1. learning_rate: 0.01-0.3（如果有算力训练更多树，越低越好）
2. n_estimators: 在验证集上使用 early stopping，而不是猜测
3. max_depth: 3-8（从 6 开始）
4. min_child_weight / min_data_in_leaf: 1-20（更高可防止过拟合）
5. subsample: 0.7-1.0
6. colsample_bytree: 0.7-1.0
7. reg_alpha (L1) 和 reg_lambda (L2): 0-10

## 快速参考

| 方法 | 降低 | 速度 | 调参工作量 | 最适合 |
|--------|---------|-------|--------------|----------|
| Random Forest | Variance | 快 | 低 | 噪声数据，快速基线 |
| AdaBoost | Bias | 快 | 低 | 简单基学习器，干净数据 |
| Gradient Boosting | Bias | 中等 | 高 | 表格数据，竞赛 |
| XGBoost | Both | 快 | 高 | 生产级表格 ML |
| LightGBM | Both | 最快 | 高 | 大数据集（50k+ 行） |
| CatBoost | Both | 中等 | 中等 | 大量类别特征 |
| Stacking | Both | 慢 | 高 | 最大准确率，多样化模型 |
| Voting | Variance | 快 | 无 | 快速组合 2-3 个模型 |
