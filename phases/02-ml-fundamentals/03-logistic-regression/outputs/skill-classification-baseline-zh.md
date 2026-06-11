---
name: skill-classification-baseline
description: 在尝试复杂模型之前，先使用 logistic regression (逻辑回归) 建立一个强大的分类基线
version: 1.0.0
phase: 2
lesson: 3
tags: [classification, logistic-regression, baseline, preprocessing]
---

# 分类基线指南

在尝试复杂模型之前，先用 logistic regression (逻辑回归) 建立一个基线。它在几秒钟内就能完成训练，输出概率，并且完全可解释。令人惊讶的是，许多现实世界的问题根本不需要更复杂的模型。

## 决策检查清单

1. Decision boundary (决策边界) 可能是线性的吗？
   - 是：logistic regression (逻辑回归) 可能就足够了
   - 否：你仍然需要它作为基线来衡量改进

2. 你有多少个特征？
   - 少于 50 个：标准 logistic regression (逻辑回归) 即可
   - 50 到 10,000 个：添加 L2 正则化（Ridge）
   - 超过 10,000 个（例如 TF-IDF 文本特征）：使用 L1 正则化（Lasso）或 LinearSVC

3. 数据集是否不平衡？
   - 比例低于 5:1：可能不需要调整
   - 5:1 到 50:1：在 sklearn 中使用 `class_weight="balanced"`
   - 超过 50:1：结合类别权重与适当的指标（precision、recall 或 F1）

4. 特征是否处于不同的尺度？
   - 在 logistic regression (逻辑回归) 之前始终进行标准化。它使用基于梯度的优化，未缩放的特征会减慢收敛速度或扭曲 decision boundary (决策边界)。

5. 是否存在缺失值？
   - 在拟合之前进行插补。Logistic regression (逻辑回归) 无法处理 NaN。
   - 数值列使用中位数插补，类别列使用众数插补。

## 何时 logistic regression (逻辑回归) 已经足够好

- 具有 mostly 线性特征关系的二分类问题
- 你需要概率输出（不仅仅是类别标签）
- 需要可解释性（系数表示特征重要性的方向和标准化后的相对幅度）
- 训练数据量小（数百到数千个样本）
- 你需要一个用于实时服务的快速模型（推理时只需一次点积运算）
- 法规或合规性要求需要可解释性

## 何时需要升级

- 在尝试特征工程后，准确率仍远低于目标
- 特征与目标之间的关系明显是非线性的（检查残差图）
- 你有大量表格数据（10k+ 行）：尝试梯度提升（XGBoost 或 LightGBM）
- 特征具有多项式特征无法捕捉的复杂交互
- 你有图像、文本或序列数据：在原始输入上使用 logistic regression (逻辑回归) 不会奏效

## 分类基线的预处理步骤

1. **先进行训练/测试拆分**，再进行任何预处理。这可以防止数据泄露。
2. **处理缺失值**：数值用中位数插补，类别用众数插补。
3. **编码类别特征**：低基数（少于 10 个值）使用 one-hot encoding (独热编码)，高基数使用目标编码。目标编码仅在训练折上拟合（使用 out-of-fold 编码以防止泄露）。
4. **缩放数值特征**：StandardScaler（零均值，单位方差）。在训练集上拟合，然后转换两者。
5. **拟合 logistic regression (逻辑回归)**，使用 `C=1.0`（默认正则化）。
6. **评估**：confusion matrix (混淆矩阵)、precision (精确率)、recall (召回率)、F1。不要只看 accuracy (准确率)。
7. **调优 threshold (阈值)**：默认 0.5 很少是最优的。在 0.1 到 0.9 之间扫描，选择符合你的 precision/recall 优先级的 threshold (阈值)。

## 常见错误

- 在不平衡数据上仅评估 accuracy (准确率)（一个总是预测多数类别的模型得分很高但毫无用处）
- 忘记缩放特征（未缩放的特征会使 logistic regression (逻辑回归) 训练缓慢并收敛到更差的解）
- 使用测试集来调优决策 threshold (阈值)（应使用验证集或交叉验证）
- 跳过基线直接跳到 XGBoost（你失去了可解释性，也没有参考点）
- 不检查多重共线性（高度相关的特征会膨胀系数方差）

## 快速参考

| 场景 | 模型 | 正则化 | 关键设置 |
|------|------|--------|---------|
| 特征少，可解释 | LogisticRegression | L2 (默认) | C=1.0 |
| 特征多，部分无关 | LogisticRegression | L1 | penalty="l1", solver="saga" |
| 高维稀疏（文本） | SGDClassifier | L1 或 ElasticNet | loss="log_loss" |
| 类别不平衡 | LogisticRegression | L2 | class_weight="balanced" |
| 需要概率 | LogisticRegression | L2 | predict_proba() |
| 只需要类别标签 | LinearSVC | L2 | 大数据比 LR 更快 |
