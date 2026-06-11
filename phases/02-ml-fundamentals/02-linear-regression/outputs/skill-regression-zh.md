---
name: skill-regression
description: 根据数据特征和问题约束选择正确的回归方法
version: 1.0.0
phase: 2
lesson: 2
tags: [regression, linear-regression, polynomial-regression, ridge, regularization]
---

# 回归策略指南

回归预测连续值。正确的方法取决于特征与目标之间的关系、特征数量以及 overfitting (过拟合) 的风险。

## 决策清单

1. 特征与目标之间的关系大致是线性的吗？
   - 是：从 ordinary linear regression (普通线性回归) 开始
   - 否：尝试 polynomial features (多项式特征) 或非线性模型

2. 相对于样本数量，你有多少特征？
   - 特征少，样本多：ordinary linear regression (普通线性回归) 工作良好
   - 特征多，样本少：使用 regularization (正则化) (Ridge 或 Lasso)
   - 特征多于样本：Lasso (L1) 选择特征，或 Ridge (L2) 收缩所有权重

3. 你需要可解释性吗？
   - 是：使用少量特征的 linear regression (线性回归)，或 Lasso 自动特征选择
   - 否：polynomial features (多项式特征)，或转向基于树的模型或神经网络

4. 你的数据集小（少于 10,000 行）吗？
   - 使用 normal equation (正规方程)（闭式解）以获得速度
   - 交叉验证对可靠评估至关重要

5. 你的数据集大（数百万行）吗？
   - 使用 stochastic gradient descent (SGD) (随机梯度下降) 或 mini-batch gradient descent (小批量梯度下降)
   - normal equation (正规方程) 由于 O(n^3) 矩阵求逆太慢

## 何时使用每种方法

**Ordinary Linear Regression (普通线性回归)**：任何回归任务的基线。从这里开始。如果 R-squared (R平方) 可接受且模型简单，就停在这里。

**Polynomial Regression (多项式回归)**：散点图显示曲线而非直线。从 2 次开始。仅在验证性能证明合理时才增加。次数 > 5 几乎总是 overfitting (过拟合)。

**Ridge Regression (岭回归) (L2)**：许多相关特征。所有权重向零收缩，但没有完全变为零。当你相信所有特征都有贡献时很好。

**Lasso Regression (L1)**：许多特征，你怀疑只有少数重要。Lasso 将不相关特征的权重精确驱动到零，执行自动特征选择。

**Elastic Net**：结合 L1 和 L2 惩罚。当你有许多相关特征且想要一些特征选择时使用。

## 常见错误

- 在 gradient descent (梯度下降) 前跳过 feature scaling (特征缩放)（收敛变得极慢）
- 使用测试集性能调整 hyperparameter (超参数)（使用验证集或交叉验证）
- 不检查验证误差就拟合高次多项式（训练 R-squared (R平方) 总是随次数增加）
- 忽略残差图（如果残差显示模式，R-squared (R平方) 可能有误导性）
- 将 R-squared (R平方) 视为唯一指标（检查残差分布、MAE 和领域特定阈值）

## 快速参考

| 方法 | 何时使用 | Regularization (正则化) | 特征选择 |
|------|---------|------------------------|---------|
| OLS | 基线，少量特征 | 无 | 手动 |
| Ridge | 多特征，全部相关 | L2 (收缩) | 否 |
| Lasso | 多特征，少数相关 | L1 (归零) | 自动 |
| Elastic Net | 多相关特征 | L1 + L2 | 部分 |
| Polynomial | 非线性关系 | 在其上添加 Ridge/Lasso | 手动选择次数 |
