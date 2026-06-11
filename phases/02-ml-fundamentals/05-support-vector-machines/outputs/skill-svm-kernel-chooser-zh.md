---
name: skill-svm-kernel-chooser
description: 为你的问题选择合适的 SVM kernel 并调优 C 和 gamma
version: 1.0.0
phase: 2
lesson: 5
tags: [svm, kernel, classification, hyperparameter-tuning]
---

# SVM Kernel 选择指南

SVM 由两个选择定义：kernel（决定决策边界的形状）和 regularization parameters（控制 margin width 与分类错误之间的权衡）。选对它们是模型从无用到强大的关键。

## 决策检查清单

1. 数据是否线性可分（或接近可分）？
   - 是：使用 linear kernel。更快、更可解释。
   - 否：进入步骤 2。

2. 特征数 vs 样本数？
   - 特征 >> 样本（例如 TF-IDF 文本）：使用 linear kernel。高维数据通常线性可分。RBF 增加复杂度却无收益。
   - 样本 >> 特征（例如 10-50 个特征的表格数据）：RBF kernel 是默认选择。

3. 决策边界是否预期平滑？
   - 平滑、连续的边界：RBF kernel
   - 多项式形状的边界：polynomial kernel（从 degree 2 或 3 开始）
   - 领域知识暗示特定的交互项：使用匹配 degree 的 polynomial kernel

4. 数据集有多大？
   - 10,000 个样本以下：任何 kernel 都可用，RBF 是安全的默认选择
   - 10,000 到 100,000：linear kernel 或 LinearSVC（primal formulation，每轮 epoch O(n)）
   - 100,000 以上：不要使用 kernel SVM。切换到 linear SVM、gradient boosting 或神经网络。

5. 你是否缩放了特征？
   - SVM 需要 feature scaling。拟合前务必标准化（零均值，单位方差）。未缩放的特征会扭曲 margin 几何结构。

## Kernel 选择流程图

```
Start
  |
  v
Features > 1000 or features >> samples?
  Yes --> Linear kernel (LinearSVC for speed)
  No  --> Dataset < 10k samples?
            Yes --> Try RBF first (best general-purpose kernel)
            No  --> Linear kernel (kernel SVMs are O(n^2) to O(n^3))
```

如果 RBF 效果不好，尝试 polynomial degree 2-3。如果仍失败，该问题可能不适合 SVM。

## 调优 C (regularization)

C 控制误分类的惩罚。它与 regularization strength 成反比。

| C 值 | 效果 | 何时使用 |
|---------|--------|-------------|
| 0.001 - 0.01 | 宽 margin，允许许多违反 | 噪声数据，想要 generalization |
| 0.1 - 1.0 | 平衡 | 良好的起始范围 |
| 10 - 1000 | 窄 margin，很少违反 | 干净数据，需要高精度 |

调优策略：
- 从 C=1.0 开始
- 在对数尺度上搜索：[0.001, 0.01, 0.1, 1, 10, 100, 1000]
- 使用 cross-validation 挑选最佳值
- 如果最佳 C 在范围边缘，向该方向扩展范围

## 调优 gamma (RBF kernel)

Gamma 控制单个训练点的影响范围。它定义高斯的宽度。

| gamma 值 | 效果 | 何时使用 |
|-------------|--------|-------------|
| 小 (0.001) | 每个点影响大面积。平滑、简单的边界 | Underfitting 或特征少 |
| 中 (auto: 1/n_features) | sklearn 默认值。合理的起点 | 一般用途 |
| 大 (10+) | 每个点只影响附近点。复杂、波动的边界 | Overfitting 风险 |

调优策略：
- 从 gamma="scale" 开始（1 / (n_features * X.var())，sklearn 默认值）
- 在对数尺度上搜索：[0.001, 0.01, 0.1, 1, 10]
- 低 gamma + 高 C 容易 overfit
- 高 gamma + 低 C 容易 underfit

## 联合调优 C 和 gamma

C 和 gamma 会相互影响。务必一起调优，而非独立调优。

推荐方法：
1. 粗网格搜索：C 取 [0.01, 0.1, 1, 10, 100]，gamma 取 [0.001, 0.01, 0.1, 1, 10]（25 种组合）
2. 找到最佳区域
3. 在最佳区域周围进行细网格搜索（例如 C 取 [5, 10, 20, 50]，gamma 取 [0.05, 0.1, 0.2]）
4. 全程使用 5-fold cross-validation

## 常见错误

- 在高维稀疏数据上使用 RBF kernel（linear 更好且快 100 倍）
- 忘记缩放特征（最常见的 SVM 错误）
- 在噪声数据上设 C 太高（记住噪声而非学习边界）
- 在超过 50k 样本的数据集上使用 kernel SVM（训练时间不可接受）
- 不同时调优 C 和 gamma（它们相互补偿）
- 默认使用 polynomial degree 5+（严重 overfit，先尝试 2 或 3）

## 快速参考

| Kernel | 何时使用 | 关键参数 | 训练复杂度 |
|--------|------------|----------------|-------------------|
| Linear | 文本/TF-IDF、多特征、大数据 | 仅 C | 每轮 epoch O(n) |
| RBF | 通用、10k 样本以下 | C, gamma | O(n^2) 到 O(n^3) |
| Polynomial | 已知多项式关系 | C, degree, coef0 | O(n^2) 到 O(n^3) |
| Sigmoid | 很少有用（等价于两层神经网络） | C, gamma, coef0 | O(n^2) 到 O(n^3) |
