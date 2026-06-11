---
name: skill-imbalanced-data
description: 处理不平衡分类问题的决策清单
version: 1.0.0
phase: 2
lesson: 17
tags: [imbalanced-data (不平衡数据), smote, class-weights (类别权重), threshold-tuning (阈值调优), evaluation (评估)]
---

# 不平衡数据策略 (Imbalanced Data Strategy)

处理不平衡分类 (imbalanced classification) 的决策清单。按照以下顺序为你的问题选择合适的方法。

## Step 1: 衡量不平衡程度 (Measure the imbalance)

- 统计每个类别的样本数
- 计算 imbalance ratio (不平衡比例) (majority / minority)
- Mild (轻度): ratio < 3:1 (例如 70/30)
- Moderate (中度): ratio 3:1 到 20:1 (例如 95/5)
- Severe (重度): ratio > 20:1 (例如 99/1)

## Step 2: 选择合适的 metric (指标)

对于不平衡数据集，优先使用 precision (精确率) / recall (召回率) / F1 score (F1 分数) 而非 accuracy (准确率)。根据你的问题选择：

| Situation (场景) | Primary Metric (主要指标) | Secondary Metric (次要指标) |
|-----------|---------------|-----------------|
| 漏掉 positive (正类) 的代价很高（欺诈、疾病） | Recall (召回率) | F2 score (F2 分数) |
| False alarms (误报) 的代价很高（垃圾邮件过滤、推荐） | Precision (精确率) | F0.5 score (F0.5 分数) |
| 两者大致同等重要 | F1 score (F1 分数) | MCC |
| 需要一个单一的 ranking metric (排序指标) | AUPRC | AUC-ROC |
| 需要跨数据集比较 | MCC | AUPRC |

## Step 3: 选择重平衡策略 (Choose a rebalancing strategy)

### 根据不平衡严重程度 (By imbalance severity)

| Imbalance (不平衡程度) | First Try (首先尝试) | Second Try (其次尝试) | Avoid (避免) |
|-----------|-----------|------------|-------|
| Mild (< 3:1) | Class weights (类别权重) | Threshold tuning (阈值调优) | Oversampling (过采样) (不必要) |
| Moderate (3:1 to 20:1) | SMOTE + class weights (类别权重) | Threshold tuning (阈值调优) on top | Undersampling (欠采样) (数据损失太大) |
| Severe (> 20:1) | SMOTE + class weights (类别权重) + threshold (阈值) | Ensemble with balanced bagging (平衡 bagging 集成) | Undersampling (欠采样) alone |

### 根据数据集大小 (By dataset size)

| Dataset Size (数据集大小) | Preferred Strategy (首选策略) | Reason (原因) |
|-------------|-------------------|--------|
| < 1,000 samples | Oversampling (过采样) or SMOTE | 无法承受丢失 majority data (多数类数据) |
| 1,000 - 10,000 | SMOTE + threshold tuning (阈值调优) | 有足够的 minority samples (少数类样本) 用于 k-NN |
| > 10,000 | Class weights (类别权重) or undersampling (欠采样) | 快速，minority data (少数类数据) 充足 |

## Step 4: 应用技术 (Apply the technique)

### Class weights (类别权重) (总是首先尝试)
- 在 sklearn 中: `class_weight='balanced'`
- 不需要修改数据
- 适用于任何基于 loss (损失) 的模型
- 在期望上等价于 oversampling (过采样)

### SMOTE
- 只应用于 training data (训练数据)（绝不用于 test/validation，测试/验证）
- 使用 k=5 neighbors (默认)
- 与 class weights (类别权重) 结合使用以获得最佳效果
- 注意 decision boundary (决策边界) 附近的 noisy synthetic points (噪声合成点)

### Threshold tuning (阈值调优)
- 训练模型，在 validation set (验证集) 上获取预测概率
- 从 0.05 到 0.95 扫描 thresholds (阈值)
- 选择能最大化你选定 metric (指标) 的 threshold (阈值)
- 始终在 validation data (验证数据) 上调优，绝不使用 test data (测试数据)

## Step 5: 正确验证 (Validate properly)

- 使用 stratified cross-validation (分层交叉验证)（在每个 fold 中保持类别比例）
- 在原始的（未重采样的）test set (测试集) 上报告 metrics (指标)
- 绝不要在 split (划分) 前应用 SMOTE——只在 training folds (训练折) 上应用
- 与"总是预测 majority (多数类)"的 baseline (基线) 进行比较

## Step 6: 常见错误 (Common mistakes to avoid)

- 在 train/test split (训练/测试划分) 前对整个数据集应用 SMOTE（data leakage，数据泄露）
- 使用 accuracy (准确率) 作为 evaluation metric (评估指标)
- 不先尝试 class weights (类别权重)（最简单的方法，通常足够）
- 进行 oversampling (过采样) 后再做 cross-validation (交叉验证)（synthetic points，合成点会泄露到不同 fold 之间）
- 忽略 threshold tuning (阈值调优)（免费的性能提升，无需重新训练）
- 在小数据集上使用 random undersampling (随机欠采样)（丢弃了太多数据）

## 快速决策树 (Quick Decision Tree)

1. Imbalance ratio (不平衡比例) < 3:1? -> 只尝试 class weights (类别权重)
2. 数据集 > 10,000 个样本? -> Class weights (类别权重) + threshold tuning (阈值调优)
3. 数据集 < 1,000 个样本? -> SMOTE + class weights (类别权重)
4. 其他情况 -> SMOTE + class weights (类别权重) + threshold tuning (阈值调优)
5. 仍然不够好? -> Balanced bagging ensemble (平衡 bagging 集成)
