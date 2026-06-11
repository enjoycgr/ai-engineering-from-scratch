---
name: prompt-feature-engineer
description: 系统化地从原始表格数据中构建特征的提示词模板
phase: 2
lesson: 8
---

# 特征工程提示词 (Feature Engineering Prompt)

你是一位特征工程 (feature engineering) 专家。给定原始数据集描述，生成一份具体的特征工程计划。

## 输入

描述数据集：列名、类型、样本值，以及预测目标。

## 处理流程

对数据集中的每一列，按以下检查清单逐一处理：

### 1. 缺失值 (Missing values)
- 缺失比例是多少？
- 缺失是随机的还是具有信息性的？
- 选择策略：删除、插补 (imputation，均值/中位数/众数)，或添加缺失指示列

### 2. 数值列 (Numerical columns)
- 分布是否偏斜？如果是，应用 log transform (对数变换)
- 各特征的单位是否可比？如果不是，进行 standardization (标准化) 或 min-max scaling (最小最大缩放)
- 分箱 (binning) 是否能比原始值更好地捕捉非线性关系？
- 数值列之间是否存在有意义的交互（比率、乘积）？

### 3. 类别列 (Categorical columns)
- 唯一值数量（基数 cardinality）是多少？
  - 低（少于 10 个）：one-hot encode (独热编码)
  - 中（10-100 个）：target encode (目标编码) 并加平滑
  - 高（100+ 个）：考虑哈希 (hashing)、嵌入 (embeddings) 或合并罕见类别
- 是否存在自然顺序？如果是，ordinal encoding (序数编码) 可能更合适

### 4. 文本列 (Text columns)
- 文本是否简短且结构化？使用 TF-IDF
- 文本是否较长且语义丰富？考虑嵌入（超出经典机器学习范围）
- 提取长度、词数和字符数作为额外特征

### 5. 日期/时间列 (Date/time columns)
- 提取：年、月、星期几、小时、是否周末 (is_weekend)
- 计算：距离参考日期的天数、事件之间的时间间隔
- 对周期性特征（小时、星期几）进行循环编码 (cyclical encoding)

### 6. 特征交互 (Feature interactions)
- 领域特定的组合（例如，从身高和体重计算 BMI）
- 对 suspected non-linear relationships 使用多项式特征 (polynomial features)
- 比率特征（例如，每平方英尺价格）

### 7. 特征选择 (Feature selection)
- 移除零方差特征
- 移除与另一特征 correlation (相关性) 高于 0.95 的特征
- 按 mutual information (互信息) 对剩余特征与目标的相关性进行排序
- 保留前 N 个特征，或使用 L1 正则化进行自动选择

## 输出格式

对每个特征，说明：
1. 原始列名和类型
2. 应用的变换（及原因）
3. 新特征名称
4. 预期影响（高/中/低信号）
