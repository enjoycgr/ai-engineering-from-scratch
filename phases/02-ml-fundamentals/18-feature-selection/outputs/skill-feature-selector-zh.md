---
name: skill-feature-selector
description: 快速参考决策树，用于选择合适的特征选择方法
version: 1.0.0
phase: 2
lesson: 18
tags: [feature-selection, mutual-information, rfe, lasso, tree-importance]
---

# Feature Selection Strategy (特征选择策略)

快速参考：如何挑选并应用正确的 feature selection method (特征选择方法)。

## Step 1: Start with cleanup (先做清理)

在应用任何方法之前，先移除明显无用的特征：

- **Constant features (常数特征)**：variance (方差) = 0。移除它们。
- **Near-constant features (近常数特征)**：variance (方差) < 0.01（或你设定的阈值）。移除它们。
- **Duplicate features (重复特征)**：完全相同的列。保留一个，丢弃其余。
- **ID columns (ID列)**：每行唯一，不携带可泛化信息。移除它们。

这只需几秒钟，却能在混乱的真实数据集中消除 10-30% 的特征。

## Step 2: Choose a method based on your situation (根据情况选择方法)

### Quick Decision Tree (快速决策树)

1. **特征 < 50 个？** 从 mutual information ranking (互信息排序) 开始。保留 Top K。
2. **特征 50-500 个？** 先用 variance threshold (方差阈值)，若使用线性模型则用 L1 (Lasso)，若使用树模型则用 tree importance (树重要性)。
3. **特征 > 500 个？** 链式组合方法：variance threshold (方差阈值) -> mutual information filter (互信息过滤，保留前 50%) -> 在幸存者上运行 RFE。
4. **需要可解释性？** L1 regularization (L1正则化) 给出精确的零/非零结果。Tree importance (树重要性) 给出排序分数。
5. **需要捕获非线性关系？** Mutual information (互信息) 或 tree-based importance (基于树的重要性)。避免 L1（仅限线性）。
6. **需要特征交互？** RFE 或 tree-based importance (基于树的重要性)。Filter methods (过滤法) 会遗漏交互。

### Method Reference (方法参考表)

| Method | When to Use | When to Avoid |
|--------|------------|---------------|
| Variance threshold (方差阈值) | 总是作为第一步 | 永远不要跳过这一步 |
| Mutual information (互信息) | 快速排序、非线性关系 | 当你需要检测特征交互时 |
| RFE (递归特征消除) | 彻底选择、中等特征数量 | 非常昂贵的模型、> 1000 个特征 |
| L1 / Lasso | 线性模型、快速嵌入选择 | 非线性问题、高度相关特征 |
| Tree importance (树重要性) | 非线性关系、特征交互 | 受高基数特征偏置影响 |
| Permutation importance (置换重要性) | 模型无关验证、最终检查 | 初始筛选时太慢 |

## Step 3: Validate your selection (验证你的选择)

- 比较使用选中特征与全部特征的模型性能
- 使用 cross-validation (交叉验证)，而非单次 train/test split (训练/测试划分)
- 如果性能下降超过 1-2%，你可能移除了有用的特征
- 如果性能提升，说明你成功移除了噪声

## Step 4: Handle common pitfalls (处理常见陷阱)

### Correlated features (相关特征)
- L1 会从一组相关特征中任意挑选一个，其余置零
- 先计算 correlation matrix (相关矩阵)，再决定保留哪些相关特征
- Tree importance (树重要性) 会在相关特征间分散重要性

### Data leakage (数据泄露)
- 仅在 training data (训练数据) 上拟合特征选择
- 对 test data (测试数据) 应用相同的选择
- 在 cross-validation (交叉验证) 中，特征选择必须在每个 fold (折) 内部进行

### Overfitting to feature selection (对特征选择过拟合)
- 迭代次数过多的 RFE 可能对训练集过拟合
- 在 held-out data (留出数据) 上验证，而非用于选择的数据
- 使用 stability selection (稳定性选择)（在子样本上重复）以获得更稳健的结果

## Step 5: Production checklist (生产环境检查清单)

- [ ] Variance threshold (方差阈值) 已作为第一级过滤器应用
- [ ] 特征选择仅在 training data (训练数据) 上拟合
- [ ] 已记录选中的特征（名称、使用的方法、分数）
- [ ] 已比较性能：选中特征 vs 全部特征
- [ ] 使用 cross-validation (交叉验证) 而非单 split (划分) 评估
- [ ] 特征选择已集成到训练 pipeline (流水线) 中（非手动操作）
- [ ] 已设置 feature drift (特征漂移) 监控（选中的特征可能变得过时）
