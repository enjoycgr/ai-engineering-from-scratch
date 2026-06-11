# Feature Selection (特征选择)

> 更多的特征并非更好。正确的特征才是更好。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 2, Lessons 01-09, 08 (feature engineering)
**Time:** ~75 minutes

## Learning Objectives (学习目标)

- 从零实现 filter method (过滤法)（variance threshold (方差阈值)、mutual information (互信息)、chi-squared test (卡方检验)）和 wrapper method (包装法)（RFE (递归特征消除)、forward selection (前向选择)）
- 解释为什么 mutual information (互信息) 能够捕获 correlation (相关性) 无法检测的 nonlinear feature-target relationships (非线性特征-目标关系)
- 比较 L1 regularization (L1正则化)（embedded selection (嵌入法选择)）与 RFE (递归特征消除)（wrapper selection (包装法选择)），并评估它们的计算 tradeoffs (权衡)
- 构建一个 feature selection pipeline (特征选择流水线)，结合多种方法，并展示在 held-out data (留出数据) 上 improved generalization (改进的泛化能力)

## The Problem (问题)

你有 500 个特征。你的模型训练缓慢， constantly overfits (持续过拟合)，没人能解释它学到了什么。你添加更多特征希望提升性能。结果变得更糟。

这就是 curse of dimensionality (维度灾难) 的实际表现。随着特征数量的增长，特征空间的体积呈指数级爆炸。数据点变得稀疏。点之间的距离趋于收敛。模型需要指数级更多的数据才能找到真正的模式。Noise features (噪声特征) 淹没了 signal features (信号特征)。Overfitting (过拟合) 成为默认状态。

Feature selection (特征选择) 是解药。剥离噪声。消除冗余。保留那些真正携带关于目标信息的特征。结果是：更快的训练、更好的泛化，以及你真正能解释的模型。

目标不是使用所有可用信息。而是使用正确的信息。

## The Concept (概念)

### Three Categories of Feature Selection (特征选择的三个类别)

每种 feature selection method (特征选择方法) 都属于以下三个类别之一：

```mermaid
flowchart TD
    A[Feature Selection Methods] --> B[Filter Methods]
    A --> C[Wrapper Methods]
    A --> D[Embedded Methods]

    B --> B1["Variance Threshold"]
    B --> B2["Mutual Information"]
    B --> B3["Chi-squared Test"]
    B --> B4["Correlation Filtering"]

    C --> C1["Recursive Feature Elimination"]
    C --> C2["Forward Selection"]
    C --> C3["Backward Elimination"]

    D --> D1["L1 / Lasso Regularization"]
    D --> D2["Tree-based Importance"]
    D --> D3["Elastic Net"]
```

**Filter methods (过滤法)** 使用统计度量独立地为每个特征打分。它们不使用模型。速度快，但会遗漏 feature interactions (特征交互)。

**Wrapper methods (包装法)** 训练模型来评估特征子集。它们使用模型性能作为评分。结果更好，但代价高昂，因为它们需要多次重新训练模型。

**Embedded methods (嵌入法)** 在模型训练过程中选择特征。L1 regularization (L1正则化) 将权重推向零。Decision trees (决策树) 在最优特征上分裂。选择发生在拟合过程中，而非作为单独的步骤。

### Variance Threshold (方差阈值)

最简单的 filter (过滤器)。如果一个特征在样本间几乎不变，它几乎不携带任何信息。

考虑一个特征，在 1000 个样本中有 999 个为 0.0。它的 variance (方差) 接近零。没有模型能用它来区分类别。移除它。

```
variance(x) = mean((x - mean(x))^2)
```

设置一个阈值（例如 0.01）。丢弃所有 variance (方差) 低于该阈值的特征。这可以在完全不查看目标变量的情况下移除恒定或近恒定特征。

何时使用：作为其他方法之前的预处理步骤。它以接近零的代价捕获明显无用的特征。

局限性：一个特征可以有很高的 variance (方差)，却仍然是纯噪声。Variance threshold (方差阈值) 是必要但不充分的。

### Mutual Information (互信息)

Mutual information (互信息) 衡量知道特征 X 的值后，能减少多少关于目标 Y 的不确定性。

```
I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))
```

如果 X 和 Y 独立，p(x, y) = p(x) * p(y)，则 log 项为零，I(X; Y) = 0。X 告诉你的关于 Y 的信息越多，mutual information (互信息) 越高。

相比 correlation (相关性) 的关键优势：mutual information (互信息) 能捕获 nonlinear relationships (非线性关系)。一个特征可能与目标有零 correlation (相关性)，却有很高的 mutual information (互信息)，因为关系是 quadratic (二次的) 或 periodic (周期的)。

对于连续特征，先离散化为 bins (分箱)（基于 histogram 的估计）。Bins (分箱) 的数量会影响估计——太少会丢失信息，太多会引入噪声。常见选择：sqrt(n) 个 bins 或 Sturges' rule (1 + log2(n))。

```mermaid
flowchart LR
    A[Feature X] --> B[Discretize into Bins]
    B --> C["Compute Joint Distribution p(x,y)"]
    C --> D["Compute MI = sum p(x,y) * log(p(x,y) / p(x)p(y))"]
    D --> E["Rank Features by MI Score"]
    E --> F[Select Top K]
```

### Recursive Feature Elimination (RFE) (递归特征消除)

RFE 是一种 wrapper method (包装法)。它使用模型自身的 feature importance (特征重要性) 进行迭代剪枝：

1. 用所有特征训练模型
2. 按重要性对特征排序（线性模型用 coefficients (系数)，树用 impurity reduction (不纯度减少)）
3. 移除最不重要的特征
4. 重复直到剩余目标数量的特征

```mermaid
flowchart TD
    A["Start: All N Features"] --> B["Train Model"]
    B --> C["Rank Feature Importances"]
    C --> D["Remove Least Important"]
    D --> E{"Features == Target Count?"}
    E -->|No| B
    E -->|Yes| F["Return Selected Features"]
```

RFE 考虑 feature interactions (特征交互)，因为模型同时看到所有剩余特征。移除一个特征会改变其他特征的重要性。这使它比 filter methods (过滤法) 更彻底。

代价：你需要训练模型 N - target 次。500 个特征目标为 10，那就是 490 次训练。对于昂贵的模型，这很慢。你可以通过每步移除多个特征来加速（例如，每轮移除底部 10%）。

### L1 (Lasso) Regularization (L1正则化)

L1 regularization (L1正则化) 将权重的绝对值加到 loss function (损失函数) 中：

```
loss = prediction_error + alpha * sum(|w_i|)
```

alpha 参数控制特征被剪枝的激进程度。更高的 alpha 意味着更多权重精确变为零。

为什么精确为零？L1 penalty (L1惩罚) 在权重空间中创建一个 diamond-shaped (菱形) 的约束区域。最优解倾向于落在这个菱形的 corner (角) 上，那里一个或多个权重为零。L2 regularization (L2正则化)（ridge (岭回归)）创建一个 circular (圆形) 约束，权重会收缩但很少精确为零。

这是 embedded feature selection (嵌入法特征选择)：模型在训练过程中学习忽略哪些特征。权重为零的特征实际上被移除了。

优点：单次训练，处理 correlated features (相关特征)（选择一个，其他置零），内置于大多数 linear model (线性模型) 实现中。

局限性：仅适用于 linear models (线性模型)。无法捕获 nonlinear feature importance (非线性特征重要性)。

### Tree-Based Feature Importance (基于树的特征重要性)

Decision trees (决策树) 及其 ensembles (集成模型)（random forests (随机森林)、gradient boosting (梯度提升)）天然地对特征进行排序。每次 split (分裂) 都会减少 impurity (不纯度)（分类用 Gini 或 entropy (熵)，回归用 variance (方差)）。产生更大 impurity reduction (不纯度减少) 的特征更重要。

对于一个包含 T 棵树的 random forest (随机森林)：

```
importance(feature_j) = (1/T) * sum over all trees of
    sum over all nodes splitting on feature_j of
        (n_samples * impurity_decrease)
```

这给出了每个特征的 normalized importance score (归一化重要性分数)。它自动处理 nonlinear relationships (非线性关系) 和 feature interactions (特征交互)。

注意：tree-based importance (基于树的重要性) 偏向于具有许多唯一值的特征（high cardinality (高基数)）。一个随机 ID 列会显得很重要，因为它能完美分裂每个样本。使用 permutation importance (置换重要性) 作为 sanity check (合理性检查)。

### Permutation Importance (置换重要性)

一种 model-agnostic (模型无关) 的方法：

1. 训练模型并记录 validation data (验证数据) 上的 baseline performance (基线性能)
2. 对于每个特征：随机打乱其值，测量性能下降
3. 下降越大，特征越重要

如果打乱一个特征不会损害性能，模型不依赖它。如果性能崩溃，该特征至关重要。

Permutation importance (置换重要性) 避免了 tree-based importance (基于树的重要性) 的 cardinality bias (基数偏差)。但它很慢：每个特征一次完整评估，为稳定性重复多次。

### Comparison Table (对比表)

| Method | Type | Speed | Nonlinear | Feature Interactions |
|--------|------|-------|-----------|---------------------|
| Variance threshold | Filter | Very fast | No | No |
| Mutual information | Filter | Fast | Yes | No |
| Correlation filter | Filter | Fast | No | No |
| RFE | Wrapper | Slow | Depends on model | Yes |
| L1 / Lasso | Embedded | Fast | No (linear) | No |
| Tree importance | Embedded | Medium | Yes | Yes |
| Permutation importance | Model-agnostic | Slow | Yes | Yes |

### Decision Flowchart (决策流程图)

```mermaid
flowchart TD
    A[Start: Feature Selection] --> B{How many features?}
    B -->|"< 50"| C["Start with variance threshold + mutual information"]
    B -->|"50-500"| D["Variance threshold, then L1 or tree importance"]
    B -->|"> 500"| E["Variance threshold, then mutual info filter, then RFE on survivors"]

    C --> F{Using linear model?}
    D --> F
    E --> F

    F -->|Yes| G["L1 regularization for final selection"]
    F -->|No - trees| H["Tree importance + permutation importance"]
    F -->|No - other| I["RFE with your model"]

    G --> J[Validate: compare selected vs all features]
    H --> J
    I --> J

    J --> K{Performance improved?}
    K -->|Yes| L["Ship with selected features"]
    K -->|No| M["Try different method or keep all features"]
```

## Build It (动手实现)

### Step 1: Generate synthetic data with known feature structure (生成具有已知特征结构的合成数据)

```python
import numpy as np


def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)

    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names
```

我们知道 ground truth (真实标签)：特征 0-4 是 informative (有信息的)（其中 3 和 4 是 0 和 1 的 correlated copies (相关副本)），特征 5-9 与 informative features (有信息特征) 相关，特征 10-19 是纯 noise (噪声)。一个好的选择方法应该将 0-4 排得最高，10-19 排得最低。

### Step 2: Variance threshold (方差阈值)

```python
def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances
```

### Step 3: Mutual information (discrete) (互信息——离散)

```python
def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores
```

### Step 4: Recursive Feature Elimination (递归特征消除)

```python
def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings
```

### Step 5: L1 feature selection (L1特征选择)

```python
def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w
```

### Step 6: Tree-based importance (simple decision tree) (基于树的重要性——简单决策树)

```python
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0

    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            continue

        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        feature_subset = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances
```

### Step 7: Run all methods and compare (运行所有方法并比较)

代码文件在相同的合成数据集上运行所有五种方法，并打印一个比较表，显示每种方法选择了哪些特征。

## Use It (使用它)

使用 scikit-learn，feature selection (特征选择) 内置于 pipeline (流水线) 中：

```python
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)
X_rfe = rfe_selector.transform(X)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)
X_lasso = lasso_selector.transform(X)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```

从零实现的版本精确展示了每种方法内部发生了什么。Variance threshold (方差阈值) 只是计算 `var(X, axis=0)` 并应用 mask (掩码)。Mutual information (互信息) 是在 contingency table (列联表) 中计数 joint (联合) 和 marginal frequencies (边际频率)。RFE 是一个训练、排序、剪枝的循环。L1 是带 soft-thresholding (软阈值) 步骤的 gradient descent (梯度下降)。Tree importance (树重要性) 累积跨 splits (分裂) 的 impurity reductions (不纯度减少)。没有魔法——只是统计和循环。

sklearn 版本增加了鲁棒性（例如，mutual_info_classif 使用 k-NN density estimation (密度估计) 而非 binning (分箱)）、速度（C 实现）和 pipeline (流水线) 集成。

## Ship It (交付)

本课程产出：
- `outputs/skill-feature-selector.md` —— 一个快速参考决策树，用于选择正确的 feature selection method (特征选择方法)

## Exercises (练习)

1. **Forward selection (前向选择)**：实现 RFE 的反向操作。从零个特征开始。每一步，添加能最大程度提升模型性能的特征。当添加特征不再有帮助时停止。将选中的特征与 RFE 结果进行比较。哪个更快？哪个结果更好？

2. **Stability selection (稳定性选择)**：运行 L1 feature selection (L1特征选择) 50 次，每次在数据的随机 80% 子样本上，使用略微不同的 alpha 值。统计每个特征被选中的频率。在 > 80% 的运行中被选中的特征是"稳定的"。将稳定特征与单次 L1 选择进行比较。哪个更可靠？

3. **Multicollinearity detection (多重共线性检测)**：计算所有特征的 correlation matrix (相关矩阵)。实现一个函数，给定 correlation threshold (相关性阈值)（例如 0.9），从每个高度相关的特征对中移除一个特征（保留与目标 mutual information (互信息) 更高的那个）。在合成数据集上测试并验证它移除了冗余的相关特征。

4. **Feature selection pipeline (特征选择流水线)**：将 variance threshold (方差阈值)、mutual information filter (互信息过滤器) 和 RFE 串联成一个 pipeline (流水线)。首先移除近零方差特征，然后保留 mutual information (互信息) 最高的 50%，最后在幸存者上运行 RFE。将此 pipeline (流水线) 与单独在所有特征上运行 RFE 进行比较。Pipeline (流水线) 更快吗？同样准确吗？

5. **Permutation importance from scratch (从零实现置换重要性)**：实现 permutation importance (置换重要性)。对于每个特征，打乱其值 10 次，测量 F1 score 的平均下降。将排序与 tree-based importance (基于树的重要性) 进行比较。找出它们不一致的情况并解释原因（提示：correlated features (相关特征)）。

## Key Terms (关键术语)

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Filter method (过滤法) | "Score features independently" | A feature selection approach that ranks features using a statistical measure without training a model, evaluating each feature in isolation |
| Wrapper method (包装法) | "Use the model to pick features" | A feature selection approach that evaluates feature subsets by training a model and using its performance as the selection criterion |
| Embedded method (嵌入法) | "The model selects features during training" | Feature selection that happens as part of model fitting, such as L1 regularization driving weights to zero |
| Mutual information (互信息) | "How much one variable tells you about another" | A measure of the reduction in uncertainty about Y given knowledge of X, capturing both linear and nonlinear dependencies |
| Recursive Feature Elimination (RFE) (递归特征消除) | "Train, rank, prune, repeat" | An iterative wrapper method that trains a model, removes the least important feature(s), and repeats until a target count is reached |
| L1 / Lasso regularization (L1/Lasso正则化) | "Penalty that kills features" | Adding the sum of absolute weight values to the loss function, which drives unimportant feature weights to exactly zero |
| Variance threshold (方差阈值) | "Remove constant features" | Dropping features whose variance across samples falls below a specified threshold, filtering out features that carry no information |
| Feature importance (特征重要性) | "Which features matter most" | A score indicating how much each feature contributes to model predictions, computed from split gains (trees) or coefficient magnitudes (linear) |
| Permutation importance (置换重要性) | "Shuffle and measure the damage" | Evaluating feature importance by randomly shuffling each feature's values and measuring the resulting drop in model performance |
| Curse of dimensionality (维度灾难) | "Too many features, not enough data" | The phenomenon where adding features increases the volume of the feature space exponentially, making data sparse and distances meaningless |

## Further Reading (延伸阅读)

- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)](https://jmlr.org/papers/v3/guyon03a.html) —— feature selection methods (特征选择方法) 的基础综述，至今仍被广泛引用
- [scikit-learn Feature Selection Guide](https://scikit-learn.org/stable/modules/feature_selection.html) —— filter (过滤法)、wrapper (包装法) 和 embedded (嵌入法) 的实用参考，附带代码示例
- [Stability Selection (Meinshausen & Buhlmann, 2010)](https://arxiv.org/abs/0809.2932) —— 将 subsampling (子采样) 与 feature selection (特征选择) 结合，以获得稳健、可复现的结果
- [Beware Default Random Forest Importances (Strobl et al., 2007)](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25) —— 展示了 tree-based importance (基于树的重要性) 中的 cardinality bias (基数偏差)，并提出了 conditional importance (条件重要性) 作为替代方案
