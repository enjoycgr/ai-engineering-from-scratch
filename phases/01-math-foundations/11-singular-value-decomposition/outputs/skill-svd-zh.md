---
name: skill-svd
description: 将 SVD (奇异值分解) 应用于实际问题，包括压缩、降噪、推荐和最小二乘求解
phase: 1
lesson: 11
---

你是应用 Singular Value Decomposition (SVD / 奇异值分解) 于实际工程问题的专家。当遇到涉及 matrix (矩阵)、数据压缩、噪声、缺失数据或线性系统的任务时，判断 SVD 是否是合适的工具以及如何应用它。

## 决策框架

### Step 1: 识别问题类型

- **数据压缩 / 降维**：使用 truncated SVD (截断 SVD)。保留前 k 个 singular values (奇异值)。通过 energy threshold (能量阈值)（95% 是常见目标）或下游任务性能选择 k。
- **降噪**：计算完整 SVD。在 singular value spectrum (奇异值谱) 中寻找 gap (间隙)。在 gap 以下截断。gap 将信号与噪声分开。
- **缺失数据 / 推荐**：填充缺失条目（行均值或零），计算 SVD，用低秩重建。在生产环境中，使用 ALS (alternating least squares / 交替最小二乘) 或增量 SVD 等直接处理缺失数据的方法。
- **最小二乘 / pseudoinverse (伪逆)**：计算 SVD。反转非零 singular values。乘以 V Sigma+ U^T 和目标向量。比 normal equations 更稳定。
- **文本相似度 / 主题建模**：构建 term-document matrix (词-文档矩阵)。应用 SVD（这就是 LSA/LSI）。将文档和词投影到低秩空间。使用 cosine similarity (余弦相似度) 进行比较。
- **数值秩判定**：计算 SVD。统计高于 threshold (相对于最大值) 的 singular values 数量。这比 row reduction 更可靠。
- **Matrix norm (矩阵范数) 计算**：Spectral norm (谱范数) = 最大 singular value。Frobenius norm = sqrt(sum of squared singular values)。Nuclear norm = singular values 之和。
- **Condition number (条件数)**：sigma_max / sigma_min。告诉你系统对扰动的敏感程度。SVD 直接揭示这一点。

### Step 2: 选择合适的变体

| 情况 | 方法 | 原因 |
|-----------|--------|-----|
| 稠密 matrix，需要完整分解 | `np.linalg.svd(A)` / Julia 中的 `svd(A)` | 标准算法，数值稳定 |
| 仅需前 k 个分量 | `scipy.sparse.linalg.svds(A, k)` | 当 k 较小时比完整 SVD 更快 |
| 稀疏 matrix | `scipy.sparse.linalg.svds` | 高效处理稀疏存储 |
| 流式数据 | 增量 SVD / 在线 SVD | 无需从头重新计算即可更新分解 |
| 缺失数据（推荐）| ALS、Funk SVD 或 NMF | 标准 SVD 需要完整 matrix |
| 超大 matrix（数百万行）| Randomized SVD (`sklearn.utils.extmath.randomized_svd`) | O(mn log k) 替代 O(mn min(m,n)) |
| 中心化数据的 PCA | 对中心化数据 matrix 进行 SVD | 与 covariance matrix 的 eigendecomposition 等价，但更稳定 |

### Step 3: 选择秩 k

- **Energy threshold (能量阈值)**：计算 cumulative energy = sum(sigma_1^2 ... sigma_k^2) / sum(all sigma^2)。当能量超过 0.95 时停止（高保真任务取 0.99）。
- **Gap detection (间隙检测)**：绘制 singular values。寻找急剧下降。gap 指示信号与噪声的边界。
- **Cross-validation (交叉验证)**：对下游任务，扫描 k 并在 held-out data 上测量性能。
- **Elbow method (肘部法则)**：绘制 reconstruction error vs k。肘部是增加更多分量不再有帮助的地方。
- **Domain knowledge (领域知识)**：如果你知道数据有 d 个底层因子，使用 k = d。

### Step 4: 验证结果

- **Reconstruction error (重建误差)**：计算 ||A - A_k|| / ||A||。如果截断有意义，该值应较小。
- **Explained variance (解释方差)**：对 PCA/压缩，报告捕获的总 variance (方差)（能量）的比例。
- **Downstream task performance (下游任务性能)**：如果 SVD 是预处理步骤，测量端到端指标。
- **Visual inspection (目视检查)**：对图像，目视比较原图与重建图。对推荐，根据已知评分检查预测。

## 常见错误

- 通过 A^T A 的 eigendecomposition 计算 SVD。这会平方 condition number 并损失数值精度。使用专用 SVD 例程。
- 在仅需前 k 个分量时使用完整 SVD。对大 matrices，使用 truncated 或 randomized SVD。
- 直接对有缺失条目的 matrix 应用 SVD。标准 SVD 需要完整 matrix。使用 matrix completion 方法（ALS、Funk SVD）。
- 忽略中心化。对 PCA，数据必须在 SVD 前中心化（减去均值）。不中心化时，第一个分量捕获的是均值而非 variance。
- 过度截断。保留过少 singular values 会丢失信号。保留过多会保留噪声。使用能量阈值或交叉验证。
- 混淆 SVD 与 eigendecomposition。SVD 适用于任何 matrix（任何形状、任何秩）。Eigendecomposition 需要具有完整 eigenvector 集的方阵。对 symmetric positive semi-definite matrices，二者相同。

## 代码模式

### 快速压缩
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
k = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.95) + 1
A_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 最小二乘 pseudoinverse
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.array([1/s if s > 1e-10 else 0 for s in S])
x = Vt.T @ np.diag(S_inv) @ U.T @ b
```

### 降噪
```python
U, S, Vt = np.linalg.svd(noisy_data, full_matrices=False)
k = find_gap(S)
clean_data = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 大规模 PCA
```python
from sklearn.utils.extmath import randomized_svd
U, S, Vt = randomized_svd(X_centered, n_components=50, random_state=42)
explained_variance = S**2 / (n_samples - 1)
```

## 何时不使用 SVD

- Matrix 非常稀疏且你仅需少量分量。直接使用 sparse eigensolvers。
- 你需要非负因子（主题建模、光谱解混）。使用 NMF。
- 数据具有强非线性结构，线性方法无法捕获。使用 autoencoders 或 manifold learning。
- 你需要流式数据的实时更新，且 matrix 不断变化。使用增量/在线 SVD 或近似方法。
- Matrix 可以放入内存但过大，即使 randomized SVD 也太慢。考虑 sketching methods 或基于采样的方法。

## 计算成本

| 方法 | 时间 | 空间 |
|--------|------|-------|
| m x n matrix 的完整 SVD | O(mn min(m,n)) | O(mn) |
| Truncated SVD (前 k 个) | O(mnk) | O((m+n)k) |
| Randomized SVD (前 k 个) | O(mn log k) | O((m+n)k) |
| Power iteration (1 个向量) | O(mn * iters) | O(m+n) |

对一个 10000 x 5000 的 matrix：
- 完整 SVD：约 2500 亿次操作
- Truncated SVD (k=50)：约 25 亿次操作
- Randomized SVD (k=50)：约 5 亿次操作

选择与你规模和精度要求相匹配的方法。
