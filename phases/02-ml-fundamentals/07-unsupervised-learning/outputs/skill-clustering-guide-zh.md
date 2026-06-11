---
name: skill-clustering-guide
description: 根据数据形状、噪声和约束条件选择合适的聚类算法
version: 1.0.0
phase: 2
lesson: 7
tags: [clustering, k-means, dbscan, hierarchical, gmm, unsupervised]
---

# 聚类算法选择指南

聚类 (clustering) 没有单一的"最佳"算法。正确的选择取决于聚类形状、是否知道聚类数量、数据中的噪声量以及数据集的大小。

## 决策清单

1. 你知道聚类的数量吗？
   - 是：K-Means 或 GMM
   - 否：DBSCAN（自动发现聚类），或 hierarchical clustering（层次聚类，在不同层级切割 dendrogram）

2. 聚类是什么形状？
   - 大致球形（blob 状）：K-Means
   - 椭圆形且大小不同：GMM
   - 任意形状（新月形、环形、链状）：DBSCAN
   - 嵌套或层次结构：hierarchical clustering

3. 数据包含噪声或异常值吗？
   - 是：DBSCAN（显式标记 noise point）或 GMM（低概率点为异常值）
   - 否：K-Means 即可

4. 你需要软分配（概率）吗？
   - 是：GMM 给出每个聚类的 P(cluster | data point)
   - 否：K-Means 或 DBSCAN 给出硬分配

5. 数据集有多大？
   - 少于 10,000：任何算法都适用
   - 10,000 到 1,000,000：K-Means（快），Mini-Batch K-Means（更快）
   - 超过 1,000,000：Mini-Batch K-Means 或 BIRCH。Hierarchical clustering 太慢。

## 何时使用每种方法

**K-Means**：默认的起点。快（O(n * k * iterations)），简单，对许多问题足够好。使用 elbow method 或 silhouette score 选择 K。局限性：假设球形聚类，对初始化敏感（使用 K-Means++ 或多次运行），不能很好地处理大小差异大的聚类。

**DBSCAN**：最适合发现任意形状的聚类并自动检测异常值。两个参数：eps（邻域半径）和 min_samples（最小密度）。不需要指定 K。局限性：在密度差异很大的聚类上表现不佳，调整 eps 可能棘手。使用 k-distance plot 估计 eps：计算每个点到其第 k 近邻居的距离，排序，寻找肘部。

**Hierarchical (Agglomerative)**：构建合并的树。当你想以多个粒度探索聚类结构时很有用（在不同高度切割 dendrogram）。Ward's linkage 最适合紧凑聚类。Single linkage 发现细长聚类但对噪声敏感。局限性：O(n^2) 内存和 O(n^3) 时间，因此对大数据集不实用。

**GMM (Gaussian Mixture Models)**：带有概率分配的软聚类。将每个聚类建模为具有自己 mean 和 covariance 的高斯分布。当聚类是椭圆形或重叠时，比 K-Means 更好。使用 BIC (Bayesian Information Criterion) 选择分量数量。局限性：假设高斯分布，在非凸形状上可能失败，对初始化敏感。

## 评估聚类质量（无标签）

| 指标 | 衡量内容 | 范围 | 适用场景 |
|------|---------|------|---------|
| Silhouette score | 紧密度 vs 分离度 | -1 到 1（越高越好） | 比较 K 值或算法 |
| Inertia（簇内平方和） | 聚类的紧密程度 | 0 到 inf（越低越好） | K-Means 的 elbow method |
| BIC / AIC | 带复杂度惩罚的模型拟合 | 越低越好 | 选择 GMM 分量数 |
| Calinski-Harabasz index | 组间与组内方差比 | 越高越好 | 快速比较 |
| Davies-Bouldin index | 聚类间平均相似度 | 越低越好 | 惩罚重叠聚类 |

## 常见错误

- 运行 K-Means 前不进行特征缩放（大尺度特征主导距离计算）
- 在二维中肉眼观察数据来选择 K，而实际数据是高维的（使用 silhouette score）
- 在非球形聚类上使用 K-Means（新月形或环形数据需要 DBSCAN）
- 将 DBSCAN 的 eps 设置得太大（所有点在一个聚类中）或太小（所有点都是噪声）
- 将聚类标签视为 ground truth（聚类是探索性的；用领域知识验证）
- 在超过 20,000 个点的数据集上运行 hierarchical clustering（内存和时间爆炸）

## 快速参考

| 算法 | 聚类形状 | 自动发现 K | 处理噪声 | 软分配 | 可扩展性 |
|------|---------|-----------|---------|--------|---------|
| K-Means | 球形 | 否（你设置 K） | 否 | 否 | 数百万 |
| Mini-Batch K-Means | 球形 | 否 | 否 | 否 | 数千万 |
| DBSCAN | 任意 | 是 | 是 | 否 | 数十万 |
| Hierarchical | 任意（取决于 linkage） | 灵活（切割 dendrogram） | 取决于 linkage | 否 | 少于 20k |
| GMM | 椭圆形 | 否（你设置 K） | 部分（低概率） | 是 | 少于 100k |
| HDBSCAN | 任意 | 是 | 是 | 部分 | 数十万 |
