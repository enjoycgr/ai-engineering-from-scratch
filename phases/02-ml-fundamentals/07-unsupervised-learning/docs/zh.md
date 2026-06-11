# 无监督学习 (Unsupervised Learning)

> 没有标签，没有老师。算法自己发现结构。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Norms & Distances, Probability & Distributions), Phase 2 Lessons 1-6
**Time:** ~90 分钟

## 学习目标

- 从零实现 K-Means (K均值)、DBSCAN 和 Gaussian Mixture Model (高斯混合模型，GMM)，并比较它们的聚类行为
- 使用 silhouette score (轮廓系数) 和 elbow method (肘部法则) 评估聚类质量并选择最优 K
- 解释 DBSCAN 何时优于 K-Means，并识别哪种算法能处理非球形聚类和异常值
- 使用聚类方法构建异常检测 (anomaly detection) 流程，标记偏离正常模式的点

## 问题

到目前为止，每节 ML 课都假设有标签数据："这是输入，这是正确输出。" 在现实世界中，标签很昂贵。一家医院有数百万份患者记录，但没有人手动为每一份标记疾病类别。一个电商网站有数百万次用户会话，但没有人手动标注客户细分。一个安全团队有网络日志，但没有人标记每一个异常。

无监督学习 (unsupervised learning) 无需被告知要找什么就能发现模式。它将相似的数据点分组，发现隐藏的结构，并浮现异常。如果说监督学习是从带有答案的教科书中学习，那么无监督学习就是盯着原始数据直到模式自己显现。

难点在于：没有标签，你无法直接衡量"对"或"错"。你需要不同的工具来评估算法发现的结构是否有意义。

## 概念

### 聚类 (Clustering)：将相似的事物分组

聚类 (clustering) 将每个数据点分配到一个组（聚类），使得同一组内的点彼此之间的相似度高于与其他组内点的相似度。问题始终是："相似"是什么意思？

```mermaid
flowchart LR
    A[原始数据] --> B{选择方法}
    B --> C[K-Means]
    B --> D[DBSCAN]
    B --> E[层次聚类]
    B --> F[GMM]
    C --> G[扁平的球形聚类]
    D --> H[任意形状，噪声检测]
    E --> I[嵌套聚类的树]
    F --> J[软分配，椭圆形聚类]
```

### K-Means：主力算法

K-Means (K均值) 将数据划分为恰好 K 个聚类。每个聚类有一个 centroid (质心，即其质量中心)，每个点属于最近的质心。

Lloyd 算法：

1. 随机选择 K 个点作为初始质心
2. 将每个数据点分配到最近的质心
3. 将每个质心重新计算为其分配点的均值
4. 重复步骤 2-3，直到分配不再变化

目标函数 inertia (惯性) 衡量每个点到其分配质心的总平方距离。K-Means 最小化这个值，但只能找到局部最小值。不同的初始化可能产生不同的结果。

### 选择 K

两种标准方法：

**Elbow method (肘部法则)：** 对 K = 1, 2, 3, ..., n 运行 K-Means。绘制 inertia 与 K 的关系图。寻找"肘部"，即增加更多聚类不再显著降低 inertia 的位置。

**Silhouette score (轮廓系数)：** 对每个点，衡量它与其自身聚类 (a) 与最近的其他聚类 (b) 的相似度。Silhouette coefficient 为 (b - a) / max(a, b)，范围从 -1（错误的聚类）到 +1（聚类良好）。对所有点取平均得到全局分数。

### DBSCAN：基于密度的聚类

K-Means 假设聚类是球形的，并要求你预先选择 K。DBSCAN 不做这两种假设。它将聚类发现为被稀疏区域分隔的密集区域。

两个参数：
- **eps**：邻域半径
- **min_samples**：形成密集区域所需的最少点数

三种点类型：
- **Core point (核心点)**：在 eps 距离内至少有 min_samples 个点
- **Border point (边界点)**：在核心点的 eps 范围内，但本身不是核心点
- **Noise point (噪声点)**：既不是核心点也不是边界点。这些是异常值。

DBSCAN 将彼此在 eps 范围内的核心点连接成同一个聚类。边界点加入附近核心点的聚类。噪声点不属于任何聚类。

优势：发现任意形状的聚类，自动确定聚类数量，识别异常值。弱点：在密度差异很大的聚类上表现不佳。

### 层次聚类 (Hierarchical Clustering)

构建嵌套聚类的树（dendrogram，树状图）。

Agglomerative（自底向上）：
1. 从每个点作为其自己的聚类开始
2. 合并两个最近的聚类
3. 重复直到只剩一个聚类
4. 在所需层级切割 dendrogram 以获得 K 个聚类

聚类之间的"接近度"可以衡量为：
- **Single linkage (单链接)**：两个聚类中任意两点之间的最小距离
- **Complete linkage (全链接)**：两个聚类中任意两点之间的最大距离
- **Average linkage (平均链接)**：所有点对之间的平均距离
- **Ward's method (Ward 法)**：使总 within-cluster variance 增加最小的合并

### Gaussian Mixture Model (高斯混合模型，GMM)

K-Means 给出 hard assignments（硬分配）：每个点恰好属于一个聚类。GMM 给出 soft assignments（软分配）：每个点有属于每个聚类的概率。

GMM 假设数据由 K 个 Gaussian distribution（高斯分布）混合生成，每个分布有自己的 mean（均值）和 covariance（协方差）。Expectation-Maximization algorithm (EM算法，期望最大化算法) 在以下两步之间交替：

- **E-step (期望步)**：计算每个点属于每个高斯分布的概率
- **M-step (最大化步)**：更新每个高斯分布的均值、协方差和混合权重，以最大化数据的似然

GMM 可以建模椭圆形聚类（不只是像 K-Means 那样的球形），并自然地处理重叠聚类。

### 何时使用哪种方法

| 方法 | 最适合 | 避免使用的情况 |
|------|--------|--------------|
| K-Means | 大数据集，球形聚类，已知 K | 不规则形状，存在异常值 |
| DBSCAN | 未知 K，任意形状，异常值检测 | 密度差异大，维度非常高 |
| Hierarchical (层次聚类) | 小数据集，需要 dendrogram，未知 K | 大数据集（O(n^2) 内存） |
| GMM | 重叠聚类，需要软分配 | 非常大的数据集，维度太多 |

### 使用聚类进行异常检测 (Anomaly Detection)

聚类天然支持异常检测：
- **K-Means**：远离任何质心的点是异常
- **DBSCAN**：噪声点按定义就是异常
- **GMM**：在所有高斯分布下概率低的点是异常

## 动手实现

### 步骤 1：从零实现 K-Means

```python
import math
import random


def euclidean_distance(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n_features = len(data[0])

    centroids = random.sample(data, k)

    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]
        assignments = []

        for point in data:
            distances = [euclidean_distance(point, c) for c in centroids]
            nearest = distances.index(min(distances))
            clusters[nearest].append(point)
            assignments.append(nearest)

        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(random.choice(data))
                continue
            centroid = [
                sum(point[j] for point in cluster) / len(cluster)
                for j in range(n_features)
            ]
            new_centroids.append(centroid)

        if all(
            euclidean_distance(old, new) < 1e-6
            for old, new in zip(centroids, new_centroids)
        ):
            print(f"  在第 {iteration + 1} 次迭代时收敛")
            break

        centroids = new_centroids

    return assignments, centroids
```

### 步骤 2：肘部法则和轮廓系数

```python
def compute_inertia(data, assignments, centroids):
    total = 0.0
    for point, cluster_id in zip(data, assignments):
        total += euclidean_distance(point, centroids[cluster_id]) ** 2
    return total


def silhouette_score(data, assignments):
    n = len(data)
    if n < 2:
        return 0.0

    clusters = {}
    for i, c in enumerate(assignments):
        clusters.setdefault(c, []).append(i)

    if len(clusters) < 2:
        return 0.0

    scores = []
    for i in range(n):
        own_cluster = assignments[i]
        own_members = [j for j in clusters[own_cluster] if j != i]

        if len(own_members) == 0:
            scores.append(0.0)
            continue

        a = sum(euclidean_distance(data[i], data[j]) for j in own_members) / len(own_members)

        b = float("inf")
        for cluster_id, members in clusters.items():
            if cluster_id == own_cluster:
                continue
            avg_dist = sum(euclidean_distance(data[i], data[j]) for j in members) / len(members)
            b = min(b, avg_dist)

        if max(a, b) == 0:
            scores.append(0.0)
        else:
            scores.append((b - a) / max(a, b))

    return sum(scores) / len(scores)


def find_best_k(data, max_k=10):
    print("肘部法则：")
    inertias = []
    for k in range(1, max_k + 1):
        assignments, centroids = kmeans(data, k)
        inertia = compute_inertia(data, assignments, centroids)
        inertias.append(inertia)
        print(f"  K={k}: inertia={inertia:.2f}")

    print("\nSilhouette scores:")
    for k in range(2, max_k + 1):
        assignments, centroids = kmeans(data, k)
        score = silhouette_score(data, assignments)
        print(f"  K={k}: silhouette={score:.4f}")

    return inertias
```

### 步骤 3：从零实现 DBSCAN

```python
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    cluster_id = 0

    def region_query(point_idx):
        neighbors = []
        for i in range(n):
            if euclidean_distance(data[point_idx], data[i]) <= eps:
                neighbors.append(i)
        return neighbors

    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(i)

        if len(neighbors) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]

            if not visited[q]:
                visited[q] = True
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

            j += 1

        cluster_id += 1

    return labels
```

### 步骤 4：高斯混合模型（EM 算法）

```python
def gmm(data, k, max_iterations=100, seed=42):
    random.seed(seed)
    n = len(data)
    d = len(data[0])

    indices = random.sample(range(n), k)
    means = [list(data[i]) for i in indices]
    variances = [1.0] * k
    weights = [1.0 / k] * k

    def gaussian_pdf(x, mean, variance):
        d = len(x)
        coeff = 1.0 / ((2 * math.pi * variance) ** (d / 2))
        exponent = -sum((xi - mi) ** 2 for xi, mi in zip(x, mean)) / (2 * variance)
        return coeff * math.exp(max(exponent, -500))

    for iteration in range(max_iterations):
        responsibilities = []
        for i in range(n):
            probs = []
            for j in range(k):
                probs.append(weights[j] * gaussian_pdf(data[i], means[j], variances[j]))
            total = sum(probs)
            if total == 0:
                total = 1e-300
            responsibilities.append([p / total for p in probs])

        old_means = [list(m) for m in means]

        for j in range(k):
            r_sum = sum(responsibilities[i][j] for i in range(n))
            if r_sum < 1e-10:
                continue

            weights[j] = r_sum / n

            for dim in range(d):
                means[j][dim] = sum(
                    responsibilities[i][j] * data[i][dim] for i in range(n)
                ) / r_sum

            variances[j] = sum(
                responsibilities[i][j]
                * sum((data[i][dim] - means[j][dim]) ** 2 for dim in range(d))
                for i in range(n)
            ) / (r_sum * d)
            variances[j] = max(variances[j], 1e-6)

        shift = sum(
            euclidean_distance(old_means[j], means[j]) for j in range(k)
        )
        if shift < 1e-6:
            print(f"  GMM 在第 {iteration + 1} 次迭代时收敛")
            break

    assignments = []
    for i in range(n):
        assignments.append(responsibilities[i].index(max(responsibilities[i])))

    return assignments, means, weights, responsibilities
```

### 步骤 5：生成测试数据并运行所有算法

```python
def make_blobs(centers, n_per_cluster=50, spread=0.5, seed=42):
    random.seed(seed)
    data = []
    true_labels = []
    for label, (cx, cy) in enumerate(centers):
        for _ in range(n_per_cluster):
            x = cx + random.gauss(0, spread)
            y = cy + random.gauss(0, spread)
            data.append([x, y])
            true_labels.append(label)
    return data, true_labels


def make_moons(n_samples=200, noise=0.1, seed=42):
    random.seed(seed)
    data = []
    labels = []
    n_half = n_samples // 2
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = math.cos(angle) + random.gauss(0, noise)
        y = math.sin(angle) + random.gauss(0, noise)
        data.append([x, y])
        labels.append(0)
    for i in range(n_half):
        angle = math.pi * i / n_half
        x = 1 - math.cos(angle) + random.gauss(0, noise)
        y = 1 - math.sin(angle) - 0.5 + random.gauss(0, noise)
        data.append([x, y])
        labels.append(1)
    return data, labels


if __name__ == "__main__":
    centers = [[2, 2], [8, 3], [5, 8]]
    data, true_labels = make_blobs(centers, n_per_cluster=50, spread=0.8)

    print("=== K-Means 在 3 个 blobs 上 ===")
    assignments, centroids = kmeans(data, k=3)
    print(f"  Centroids: {[[round(c, 2) for c in cent] for cent in centroids]}")
    sil = silhouette_score(data, assignments)
    print(f"  Silhouette score: {sil:.4f}")

    print("\n=== 肘部法则 ===")
    find_best_k(data, max_k=6)

    print("\n=== DBSCAN 在 3 个 blobs 上 ===")
    db_labels = dbscan(data, eps=1.5, min_samples=5)
    n_clusters = len(set(db_labels) - {-1})
    n_noise = db_labels.count(-1)
    print(f"  发现 {n_clusters} 个聚类，{n_noise} 个噪声点")

    print("\n=== GMM 在 3 个 blobs 上 ===")
    gmm_assignments, gmm_means, gmm_weights, _ = gmm(data, k=3)
    print(f"  Means: {[[round(m, 2) for m in mean] for mean in gmm_means]}")
    print(f"  Weights: {[round(w, 3) for w in gmm_weights]}")
    gmm_sil = silhouette_score(data, gmm_assignments)
    print(f"  Silhouette score: {gmm_sil:.4f}")

    print("\n=== DBSCAN 在 moons 上（非球形聚类） ===")
    moon_data, moon_labels = make_moons(n_samples=200, noise=0.1)
    moon_db = dbscan(moon_data, eps=0.3, min_samples=5)
    n_moon_clusters = len(set(moon_db) - {-1})
    n_moon_noise = moon_db.count(-1)
    print(f"  发现 {n_moon_clusters} 个聚类，{n_moon_noise} 个噪声点")

    print("\n=== K-Means 在 moons 上（将无法分离） ===")
    moon_km, moon_centroids = kmeans(moon_data, k=2)
    moon_sil = silhouette_score(moon_data, moon_km)
    print(f"  Silhouette score: {moon_sil:.4f}")
    print("  K-Means 对 moons 分割很差，因为它们不是球形的")

    print("\n=== 使用 DBSCAN 进行异常检测 ===")
    anomaly_data = list(data)
    anomaly_data.append([20.0, 20.0])
    anomaly_data.append([-5.0, -5.0])
    anomaly_data.append([15.0, 0.0])
    anomaly_labels = dbscan(anomaly_data, eps=1.5, min_samples=5)
    anomalies = [
        anomaly_data[i]
        for i in range(len(anomaly_labels))
        if anomaly_labels[i] == -1
    ]
    print(f"  检测到 {len(anomalies)} 个异常")
    for a in anomalies[-3:]:
        print(f"    点 {[round(v, 2) for v in a]}")
```

## 使用它

使用 scikit-learn，同样的算法只需一行代码：

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sklearn_silhouette

km = KMeans(n_clusters=3, random_state=42).fit(data)
db = DBSCAN(eps=1.5, min_samples=5).fit(data)
agg = AgglomerativeClustering(n_clusters=3).fit(data)
gmm_model = GaussianMixture(n_components=3, random_state=42).fit(data)
```

从零实现的版本向你展示了这些库到底在计算什么。K-Means 在分配和重新计算之间迭代。DBSCAN 从密集的种子点生长聚类。GMM 在期望和最大化之间交替。库版本增加了数值稳定性、更智能的初始化（K-Means++）和 GPU 加速，但核心逻辑是相同的。

## 交付

本课从零实现了 K-Means、DBSCAN 和 GMM 的工作版本。聚类代码可以作为更高级无监督方法的基础重复使用。

## 练习

1. 实现 K-Means++ 初始化：不是随机选择质心，而是随机选择第一个，后续每个质心的选择概率与其到最近现有质心的平方距离成正比。比较与随机初始化的收敛速度。
2. 将层次 agglomerative clustering（凝聚式层次聚类）添加到代码中。实现 Ward's linkage 并生成 dendrogram（作为嵌套合并列表）。在不同层级切割并与 K-Means 结果比较。
3. 构建一个简单的异常检测 (anomaly detection) 流程：在同一数据上运行 DBSCAN 和 GMM，标记两种方法都认为是异常值的点（DBSCAN 中的噪声，GMM 中的低概率）。衡量重叠并讨论方法何时不一致。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------|---------|
| Clustering (聚类) | "将相似的事物分组" | 将数据划分为子集，其中组内相似度超过组间相似度，由特定的距离度量衡量 |
| Centroid (质心) | "聚类的中心" | 分配给聚类的所有点的均值；K-Means 用它作为聚类代表 |
| Inertia (惯性) | "聚类有多紧密" | 每个点到其分配质心的平方距离之和；越低越紧密 |
| Silhouette score (轮廓系数) | "聚类分离得有多好" | 对每个点，(b - a) / max(a, b)，其中 a 是平均簇内距离，b 是平均最近簇距离 |
| Core point (核心点) | "密集区域中的点" | 在 DBSCAN 中，在 eps 距离内至少有 min_samples 个邻居的点 |
| EM algorithm (EM算法) | "软 K-Means" | Expectation-Maximization（期望最大化）：迭代计算成员概率（E步）并更新分布参数（M步） |
| Dendrogram (树状图) | "聚类的树" | 显示层次聚类中聚类合并顺序和距离的树形图 |
| Anomaly (异常) | "异常值" | 不符合预期模式的数据点，被 DBSCAN 识别为噪声或被 GMM 识别为低概率点 |

## 延伸阅读

- [Stanford CS229 - Unsupervised Learning](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) - Andrew Ng 关于聚类和 EM 的讲义
- [scikit-learn Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html) - 所有聚类算法的实用比较，附带可视化示例
- [DBSCAN original paper (Ester et al., 1996)](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) - 引入基于密度聚类的论文
