# 范数与距离

> 你的距离函数定义了“相似”的含义。选错了，下游一切都会崩溃。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 分钟

## Learning Objectives

- 从零实现 L1、L2、余弦、Mahalanobis distance (马氏距离)、Jaccard similarity (杰卡德相似度) 和 edit distance (编辑距离) 函数
- 为给定的 ML 任务选择合适的距离度量，并解释为什么替代方案会失败
- 将 L1 和 L2 范数与 LASSO 和 Ridge regularization (正则化) 及其几何约束区域联系起来
- 演示同一数据集在不同度量下产生不同的最近邻

## The Problem

你有两个向量。它们可能是 word embeddings (词嵌入)。可能是用户画像。可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择哪个距离函数。两个数据点可能在一个度量下是最近邻，在另一个度量下却相距甚远。你的 KNN classifier (K近邻分类器)、推荐引擎、向量数据库、聚类算法、loss function (损失函数) —— 它们都依赖这个选择。选错了，你的模型就会优化错误的目标。

没有放之四海而皆准的最佳距离。L2 适用于空间数据。Cosine similarity (余弦相似度) 主导 NLP。Jaccard similarity 处理集合。Edit distance 处理字符串。Mahalanobis distance 考虑相关性。Wasserstein distance 移动概率质量。每一种都编码了关于“相似”含义的不同假设。

本课从零构建每一个主要距离函数，告诉你何时该用哪一种，并演示同一数据在不同度量下产生完全不同的最近邻。

## The Concept

### 范数：度量向量的大小

范数度量向量的“大小”。两个向量之间的每个距离函数都可以写成它们差的范数：d(a, b) = ||a - b||。所以理解范数就是理解距离。

### L1 范数（Manhattan distance / 曼哈顿距离）

L1 范数对所有分量的绝对值求和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为 Manhattan distance，因为它度量你在城市网格上行走的距离，只能沿坐标轴移动，不能走对角线。

```
点 A = (1, 1)
点 B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

在网格上，你向东走 3 个街区，向北走 4 个街区。
```

何时使用 L1：
- 高维稀疏数据（文本特征、one-hot 编码）
- 当你希望对异常值 robust（鲁棒）时（单个巨大差异不会主导结果）
- 特征选择问题（L1 regularization (L1正则化 / Lasso) 促进稀疏性）

与 L1 regularization (Lasso) 的联系：将 ||w||_1 加入 loss function 会惩罚权重绝对值之和。这会将小权重精确推至零，实现自动特征选择。L1 惩罚在权重空间中形成菱形约束区域，菱形的角落在坐标轴上，该处部分权重为零。

与 loss function 的联系：Mean Absolute Error (MAE) (平均绝对误差) 是预测值与目标值之间 L1 distance 的平均。它对所有误差线性惩罚，与 MSE 相比对异常值更 robust。

### L2 范数（Euclidean distance / 欧氏距离）

L2 范数是直线距离。各分量平方和开根号。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这是你在几何课上学过的距离。n 维空间中的毕达哥拉斯定理。

```
点 A = (1, 1)
点 B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

直线穿过网格对角线。
```

何时使用 L2：
- 低至中等维度的连续数据
- 当特征尺度可比时
- 物理距离（空间数据、传感器读数）
- 像素级别的图像相似度

与 L2 regularization (Ridge) 的联系：将 ||w||_2^2 加入 loss function 会惩罚大权重。与 L1 不同，它不会将权重推至零。它按比例将所有权重收缩向零。L2 惩罚形成圆形约束区域，因此在坐标轴上没有角。权重变小但很少精确为零。

与 loss function 的联系：Mean Squared Error (MSE) (均方误差) 是 L2 distance 平方的平均。平方项对较大误差的惩罚远大于小误差。

```
MAE (L1 loss):  |y - y_hat|         线性惩罚。对异常值 robust。
MSE (L2 loss):  (y - y_hat)^2       二次惩罚。对异常值敏感。
```

### Lp 范数：一般族

L1 和 L2 是 Lp 范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 p 值产生不同形状的“单位球”（距离原点为 1 的所有点的集合）：

```
p=1:    菱形      （角落在坐标轴上）
p=2:    圆/球      （通常的圆球）
p=3:    超椭圆     （圆角正方形）
p=inf:  正方形/超立方体  （边沿坐标轴平齐）
```

### L-无穷范数（Chebyshev distance / 切比雪夫距离）

当 p 趋近无穷大时，Lp 范数收敛于最大绝对分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们差异最大的单一维度决定。其他所有维度都被忽略。

```
点 A = (1, 1)
点 B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用 L-无穷：
- 当任何单一维度的最坏情况偏差至关重要时
- 棋盘游戏（国际象棋中的王以 L-无穷移动：任何方向一步代价为 1）
- 制造公差（每个维度都必须在规格内）

### Cosine Similarity（余弦相似度）与 Cosine Distance（余弦距离）

Cosine similarity 度量两个向量之间的夹角，忽略它们的模长。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

范围从 -1（方向相反）到 +1（方向相同）。垂直向量的 cosine similarity 为 0。

Cosine distance 将其转换为距离：cosine_distance = 1 - cosine_similarity。范围从 0（方向相同）到 2（方向相反）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么 cosine similarity 主导 NLP 和 embeddings：在文本中，文档长度不应影响相似度。一篇关于猫的长文档和另一篇关于猫的短文档仍然应该“相似”。Cosine similarity 忽略模长（长度），只关心方向。两个具有相同词分布但长度不同的文档指向同一方向，得到 cosine similarity 1.0。

何时使用 cosine similarity：
- 文本相似度（TF-IDF 向量、word embeddings、sentence embeddings）
- 任何模长是噪声而方向是信号的域
- 推荐系统（用户偏好向量）
- Embedding 搜索（向量数据库几乎总是使用 cosine 或 dot product）

### Dot Product Similarity（点积相似度） vs Cosine Similarity

两个向量的 dot product（点积）为：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity 是 dot product 被双方模长归一化后的结果。当两个向量都已单位归一化（模长 = 1）时，dot product 和 cosine similarity 完全相同。

```
如果 ||a|| = 1 且 ||b|| = 1：
    a . b = cos(a 与 b 之间的夹角)
```

当它们不同时：dot product 包含模长信息。模长更大的向量获得更高的 dot product 分数。这在某些检索系统中很重要，你希望“热门”项目排名更高。模长充当隐式的质量或重要性信号。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

两者在方向上一致，但 dot product 还反映了模长。
```

在实践中：
- 当你想要纯方向相似度时使用 cosine similarity
- 当模长携带有意义信息时使用 dot product
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许你在两者之间选择
- 如果你的 embeddings 已 L2 归一化，选择无关紧要

### Mahalanobis Distance（马氏距离）

Euclidean distance 平等对待所有维度。但如果你的特征相关或尺度不同，L2 会产生误导结果。

Mahalanobis distance 考虑数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中 S 是数据的协方差矩阵。

直观理解：Mahalanobis distance 首先对数据去相关并归一化（白化），然后在该变换后的空间中计算 L2 distance。如果 S 是单位矩阵（特征不相关且单位方差），Mahalanobis distance 退化为 Euclidean distance。

```
示例：身高和体重相关。
身高 6'2"、体重 180 磅的人并不异常。
身高 5'0"、体重 180 磅的人很异常。

Euclidean distance 可能说它们离均值一样远。
Mahalanobis distance 正确识别出第二个是异常值，
因为它考虑了身高-体重的相关性。
```

何时使用 Mahalanobis distance：
- 异常值检测（离均值 Mahalanobis distance 很大的点是异常值）
- 特征具有不同尺度和相关性时的分类
- 当有足够数据估计可靠协方差矩阵时
- 制造业质量控制（多变量过程监控）

### Jaccard Similarity（杰卡德相似度，用于集合）

Jaccard similarity 度量两个集合之间的重叠。

```
J(A, B) = |A 交 B| / |A 并 B|
```

范围从 0（无重叠）到 1（相同集合）。Jaccard distance = 1 - Jaccard similarity。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

交集 = {cat, fish}         大小 = 2
并集 = {cat, dog, fish, bird, snake}  大小 = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用 Jaccard：
- 比较标签集、类别或特征集
- 基于词出现（而非频率）的文档相似度
- 近似重复检测（MinHash 对 Jaccard 的近似）
- 比较二元特征向量（出现/不出现数据）
- 评估分割模型（Intersection over Union = Jaccard）

### Edit Distance（编辑距离 / Levenshtein Distance）

Edit distance 计算将一个字符串转换为另一个字符串所需的最少单字符操作数。操作包括：插入、删除、替换。

```
"kitten" -> "sitting"

kitten -> sitten  (替换 k -> s)
sitten -> sittin  (替换 e -> i)
sittin -> sitting (插入 g)

Edit distance = 3
```

使用动态规划计算。填充一个矩阵，其中条目 (i, j) 是字符串 A 前 i 个字符与字符串 B 前 j 个字符之间的 edit distance。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用 edit distance：
- 拼写检查和纠正
- DNA 序列比对（带加权操作）
- 模糊字符串匹配
-  messy 文本数据去重

### KL Divergence（KL 散度）

KL divergence 度量一个概率分布与另一个的差异。在 Lesson 09 中已涵盖，但它属于本次讨论，因为人们将其当作“距离”使用，尽管它并不是。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键性质：KL divergence 不是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它不满足距离的基本要求。也不满足三角不等式。它是一种 divergence（散度），而非 distance（距离）。

Forward KL（D_KL(P || Q)）是“mean-seeking（均值追寻）”：Q 试图覆盖 P 的所有模态。
Reverse KL（D_KL(Q || P)）是“mode-seeking（模态追寻）”：Q 聚焦在 P 的单个模态上。

当你看到 KL divergence：
- VAE（ELBO 中的 KL 项将隐分布推向先验）
- Knowledge distillation（学生试图匹配教师的分布）
- RLHF（KL 惩罚使 fine-tuned (微调) 模型保持接近基础模型）
- Policy gradient 方法（约束策略更新）

### Wasserstein Distance（Wasserstein 距离 / Earth Mover's Distance / 推土机距离）

Wasserstein distance 度量将一个概率分布转换为另一个所需的最小“工作量”。可以想象为：一个分布是一堆土，另一个是坑，你需要移动多少土、移动多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它简化为累积分布函数绝对差值的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Wasserstein 重要的原因：
- 它是一种真正的度量（对称，满足三角不等式）
- 即使分布不重叠，它也能提供梯度（KL divergence 趋于无穷）
- 这一性质使其成为 Wasserstein GANs (WGANs) 的核心，解决了原始 GAN 的训练不稳定性

```
不重叠的分布：

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: 无穷大（log of zero）
Wasserstein: 4（将所有质量移动 4 个 bin）

Wasserstein 给出有意义的梯度。KL 不能。
```

何时使用 Wasserstein：
- GAN 训练（WGAN, WGAN-GP）
- 比较可能不重叠的分布
- 最优输运问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同的距离

| 任务 | 最佳距离 | 原因 |
|------|---------|------|
| 文本相似度 | Cosine | 模长是噪声，方向是意义 |
| 图像像素比较 | L2 | 空间关系重要，特征尺度可比 |
| 稀疏高维特征 | L1 | Robust，不会放大罕见的大差异 |
| 集合重叠（标签、类别） | Jaccard | 数据本质上是集合值，而非向量 |
| 字符串匹配 | Edit distance | 操作映射到人类编辑直觉 |
| 异常值检测 | Mahalanobis | 考虑特征相关性和尺度 |
| 比较分布 | KL divergence | 度量使用 Q 代替 P 所损失的信息 |
| GAN 训练 | Wasserstein | 即使分布不重叠也能提供梯度 |
| Embeddings（向量数据库） | Cosine 或 dot product | Embeddings 被训练成将意义编码在方向中 |
| 推荐系统 | Dot product | 模长可以编码流行度或置信度 |
| DNA 序列 | Weighted edit distance | 替换成本因核苷酸对而异 |
| 制造 QC | L-infinity | 任何维度的最坏情况偏差都重要 |

### 与 Loss Functions 的联系

Loss function 是应用于预测值与目标值之间的距离函数。

```
Loss function       使用的距离       行为
MSE                 L2 平方          对较大误差惩罚更重
MAE                 L1               对所有误差平等惩罚
Huber loss          大误差用 L1，    两者兼顾：对异常值 robust，
                    小误差用 L2      零点附近梯度平滑
Cross-entropy       KL divergence    度量分布不匹配
Hinge loss          max(0, margin - d) 只惩罚低于边距的
Triplet loss        L2（通常）       拉近正样本，推远负样本
Contrastive loss    L2               相似对拉近，不相似对推过边距
```

### 与 Regularization 的联系

Regularization（正则化）向 loss function 添加权重的范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> 稀疏权重。某些权重精确变为零。
  -> 自动特征选择。
  -> 解在角落（零处不可微）。

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> 小权重。所有权重收缩向零。
  -> 无特征选择（没有精确变为零的）。
  -> 处处平滑。

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> 结合 L1 的稀疏性和 L2 的稳定性。
  -> 相关特征组被一起保留或丢弃。
```

为什么 L1 产生稀疏性而 L2 不：想象二维权重空间中的约束区域。L1 是菱形，L2 是圆形。Loss function 的等高线（椭圆）最可能先碰到菱形的角，该处一个权重为零。它们碰到圆的光滑点，该处两个权重都非零。

### 最近邻搜索

每个距离函数都隐含一个最近邻搜索问题：给定一个查询点，在数据集中找到最近的点。

精确最近邻搜索在 n 个 d 维点的数据集中每次查询复杂度为 O(n * d)。对于大数据集，这太慢了。

Approximate Nearest Neighbor (ANN)（近似最近邻）算法以少量精度换取巨大的速度提升：

```
Algorithm         方法                      被用于
KD-trees          轴对齐空间划分             scikit-learn（低维）
Ball trees        嵌套超球体                 scikit-learn（中维）
LSH               随机哈希投影               近似重复检测
HNSW              分层可导航                 FAISS, Qdrant, Weaviate
                  小世界图
IVF               倒排文件索引               FAISS（十亿级）
                  基于聚类的搜索
Product quant.    压缩向量，在               FAISS（内存受限）
                  压缩空间中搜索
```

HNSW (Hierarchical Navigable Small World) 是现代向量数据库中的主导算法。它构建多层图，每个节点连接到其近似最近邻。搜索从顶层（稀疏、长跳）开始，下降到底层（密集、短跳）。

## Build It

### Step 1: 所有范数和距离函数

参见 `code/distances.py` 获取完整实现。每个函数都使用基础 Python 数学从零构建。

### Step 2: 相同数据，不同距离，不同邻居

`distances.py` 中的演示创建一个数据集，选取一个查询点，并展示最近邻如何随距离度量而变化。在 L1 下“最近”的点在 L2 或 cosine 下可能不是最近。

### Step 3: Embedding 相似度搜索

代码包含一个 mock embedding 相似度搜索，使用 cosine similarity 与 L2 distance 找到与查询最“相似”的“文档”，展示排名可能不同。

## Use It

最常见的实际用途：在向量数据库中查找相似项。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 然后搜索向量数据库时，这就是底层发生的事。Embedding 模型将文本映射为向量。向量数据库计算你的查询向量与每个存储向量之间的 cosine similarity（或 dot product），使用 ANN 算法避免检查所有向量。

## Exercises

1. 计算 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-infinity distance。验证对于任意点对，L-inf <= L2 <= L1 始终成立。证明为什么这种排序是保证的。

2. 创建两个 cosine similarity 高（> 0.9）但 L2 distance 大（> 10）的向量。从几何上解释发生了什么。然后创建两个 cosine similarity 低（< 0.3）但 L2 distance 小（< 0.5）的向量。

3. 实现一个函数，接受数据集和查询点，返回在 L1、L2、cosine 和 Mahalanobis distance 下的最近邻。找到一个所有四种度量对最近邻意见不一致的数据集。

4. 使用 CDF 方法手工计算 [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein distance。然后计算 [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein distance。哪个更大，为什么？

5. 实现 MinHash 以近似 Jaccard similarity。生成 100 个随机集合，计算所有配对的精确 Jaccard，并与使用 50、100 和 200 个哈希函数的 MinHash 近似比较。绘制近似误差。

## Key Terms

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Norm | “向量的大小” | 将向量映射到非负标量的函数，满足三角不等式、绝对齐次性，且仅对零向量为零 |
| L1 norm | “Manhattan distance” | 绝对分量值之和。在优化中产生稀疏性。对异常值 robust |
| L2 norm | “Euclidean distance” | 分量平方和开根号。欧氏空间中的直线距离 |
| Lp norm | “广义范数” | 绝对分量 p 次幂之和的 p 次根。L1 和 L2 是特例 |
| L-infinity norm | “Max norm” 或 “Chebyshev distance” | 最大绝对分量值。Lp 当 p 趋近无穷时的极限 |
| Cosine similarity | “向量之间的夹角” | Dot product 被双方模长归一化。范围 [-1, +1]。忽略向量长度 |
| Cosine distance | “1 减 cosine similarity” | 将 cosine similarity 转换为距离。范围 [0, 2] |
| Dot product | “未归一化的 cosine” | 分量乘积之和。等于 cosine similarity 乘以双方模长 |
| Mahalanobis distance | “考虑相关性的距离” | 在已被白化（用数据协方差矩阵去相关和归一化）的空间中的 L2 distance |
| Jaccard similarity | “集合重叠” | 交集大小除以并集大小。用于集合，而非向量 |
| Edit distance | “Levenshtein distance” | 将一个字符串转换为另一个所需的最少插入、删除和替换次数 |
| KL divergence | “分布之间的距离” | 不是真正的距离（不对称）。度量使用 Q 编码 P 所需的额外比特 |
| Wasserstein distance | “Earth mover's distance” | 将质量从一个分布运输到另一个所需的最小工作量。真正的度量 |
| Approximate nearest neighbor | “ANN 搜索” | 算法（HNSW、LSH、IVF）以比精确搜索快得多的速度找到近似最近的点 |
| HNSW | “向量数据库算法” | Hierarchical Navigable Small World graph。多层图，用于快速近似最近邻搜索 |
| L1 regularization | “Lasso” | 将权重的 L1 范数加入 loss。将权重推至零（稀疏性） |
| L2 regularization | “Ridge” 或 “weight decay” | 将权重的平方 L2 范数加入 loss。将权重收缩向零，无稀疏性 |
| Elastic Net | “L1 + L2” | 结合 L1 和 L2 regularization。比单独使用更好地处理相关特征组 |

## Further Reading

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 的十亿级 ANN 搜索库
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 将 Earth Mover's distance 引入 GAN 的论文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 基础 ANN 算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，cosine similarity 成为 embeddings 默认选择的论文
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量和邻居算法的实用指南
