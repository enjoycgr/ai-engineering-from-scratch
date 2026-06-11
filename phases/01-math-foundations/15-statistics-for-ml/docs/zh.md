# 面向机器学习的统计学

> 统计学是你判断模型到底真有效还是纯粹走运的方法。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 分钟

## Learning Objectives

- 从零计算描述性统计量、Pearson correlation (皮尔逊相关系数)、Spearman correlation (斯皮尔曼相关系数) 和 covariance matrix (协方差矩阵)
- 执行假设检验（t-test、t检验、chi-squared test、卡方检验），并正确解读 p-value (p值) 和 confidence interval (置信区间)
- 使用 bootstrap resampling (自助重采样) 为任意指标构建无需分布假设的 confidence interval
- 使用 effect size (效应量) 区分统计显著性与实际显著性

## The Problem

你训练了两个模型。模型 A 在测试集上得分 0.87。模型 B 得分 0.89。你部署了模型 B。三周后，线上指标比之前更差。发生了什么？

模型 B 实际上并没有超越模型 A。0.02 的差异只是噪声。你的测试集太小，或者方差太大，或者两者兼有。你把随机性打扮成了改进 shipped 了出去。

这 constantly 发生。Kaggle 排行榜的剧烈波动。无法复现的论文。基于几百个样本就宣布胜者的 A/B 测试。根本原因总是同一个：有人跳过了统计学。

统计学给你区分信号与噪声的工具。它告诉你差异何时是真实的，你应该有多自信，以及在你能信任一个结果之前需要多少数据。每个 ML pipeline、每个模型比较、每个实验都需要统计学。没有它，你只是在猜测。

## The Concept

### 描述性统计：总结你的数据

在你建模任何东西之前，你需要知道数据长什么样。描述性统计将数据集压缩成几个数字，捕捉其形状。

**集中趋势度量**回答“中间在哪里？”

```
Mean (均值):   所有值之和 / 计数
        mu = (1/n) * sum(x_i)

Median (中位数): 排序后的中间值
        对异常值 robust。如果你有 [1, 2, 3, 4, 1000]，mean 是 202
        但 median 是 3。

Mode (众数):   出现最频繁的值
        对分类数据有用。对连续数据，很少能提供信息。
```

Mean 是平衡点。Median 是中分点。当它们分离时，你的分布是偏斜的。收入分布的 mean >> median（亿万富翁导致的右偏）。训练过程中的 loss 分布往往 mean << median（简单样本导致的左偏）。

**离散程度度量**回答“数据有多分散？”

```
Variance (方差):   与 mean 偏差平方的平均
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation (标准差):  variance 的平方根
                     sigma = sqrt(sigma^2)
                     与数据同单位，因此更可解释。

Range (极差):      max - min
            对异常值敏感。几乎从不单独有用。

IQR (四分位距):        Q3 - Q1
            中间 50% 数据的范围。
            对异常值 robust。用于箱线图和异常值检测。
```

**Percentile (百分位数)** 将排序后的数据分成 100 等份。第 25 百分位数（Q1）意味着 25% 的值低于该点。第 50 百分位数是 median。第 75 百分位数是 Q3。

```
对于延迟监控：
  P50 = median latency        （典型用户体验）
  P95 = 95th percentile       （糟糕但非最坏情况）
  P99 = 99th percentile       （尾部延迟，常常是 median 的 10 倍）
```

在 ML 中，你关心 percentile 是为了 inference latency、预测置信度分布，以及理解误差分布。一个 mean error 低但 P99 error 很差的模型对安全关键应用可能毫无用处。

**样本统计 vs 总体统计。** 从样本计算 variance 时，除以 (n-1) 而不是 n。这是 Bessel's correction (贝塞尔校正)。它补偿了你的样本均值并非真实总体均值的事实。分母为 n 时，你会系统性地低估真实 variance。分母为 (n-1) 时，估计是无偏的。

```
总体 variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
样本 variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实践中：如果 n 很大（数千样本），差异可忽略。如果 n 很小（数十个样本），它很重要。

### 相关性：变量如何共同变动

相关性度量两个变量之间线性关系的强度和方向。

**Pearson correlation coefficient (皮尔逊相关系数)** 度量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  完全正线性关系
r = -1:  完全负线性关系
r =  0:  无线性关系（但可能存在非线性关系！）

范围: [-1, 1]
```

Pearson 假设关系是线性的，且两个变量大致正态分布。它对异常值敏感。一个极端点可以把 r 从 0.1 拖到 0.9。

**Spearman rank correlation (斯皮尔曼秩相关系数)** 度量单调关联：

```
1. 将每个值替换为其秩（1, 2, 3, ...）
2. 在秩上计算 Pearson correlation

Spearman 捕捉任何单调关系，不仅仅是线性的。
如果 y = x^3，Pearson 给出 r < 1 但 Spearman 给出 rho = 1。
```

**何时使用每种：**

```
Pearson:    两个变量都是连续且大致正态的。
            你专门关心线性关系。
            没有极端异常值。

Spearman:   有序数据（排名、评分）。
            数据不是正态分布的。
            你怀疑存在单调但非线性的关系。
            存在异常值。
```

**黄金法则：** correlation does not imply causation（相关不等于因果）。冰淇淋销量和溺亡死亡人数相关，因为两者都在夏天增加。你的模型准确率和参数数量相关，但增加参数并不会自动提高准确率（参见：overfitting / 过拟合）。

### Covariance Matrix（协方差矩阵）

两个变量之间的 covariance（协方差）度量它们如何共同变动：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X 和 Y 倾向于一起增加
Cov(X, Y) < 0:  当 X 增加时，Y 倾向于减少
Cov(X, Y) = 0:  无线性共变
```

对于 d 个特征，covariance matrix C 是一个 d x d 矩阵，其中 C[i][j] = Cov(feature_i, feature_j)。对角线元素 C[i][i] 是每个特征的 variance。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

性质：
  - 对称: C[i][j] = C[j][i]
  - 半正定: 所有特征值 >= 0
  - 对角线 = variances
  - 非对角线 = covariances
```

**与 PCA 的联系。** PCA 对 covariance matrix 做特征分解。特征向量是主成分（最大 variance 的方向）。特征值告诉你每个成分捕获了多少 variance。这正是 Lesson 10 涵盖的内容，但现在你明白了为什么 covariance matrix 是正确的分解对象：它编码了数据中所有成对的线性关系。

**与 correlation 的联系。** Correlation matrix 是标准化变量（每个除以 standard deviation）的 covariance matrix。Correlation 将 covariance 归一化，使所有值落在 [-1, 1]。

### 假设检验

假设检验是在不确定性下做决策的框架。你从一个主张出发，收集数据，然后判断数据是否与该主张一致。

** setup：**

```
Null hypothesis (零假设 / H0):        默认假设，通常是“无效应”
Alternative hypothesis (备择假设 / H1): 你试图证明的

示例：
  H0: 模型 A 和模型 B 的准确率相同
  H1: 模型 B 的准确率高于模型 A
```

**p-value (p值)** 是在 H0 为真的前提下，观察到与你所得数据同样极端的数据的概率。它不是 H0 为真的概率。这是统计学中最常见的误解。

```
p-value = P(数据如此极端 | H0 为真)

如果 p-value < alpha (通常 0.05)：
    拒绝 H0。结果“统计显著”。
如果 p-value >= alpha：
    无法拒绝 H0。你没有足够证据。
    这并不意味着 H0 为真。
```

**Confidence interval (置信区间)** 给出参数的一个合理值范围：

```
Mean 的 95% confidence interval：
    x_bar +/- z * (s / sqrt(n))

其中 z = 1.96 对应 95% 置信度

解读：如果你重复这个实验很多次，95% 的
计算出的区间会包含真实 mean。这并不意味着
真实 mean 有 95% 的概率落在这个特定区间里。
```

Confidence interval 的宽度告诉你关于精度的信息。宽区间意味着高不确定性。窄区间意味着你的估计很精确（但如果数据有偏，不一定准确）。

### t-test（t检验）

t-test 比较均值。有几种变体。

**One-sample t-test（单样本 t检验）：** 总体 mean 是否与某个假设值不同？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test（双样本 t检验，独立）：** 两组 mean 是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

这是 Welch's t-test，不假设等方差。
除非你有特定理由假设等方差，否则总是使用 Welch's。
```

**Paired t-test（配对 t检验）：** 当测量成对出现时（同一模型在相同数据划分上评估）：

```
计算每对的 d_i = x_i - y_i
然后对 d_i 值运行单样本 t-test，mu_0 = 0
```

在 ML 中，配对 t-test 很常见：你在相同的 10 折交叉验证上运行两个模型并成对比较它们的分数。

### Chi-squared Test（卡方检验）

Chi-squared test 检查观察到的频率是否与期望频率匹配。对分类数据很有用。

```
chi^2 = sum((observed - expected)^2 / expected)

示例：语言模型的输出分布是否与
训练分布跨类别匹配？

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

自由度为 1 时，chi^2 = 8 给出 p < 0.005。
差异是显著的。
```

### ML 模型的 A/B Testing

ML 中的 A/B testing 与网页 A/B testing 不同。模型比较有特定的挑战：

```
1. 相同测试集:    两个模型必须在相同数据上评估。
                     不同的测试集使比较毫无意义。

2. 多个指标:    仅 Accuracy 不够。你需要 precision、
                     recall、F1、latency 和公平性指标。

3. 方差:         使用交叉验证或 bootstrap 估计
                     每个指标的 variance，而不仅仅是点估计。

4. 数据泄露:     如果测试集在模型选择中被使用过，
                     你的比较就是有偏的。保留最终测试集。
```

**流程：**

```
1. 定义你的指标和显著性水平（alpha = 0.05）
2. 在相同的 k-fold 交叉验证划分上运行两个模型
3. 收集配对分数：[(a1, b1), (a2, b2), ..., (ak, bk)]
4. 计算差异：d_i = b_i - a_i
5. 对差异运行配对 t-test
6. 检查：mean difference 是否显著不同于 0？
7. 计算 mean difference 的 confidence interval
8. 计算 effect size（Cohen's d）以判断实际显著性
```

### 统计显著性 vs 实际显著性

一个结果可以是统计显著的，但实际毫无意义。有了足够的数据，即使是微不足道的差异也会变得统计显著。

```
示例：
  模型 A 准确率: 0.9234
  模型 B 准确率: 0.9237
  n = 1,000,000 测试样本
  p-value = 0.001

统计显著？是的。
实际显著？0.03% 的提升不值得
部署新模型的工程成本。
```

**Effect size (效应量)** 量化差异有多大，独立于样本量：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  小效应
d = 0.5:  中效应
d = 0.8:  大效应
```

始终同时报告 p-value 和 effect size。p-value 告诉你差异是否真实。Effect size 告诉你差异是否重要。

### Multiple Comparison Problem（多重比较问题）

当你检验很多假设时，有些会偶然“显著”。如果你在 alpha = 0.05 下检验 20 件事，即使什么都不真实，你也期望有 1 个 false positive (假阳性)。

```
P(至少一个 false positive) = 1 - (1 - alpha)^m

m = 20 次检验, alpha = 0.05：
P(false positive) = 1 - 0.95^20 = 0.64

你有 64% 的概率至少遇到一个 false positive。
```

**Bonferroni correction (邦费罗尼校正)：** 将 alpha 除以检验次数。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

仅当 p-value < 0.0025 时才拒绝 H0。
保守但简单。在检验独立时有效。
```

在 ML 中，这在你跨多个指标比较模型、测试很多 hyperparameter (超参数) 配置，或在多个数据集上评估时很重要。

### Bootstrap Methods（自助法）

Bootstrapping 通过有放回地重采样你的数据来估计统计量的抽样分布。不需要对底层分布做任何假设。

**算法：**

```
1. 你有 n 个数据点
2. 有放回地抽取 n 个样本（某些点出现多次，
   某些不出现）
3. 在这个 bootstrap sample 上计算你的统计量
4. 重复 B 次（通常 B = 1000 到 10000）
5. Bootstrap 统计量的分布近似于
   抽样分布
```

**Bootstrap confidence interval (percentile method)：**

```
将 B 个 bootstrap 统计量排序
95% CI = [第 2.5 百分位数, 第 97.5 百分位数]
```

**为什么 bootstrap 对 ML 重要：**

```
- 测试集准确率是一个点估计。Bootstrap 给你
  confidence intervals。
- 你不能假设指标分布是正态的（尤其是
  AUC、F1、precision at k）。
- Bootstrap 对任何统计量都有效：median、两个 mean 的比值、
  两个模型 AUC 的差异。
- 不需要闭式公式。
```

**用于模型比较的 Bootstrap：**

```
1. 你有模型 A 和模型 B 在同一测试集上的预测
2. 对每个 bootstrap 迭代：
   a. 有放回地重采样测试索引
   b. 在重采样集上计算 metric_A 和 metric_B
   c. 存储 diff = metric_B - metric_A
3. 差异的 95% CI：
   [diffs 的第 2.5 百分位数, diffs 的第 97.5 百分位数]
4. 如果 CI 不包含 0，差异是显著的
```

这比配对 t-test 更 robust，因为它不做分布假设。

### Parametric vs Non-parametric Tests（参数检验 vs 非参数检验）

**Parametric tests（参数检验）** 假设特定分布（通常是正态）：

```
t-test:         假设正态分布数据（或大 n 由 CLT）
ANOVA:          假设正态性和等方差
Pearson r:      假设双变量正态性
```

**Non-parametric tests（非参数检验）** 不做分布假设：

```
Mann-Whitney U:     比较两组（替代独立 t-test）
Wilcoxon signed-rank: 比较配对数据（替代配对 t-test）
Spearman rho:       在秩上计算 correlation（替代 Pearson）
Kruskal-Wallis:     比较多组（替代 ANOVA）
```

**何时使用非参数：**

```
- 小样本量（n < 30）且数据明显非正态
- 有序数据（评分、排名）
- 无法移除的严重异常值
- 偏斜分布
```

**何时使用参数：**

```
- 大样本量（CLT 使检验统计量近似正态）
- 数据大致对称且没有极端异常值
- 更大的统计功效（更好地检测真实差异）
```

在 ML 实验中，你通常有较小的 n（5 或 10 折交叉验证），所以非参数检验如 Wilcoxon signed-rank 通常比 t-test 更合适。

### Central Limit Theorem（中心极限定理）：实际含义

CLT 说，样本 mean 的分布随着 n 增长趋近正态分布，无论底层总体分布是什么。

```
如果 X_1, X_2, ..., X_n 是 iid，mean 为 mu，variance 为 sigma^2：

    X_bar ~ Normal(mu, sigma^2 / n)    当 n -> 无穷

在大多数情况下 n >= 30 就有效。
对高度偏斜分布，你可能需要 n >= 100。
```

**为什么这对 ML 重要：**

```
1. 为聚合指标上的 confidence intervals 和 t-test 提供依据
2. 解释为什么对交叉验证折取平均给出稳定
   的估计，即使个别折差异很大
3. Mini-batch gradient descent 有效，因为 batch 上的平均 gradient
   近似真实 gradient（CLT 在作用）
4. Ensemble 方法：对多个模型的预测取平均给出
   比任何单一模型更稳定的输出
```

**CLT 不能做什么：**

```
- 不能让你的数据变成正态。它使样本 MEAN 正态。
- 对 infinite variance 的厚尾分布无效
  （Cauchy 分布）。
- 不适用于依赖数据（未经修正的时间序列）。
```

### ML 论文中常见的统计错误

1. **在训练集上测试。** 保证 overfitting (过拟合)。始终保留模型在训练期间从未见过的数据。

2. **没有 confidence intervals。** 报告单个准确率数字而不带不确定性使结果不可复现且无法验证。

3. **忽略多重比较。** 测试 50 种配置并报告最好的一个而不做校正会夸大 false positive 率。

4. **混淆统计显著性和实际显著性。** p-value 0.001 对于 0.01% 的准确率提升没有意义。

5. **在不平衡数据上使用 accuracy。** 99% 准确率在一个 99% 负类的数据集上意味着模型什么都没学到。使用 precision、recall、F1 或 AUC。

6. **挑选指标。** 只报告你的模型赢的指标。诚实的评估报告所有相关指标。

7. **在训练/测试划分间泄露信息。** 在划分前做归一化，或用未来数据预测过去。

8. **没有 variance 估计的小测试集。** 在 100 个样本上评估并声称 2% 的提升是噪声，不是信号。

9. **在数据不独立时假设独立。** 来自同一患者的医学图像，来自同一文档的多个句子。组内的观测是相关的。

10. **P-hacking。** 尝试不同的检验、子集或排除标准直到你得到 p < 0.05。结果是搜索的人工产物。

## Building It

你将从零实现：

1. **描述性统计**（mean、median、mode、standard deviation、percentiles、IQR）
2. **Correlation 函数**（Pearson 和 Spearman，含 covariance matrix）
3. **假设检验**（单样本 t-test、双样本 t-test、chi-squared test）
4. **Bootstrap confidence intervals**（对任意统计量，无需假设）
5. **A/B test 模拟器**（生成数据、检验、检查 Type I error 和 Type II error）
6. **统计 vs 实际显著性演示**（展示大 n 使一切“显著”）

全部从零实现，仅使用 `math` 和 `random`。不使用 numpy，不使用 scipy。

## Key Terms

| 术语 | 定义 |
|---|---|
| Mean | 值之和除以计数。对异常值敏感。 |
| Median | 排序数据的中间值。对异常值 robust。 |
| Standard deviation | Variance 的平方根。用原始单位度量离散程度。 |
| Percentile | 低于该百分比的数据值。 |
| IQR | 四分位距。Q3 减 Q1。中间 50% 的离散程度。 |
| Pearson correlation | 度量两个变量之间的线性关联。范围 [-1, 1]。 |
| Spearman correlation | 使用秩度量单调关联。 |
| Covariance matrix | 所有特征之间 pairwise covariance 的矩阵。 |
| Null hypothesis | 默认的无效应或无差异假设。 |
| p-value | 在零假设为真的前提下，观察到如此极端数据的概率。 |
| Confidence interval | 在给定置信水平下参数的合理值范围。 |
| t-test | 检验均值是否显著不同。使用 t 分布。 |
| Chi-squared test | 检验观察频率是否与期望频率不同。 |
| Effect size | 差异的大小，独立于样本量。Cohen's d 是常用的。 |
| Bonferroni correction | 将显著性阈值除以检验数以控制假阳性。 |
| Bootstrap | 有放回重采样以估计抽样分布。 |
| Type I error | False positive。H0 为真时拒绝 H0。 |
| Type II error | False negative。H0 为假时未拒绝 H0。 |
| Statistical power | 正确拒绝错误 H0 的概率。Power = 1 减 Type II error rate。 |
| Central limit theorem | 样本均值随着样本量增长收敛于正态分布。 |
| Parametric test | 假设数据具有特定分布（通常为正态）。 |
| Non-parametric test | 不做分布假设。基于秩或符号工作。 |
