# 概率与分布

> 概率是 AI 表达不确定性的语言。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 分钟

## 学习目标

- 从零实现伯努利分布（Bernoulli）、类别分布（Categorical）、泊松分布（Poisson）、均匀分布（Uniform）和正态分布（Normal）的 PMF 与 PDF
- 计算期望值（expected value (期望值)）和方差（variance (方差)），并用中心极限定理（Central Limit Theorem (中心极限定理)）解释为什么高斯分布无处不在
- 构建 softmax（软最大值）和 log-softmax（对数软最大值）函数，并使用数值稳定性技巧（减去最大 logit）
- 从 logit 计算 cross-entropy（交叉熵）损失函数（loss function (损失函数)），并将其与 negative log-likelihood（负对数似然) 联系起来

## 问题背景

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从 50,000 个候选词中挑选下一个词。扩散模型（diffusion model）通过从学习到的分布中采样来生成图像。这些都是概率在发挥作用。

模型的每一次预测都是一个概率分布。每一个损失函数（loss function (损失函数)）都在衡量预测分布与真实分布之间的距离。每一次训练步骤都在调整参数，让一个分布越来越像另一个。没有概率，你无法读懂任何一篇 ML 论文，无法调试任何一个模型，也无法理解为什么你的训练损失变成了 NaN。

## 核心概念

### 事件、样本空间与概率

样本空间 S 是所有可能结果的集合。事件是样本空间的子集。概率将事件映射到 0 到 1 之间的数字。

```
抛硬币：
  S = {正面, 反面}
  P(正面) = 0.5,  P(反面) = 0.5

掷一颗骰子：
  S = {1, 2, 3, 4, 5, 6}
  P(偶数) = P({2, 4, 6}) = 3/6 = 0.5
```

三条公理定义了全部概率论：
1. 对任意事件 A，P(A) >= 0
2. P(S) = 1（必然发生）
3. 当 A 和 B 不能同时发生时，P(A 或 B) = P(A) + P(B)

其他一切（贝叶斯定理、期望、分布）都源自这三条规则。

### 条件概率与独立性

P(A|B) 是在 B 发生的条件下 A 发生的概率。

```
P(A|B) = P(A 且 B) / P(B)

示例：一副扑克牌
  P(国王 | 人头牌) = P(国王 且 人头牌) / P(人头牌)
                   = (4/52) / (12/52)
                   = 4/12 = 1/3
```

当知道一件事对另一件事毫无帮助时，两事件独立：

```
独立：   P(A|B) = P(A)
等价于： P(A 且 B) = P(A) * P(B)
```

抛硬币是独立的。不放回地抽牌不是独立的。

### 概率质量函数 vs 概率密度函数

离散随机变量拥有概率质量函数（probability mass function (PMF, 概率质量函数)）。每个结果都有可以直接读取的特定概率。

```
PMF: P(X = k)

公平骰子：
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  所有概率之和 = 1
```

连续随机变量拥有概率密度函数（probability density function (PDF, 概率密度函数)）。单点处的密度不是概率。概率来自对区间上的密度进行积分。

```
PDF: f(x)

P(a <= X <= b) = 从 a 到 b 对 f(x) 积分

f(x) 可以大于 1（密度，不是概率）
从 -inf 到 +inf 对 f(x) dx 积分 = 1
```

这一区分在 ML 中很重要。分类输出是 PMF（离散选择）。VAE 的隐空间使用 PDF（连续）。

### 常见分布

**Bernoulli（伯努利）：** 一次试验，两种结果。用于二元分类建模。

```
P(X = 1) = p
P(X = 0) = 1 - p
均值 = p,  方差 = p(1-p)
```

**Categorical（类别分布）：** 一次试验，k 种结果。用于多分类建模（softmax 输出）。

```
P(X = i) = p_i,  其中 sum(p_i) = 1
示例：P(猫) = 0.7,  P(狗) = 0.2,  P(鸟) = 0.1
```

**Uniform（均匀分布）：** 所有结果等可能。用于随机初始化。

```
离散：P(X = k) = 1/n，k 属于 {1, ..., n}
连续：f(x) = 1/(b-a)，x 属于 [a, b]
```

**Normal（正态分布 / 高斯分布）：** 钟形曲线。由均值 mu 和方差 sigma^2 参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

标准正态：mu = 0, sigma = 1
  68% 的数据在 1 个 sigma 内
  95% 在 2 个 sigma 内
  99.7% 在 3 个 sigma 内
```

**Poisson（泊松分布）：** 固定区间内稀有事件的计数。用于建模事件发生率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
均值 = lambda,  方差 = lambda
```

### 期望值与方差

期望值（expected value (期望值)）是加权平均结果。

```
离散：   E[X] = 所有 x_i * P(X = x_i) 的和
连续：   E[X] = 对 x * f(x) dx 积分
```

方差（variance (方差)）衡量围绕均值的离散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
标准差 = sqrt(Var(X))
```

在 ML 中，期望值表现为损失函数（loss function (损失函数)）（对数据分布的平均损失）。方差告诉你模型的稳定性。梯度的高方差意味着训练噪声大。

### 联合分布与边缘分布

联合分布 P(X, Y) 描述两个随机变量一起的行为。

联合 PMF 示例（X = 天气，Y = 是否带伞）：

| | Y=0（没带伞） | Y=1（带伞） | 边缘 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨天） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布通过对另一变量求和得到：

```
P(X = x) = 对所有 y 求和 P(X = x, Y = y)
```

上表中的行合计和列合计就是边缘分布。

### 为什么正态分布无处不在

中心极限定理（Central Limit Theorem (中心极限定理)）：许多独立随机变量之和（或平均值）收敛于正态分布，无论原始分布是什么。

```
掷 1 颗骰子：  均匀分布（平坦）
2 颗骰子平均： 三角分布（尖峰）
30 颗骰子平均：几乎完美的钟形曲线

这对 ANY 起始分布都成立。
```

这就是为什么：
- 测量误差近似正态分布（许多小的独立来源）
- 神经网络中的权重初始化使用正态分布
- SGD 中的梯度噪声近似正态分布（许多样本梯度的和）
- 正态分布是给定均值和方差下的最大熵分布

### 对数概率

原始概率会引起数值问题。将许多小概率相乘会迅速下溢为 0。

```
P(句子) = P(词1) * P(词2) * ... * P(词_n)
        = 0.01 * 0.003 * 0.02 * ...
        -> 0.0（约 30 项后下溢）
```

对数概率（log probability）解决了这个问题。乘法变加法。

```
log P(句子) = log P(词1) + log P(词2) + ... + log P(词_n)
            = -4.6 + -5.8 + -3.9 + ...
            -> 有限数字（不下溢）
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率始终 <= 0（因为 0 < P <= 1）
- 越负 = 越不可能
- cross-entropy（交叉熵）损失就是正确类别的 negative log-likelihood（负对数似然）

### Softmax 作为概率分布

神经网络输出原始分数（logits）。Softmax（软最大值）将它们转换为合法的概率分布。

```
softmax(z_i) = exp(z_i) / 对所有 j 求和 exp(z_j)

性质：
  - 所有输出在 (0, 1) 之间
  - 所有输出之和为 1
  - 保持输入的相对排序
  - exp() 放大了 logit 之间的差距
```

Softmax 技巧：在指数化之前减去最大 logit，防止溢出。

```
z = [100, 101, 102]
exp(102) = 溢出

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1（安全）

结果相同，不溢出。
```

Log-softmax（对数软最大值）将 softmax 和 log 结合以获得数值稳定性。PyTorch 在内部为 cross-entropy（交叉熵）损失使用它。

### 采样

采样意味着从分布中抽取随机值。在 ML 中：
- Dropout（随机失活）随机采样要置零的神经元
- 数据增强采样随机变换
- 语言模型从预测分布中采样下一个 token
- 扩散模型采样噪声并逐步去噪

从任意分布采样需要逆变换采样、拒绝采样或重参数化技巧（VAE 中使用）等技术。

## 动手实现

### 第一步：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 第二步：从零实现 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 第三步：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 第四步：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 第五步：Softmax 和对数概率

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 第六步：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 第七步：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

完整实现及所有可视化见 `code/probability.py`。

## 调用库函数

使用 NumPy 和 SciPy，以上所有内容都可以一行完成：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

你从零实现了它们。现在你知道库调用在做什么。

## 练习

1. 为指数分布实现逆变换采样。采样 10,000 个值并通过与真实 PDF 比较直方图来验证。

2. 为两颗 loaded dice 构建联合分布表。计算边缘分布并检查骰子是否独立。

3. 当正确类别索引为 3 时，为一个输出 logits `[2.0, 0.5, -1.0, 3.0, 0.1]` 的 5 类分类器计算 cross-entropy（交叉熵）损失。然后用 PyTorch 的 `nn.CrossEntropyLoss` 验证你的答案。

4. 编写一个函数，接收对数概率列表并返回最可能的序列、总对数概率和等效的原始概率。用每个词概率为 0.01 的 50 个词的句子测试它。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 样本空间 | "所有可能性" | 一次实验所有可能结果的集合 S |
| PMF | "概率函数" | 给出每个离散结果精确概率的函数，所有概率之和为 1 |
| PDF | "概率曲线" | 连续变量的密度函数。需要对区间积分才能得到概率 |
| 条件概率 | "给定某事的概率" | P(A\|B) = P(A 且 B) / P(B)。贝叶斯思维和贝叶斯定理的基础 |
| 独立性 | "它们互不影响" | P(A 且 B) = P(A) * P(B)。知道一件事对另一件事毫无帮助 |
| 期望值 | "平均值" | 所有结果的概率加权和。损失函数就是一个期望值 |
| 方差 | "离散程度" | 与均值之差的平方的期望值。高方差 = 噪声大、估计不稳定 |
| 正态分布 | "钟形曲线" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。因 CLT 而无处不在 |
| 中心极限定理 | "平均值变成正态" | 许多独立样本的均值收敛于正态分布，无论来源是什么 |
| 联合分布 | "两个变量一起" | P(X, Y) 描述 X 和 Y 每种组合的概率 |
| 边缘分布 | "把另一个变量求和掉" | P(X) = 对 y 求和 P(X, Y)。从联合分布中恢复单个变量的分布 |
| 对数概率 | "概率的对数" | log P(x)。把乘积变成求和，防止长序列中的数值下溢 |
| Softmax | "把分数变成概率" | softmax(z_i) = exp(z_i) / 求和(exp(z_j))。将实数值 logit 映射为合法概率分布 |
| 交叉熵 | "损失函数" | -求和(p_true * log(p_predicted))。衡量两个分布的差异。越低越好 |
| Logits | "原始模型输出" | Softmax 之前的未归一化分数。名称来自 logistic 函数 |
| 采样 | "抽取随机值" | 按照概率分布生成值。模型生成输出的方式 |

## 延伸阅读

- [3Blue1Brown: 但什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 为什么平均值变成正态分布的可视化证明
- [Stanford CS229 概率复习](https://cs229.stanford.edu/section/cs229-prob.pdf) - 涵盖本文内容及更多的简明参考
- [Log-Sum-Exp 技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 为什么数值稳定性很重要以及如何实现
