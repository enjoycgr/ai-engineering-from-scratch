---
name: skill-probability-reasoning
description: 为给定的 ML 问题选择正确的概率分布
version: 1.0.0
phase: 1
lesson: 6
tags: [probability, distributions, modeling]
---

# 概率分布选择

在为数据建模、设计损失函数或设定先验时，如何挑选合适的分布。

## 决策清单

1. 结果是离散（类别、计数）还是连续（测量值、分数）？
2. 结果是否有界（例如 [0, 1]）还是无界？
3. 有多少种可能结果？两种？k 种？无限？
4. 数据是对称的还是偏斜的？
5. 事件是独立的还是相关的？
6. 你是在建模速率、计数、比例还是测量值？

## 分布决策树

```
变量是离散的吗？
  是 --> 只有 2 种结果？ --> Bernoulli (p)
     |    k 种结果，一次试验？ --> Categorical (p1...pk)
     |    k 种结果，n 次试验？ --> Multinomial (n, p1...pk)
     |    n 次试验中的成功次数？ --> Binomial (n, p)
     |    每区间的事件计数？ --> Poisson (lambda)
     |    首次成功前的试验次数？ --> Geometric (p)
     |    r 次成功前的试验次数？ --> Negative Binomial (r, p)
  否 --> 对称、钟形？ --> Normal (mu, sigma)
     |   正值，右偏？ --> Log-normal 或 Exponential
     |   有界于 [0, 1]？ --> Beta (alpha, beta)
     |   正值，形状灵活？ --> Gamma (alpha, beta)
     |   事件之间的时间？ --> Exponential (lambda)
     |   需要重尾？ --> Student's t (nu) 或 Cauchy
     |   多元、钟形？ --> Multivariate Normal
     |   在单纯形上（和为 1）？ --> Dirichlet (alpha)
```

## 将真实 ML 场景映射到分布

| 场景 | 分布 | 参数 |
|---|---|---|
| 二分类输出 | Bernoulli | p = sigmoid(logit) |
| 多分类输出 | Categorical | p = softmax(logits) |
| 语言模型中的 token 预测 | 词汇上的 Categorical | p 来自 softmax |
| 像素强度（归一化后） | Beta 或 Uniform [0, 1] | 取决于图像统计 |
| 文档中的词数 | Poisson | lambda = 平均词数 |
| 用户请求之间的时间 | Exponential | lambda = 请求速率 |
| 测量误差 | Normal | mu = 0, sigma 来自数据 |
| 权重初始化 | Normal 或 Uniform | Kaiming/Xavier 规则 |
| VAE 隐空间先验 | Standard Normal | mu = 0, sigma = 1 |
| 比例上的贝叶斯先验 | Beta | alpha, beta 来自信念 |
| 类别权重上的贝叶斯先验 | Dirichlet | alpha 向量 |
| 回归目标中的噪声 | Normal | mu = 0, sigma 估计得到 |
| 抗异常值回归 | Student's t | 低自由度 |
| 持续时间/寿命建模 | Weibull 或 Gamma | shape 和 scale |
| 每篇文档的主题分布（LDA） | Dirichlet | alpha < 1 用于稀疏 |

## 分布误用的情况

- 数据有硬下界（例如价格、距离）时使用 Normal。正态分布会给负值分配非零概率。改用 log-normal 或 gamma。
- 方差与均值不同时使用 Poisson。Poisson 假设均值 = 方差。如果方差 > 均值，改用 negative binomial。
- 多分类问题使用 Bernoulli。Bernoulli 严格是二元的。k > 2 时使用 categorical。
- 观测相关时假设独立。时间序列、空间数据和分组数据违反独立性。改用自回归或分层模型。

## 常见错误

- 将 PDF 值与概率混淆。PDF 可以超过 1。概率来自对 PDF 在区间上积分。
- 忘记 softmax 输出是 categorical 概率，不是独立的 Bernoulli 概率。它们按构造求和为 1。
- 有领域知识时使用均匀先验。信息性先验能减少方差，如果选择得当，不会使结果有偏。
- 将对数概率当作概率处理。对数概率始终为负（或零）。它们不求和为 1。

## 速查：分布性质

| 分布 | 支撑集 | 均值 | 方差 | 关键性质 |
|---|---|---|---|---|
| Bernoulli(p) | {0, 1} | p | p(1-p) | 最简单的离散分布 |
| Binomial(n, p) | {0..n} | np | np(1-p) | n 个 Bernoulli 之和 |
| Poisson(lam) | {0, 1, 2, ...} | lam | lam | 均值 = 方差 |
| Normal(mu, s^2) | (-inf, inf) | mu | s^2 | 给定均值/方差下的最大熵 |
| Exponential(lam) | [0, inf) | 1/lam | 1/lam^2 | 无记忆性 |
| Beta(a, b) | [0, 1] | a/(a+b) | ab/((a+b)^2(a+b+1)) | Binomial 的共轭先验 |
| Gamma(a, b) | (0, inf) | a/b | a/b^2 | Poisson 的共轭先验 |
| Dirichlet(alpha) | 单纯形 | alpha_i/sum | (见公式) | Categorical 的共轭先验 |
