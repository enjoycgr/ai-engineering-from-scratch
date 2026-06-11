---
name: skill-information-theory
description: 将信息论概念应用于 ML 损失函数、模型评估和特征选择
version: 1.0.0
phase: 1
lesson: 9
tags: [information-theory, entropy, loss-functions]
---

# 机器学习中的信息论

何时在机器学习系统中使用 entropy（熵）、cross-entropy（交叉熵）、KL divergence（KL 散度）和 mutual information（互信息）。

## 决策清单

1. 衡量单个分布的不确定性？使用 **entropy（熵）**。
2. 衡量模型对真实标签的近似程度？使用 **cross-entropy（交叉熵）**（这就是你的分类损失）。
3. 衡量两个分布之间的距离？使用 **KL divergence（KL 散度）**。
4. 检查两个变量是否相关？使用 **mutual information（互信息）**。
5. 报告语言模型质量？使用 **perplexity（困惑度）**（cross-entropy（交叉熵）的指数）。
6. 将一个模型蒸馏到另一个？最小化从教师到学生的 **KL divergence（KL 散度）**。

## 何时使用每种度量

| 度量 | 公式 | 用例 | ML 应用 |
|---|---|---|---|
| Entropy（熵） H(P) | -sum(p log p) | 这个分布有多不确定？ | 数据复杂度、最大熵模型 |
| Cross-entropy（交叉熵） H(P,Q) | -sum(p log q) | 模型 Q 预测真实 P 有多好？ | 分类损失、语言模型损失 |
| KL divergence（KL 散度） D(P\|\|Q) | sum(p log(p/q)) | P 和 Q 有多不同？ | VAE 损失 (ELBO)、知识蒸馏、RLHF |
| Mutual information（互信息） I(X;Y) | H(X) - H(X\|Y) | Y 告诉我们多少关于 X 的信息？ | 特征选择、表示学习 |
| Perplexity（困惑度） | exp(H(P,Q)) 或 2^H | 模型有多困惑？ | 语言模型评估 |
| Conditional entropy（条件熵） H(X\|Y) | -sum(p(x,y) log p(x\|y)) | 知道 Y 后 X 的剩余不确定性 | 特征信息量 |

## 关键关系

```
Cross-entropy（交叉熵）  = Entropy（熵） + KL divergence（KL 散度）
H(P, Q)        = H(P)   + D_KL(P || Q)

由于 H(P) 在训练期间是常数：
  最小化 cross-entropy（交叉熵） = 最小化 KL divergence（KL 散度）

Mutual information（互信息） = Entropy（熵） - Conditional entropy（条件熵）
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)

Perplexity（困惑度） = exp(cross-entropy（交叉熵） in nats（奈特）)
           = 2^(cross-entropy（交叉熵） in bits（比特）)
```

## 速查：公式和单位

| 公式 | Bits（比特）(log 底 2) | Nats（奈特）(log 底 e) |
|---|---|---|
| 信息: -log(p) | -log2(p) | -ln(p) |
| Entropy（熵）: -sum(p log p) | bits（比特） | nats（奈特） |
| 1 nat = | 1.4427 bits（比特） | 1 nat（奈特） |
| PyTorch 默认 | -- | nats（奈特） |
| 信息论文献 | bits（比特） | -- |

## 解释数值

| Entropy（熵）值 | 含义 |
|---|---|
| 0 | 确定性。一个结果概率为 1。 |
| log(n) | 最大不确定性。n 个结果上的均匀分布。 |
| 低 | 分布尖锐。模型自信。 |
| 高 | 分布平坦。模型不确定。 |

| Perplexity（困惑度）值 | 语言模型质量 |
|---|---|
| 1 | 完美预测（实践中永远不会发生） |
| 10 | 平均而言从约 10 个等可能 token 中选择 |
| 50 | GPT-2 在标准基准上的水平 |
| < 10 | 先进模型在充分代表领域的表现 |

## 常见错误

- 计算 KL divergence（KL 散度）后将其当作对称的。D_KL(P||Q) != D_KL(Q||P)。对于对称度量，使用 Jensen-Shannon divergence：JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)，其中 M = 0.5*(P+Q)。
- 忘记 cross-entropy（交叉熵）在 one-hot 标签下简化为 -log(p_true_class)。当真实分布是 one-hot 时，你不需要对所有类别求和。
- 代码中用 log 底 2 但报告 nats（奈特）（或反之）。PyTorch 默认使用自然对数。乘以 log2(e) = 1.4427 将 nats（奈特）转换为 bits（比特）。
- 计算空或零概率事件的 entropy（熵）。惯例：0 * log(0) = 0，因为 lim(p->0) p*log(p) = 0。
- 在不同词汇量之间比较 perplexity（困惑度）。词汇量 50k 且 perplexity（困惑度）30 的模型与词汇量 10k 且 perplexity（困惑度）30 的模型不能直接比较。

## 生产 ML 中每个概念出现的地方

| 概念 | 你在哪里看到它 |
|---|---|
| Cross-entropy（交叉熵）损失 | 每个分类模型 (nn.CrossEntropyLoss) |
| KL divergence（KL 散度） | VAE ELBO、PPO clipping、知识蒸馏 |
| Entropy（熵）正则化 | RL 中的探索奖励（更高的 entropy（熵）= 更多探索） |
| Mutual information（互信息） | 特征选择、InfoNCE 损失（对比学习） |
| Perplexity（困惑度） | 语言模型基准（越低越好） |
| Label smoothing（标签平滑） | 用软目标替换 one-hot，减少 cross-entropy（交叉熵）过度自信 |
| Temperature scaling | 在 softmax（软最大值）前将 logit 除以 T，控制输出的 entropy（熵） |
