# 信息论

> 信息论衡量惊讶程度。损失函数建立在它之上。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lesson 06 (Probability)
**Time:** ~60 分钟

## 学习目标

- 从零计算 entropy（熵）、cross-entropy（交叉熵）和 KL divergence（KL 散度），并解释它们之间的关系
- 推导为什么最小化 cross-entropy（交叉熵）损失等价于最大化对数似然
- 计算特征与目标之间的 mutual information（互信息）以排序特征重要性
- 解释 perplexity（困惑度）作为语言模型每次选择的有效词汇量大小

## 问题背景

你在每个分类模型中都调用 `CrossEntropyLoss()`。你在每篇语言模型论文中都看到 "perplexity（困惑度）"。你在 VAE、知识蒸馏和 RLHF 中读到 KL divergence（KL 散度）。这些不是互不相关的概念。它们是同一个想法戴着不同的帽子。

信息论给了你推理不确定性、压缩和预测的语言。Claude Shannon 在 1948 年发明它来解决通信问题。事实证明，训练神经网络是一个通信问题：模型试图通过由学习到的权重构成的噪声信道传输正确的标签。

本课从头构建每个公式，让你看到它们从何而来以及为什么有效。

## 核心概念

### 信息内容（惊讶度）

当不太可能发生的事件发生时，它携带更多信息。硬币正面朝上？不令人惊讶。彩票中奖？非常令人惊讶。

概率为 p 的事件的信息内容（information content (信息内容)）为：

```
I(x) = -log(p(x))
```

用以 2 为底的对数得到 bits（比特）。用自然对数得到 nats（奈特）。同样的思想，不同的单位。

```
事件              概率    惊讶度 (bits)
公平硬币正面    0.5            1.0
掷出 6          0.167          2.58
1/1000 事件     0.001          9.97
必然事件        1.0            0.0
```

必然事件携带零信息。你已经知道它会发生。

### Entropy（熵）（平均惊讶度）

Entropy（熵）是分布所有可能结果的期望惊讶度。

```
H(P) = -sum( p(x) * log(p(x)) )  对所有 x
```

公平硬币对于二元变量具有最大 entropy（熵）：1 bit。有偏硬币（99% 正面）entropy（熵）低：0.08 bits。你已经知道会发生什么，所以每次翻转几乎不告诉你任何新东西。

```
公平硬币:    H = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 bit
有偏硬币:  H = -(0.99 * log2(0.99) + 0.01 * log2(0.01)) = 0.08 bits
```

Entropy（熵）衡量分布中不可约的不确定性。你无法压缩到它以下。

### Cross-entropy（交叉熵）（你每天使用的损失函数）

Cross-entropy（交叉熵）衡量当你使用分布 Q 编码实际来自分布 P 的事件时的平均惊讶度。

```
H(P, Q) = -sum( p(x) * log(q(x)) )  对所有 x
```

P 是真实分布（标签）。Q 是模型的预测。如果 Q 完美匹配 P，cross-entropy（交叉熵）等于 entropy（熵）。任何不匹配都会使它更大。

在分类中，P 是 one-hot 向量（真实类别概率为 1，其余为 0）。这使 cross-entropy（交叉熵）简化为：

```
H(P, Q) = -log(q(true_class))
```

这就是分类 cross-entropy（交叉熵）损失函数的完整公式。最大化正确类别的预测概率。

### KL Divergence（KL 散度）（分布之间的距离）

KL divergence（KL 散度）衡量使用 Q 代替 P 带来的额外惊讶度。

```
D_KL(P || Q) = sum( p(x) * log(p(x) / q(x)) )  对所有 x
             = H(P, Q) - H(P)
```

Cross-entropy（交叉熵）等于 entropy（熵）加上 KL divergence（KL 散度）。由于真实分布的 entropy（熵）在训练期间是常数，最小化 cross-entropy（交叉熵）等同于最小化 KL divergence（KL 散度）。你在将模型的分布推向真实分布。

KL divergence（KL 散度）不对称：D_KL(P || Q) != D_KL(Q || P)。它不是真正的距离度量。

### Mutual Information（互信息）

Mutual information（互信息）衡量知道一个变量能告诉你多少关于另一个变量的信息。

```
I(X; Y) = H(X) - H(X|Y)
        = H(X) + H(Y) - H(X, Y)
```

如果 X 和 Y 独立，mutual information（互信息）为零。知道一个对另一个毫无帮助。如果它们完美相关，mutual information（互信息）等于任一变量的 entropy（熵）。

在特征选择中，特征与目标之间的高 mutual information（互信息）意味着该特征有用。低 mutual information（互信息）意味着它是噪声。

### Conditional Entropy（条件熵）

H(Y|X) 衡量在观察到 X 后 Y 还剩下多少不确定性。

```
H(Y|X) = H(X,Y) - H(X)
```

两个极端：
- 如果 X 完全确定 Y，则 H(Y|X) = 0。知道 X 消除了关于 Y 的所有不确定性。示例：X = 摄氏温度，Y = 华氏温度。
- 如果 X 对 Y 毫无帮助，则 H(Y|X) = H(Y)。知道 X 根本不减少你的不确定性。示例：X = 抛硬币，Y = 明天的天气。

Conditional entropy（条件熵）始终非负且不超过 H(Y)：

```
0 <= H(Y|X) <= H(Y)
```

在机器学习中，conditional entropy（条件熵）出现在决策树中。在每个分裂处，算法选择最小化 H(Y|X) 的特征 X——即消除关于标签 Y 最多不确定性的特征。

### Joint Entropy（联合熵）

H(X,Y) 是 X 和 Y 联合分布的 entropy（熵）。

```
H(X,Y) = -sum sum p(x,y) * log(p(x,y))   对所有 x, y
```

关键性质：

```
H(X,Y) <= H(X) + H(Y)
```

当 X 和 Y 独立时取等号。如果它们共享信息，联合 entropy（熵）小于单个 entropy（熵）之和。"缺失"的 entropy（熵）正是 mutual information（互信息）。

```mermaid
graph TD
    subgraph "信息论维恩图"
        direction LR
        HX["H(X)"]
        HY["H(Y)"]
        MI["I(X;Y)<br/>Mutual<br/>Information<br/>（互信息）"]
        HXgY["H(X|Y)<br/>= H(X) - I(X;Y)"]
        HYgX["H(Y|X)<br/>= H(Y) - I(X;Y)"]
        HXY["H(X,Y) = H(X) + H(Y) - I(X;Y)"]
    end

    HXgY --- MI
    MI --- HYgX
    HX -.- HXgY
    HX -.- MI
    HY -.- MI
    HY -.- HYgX
    HXY -.- HXgY
    HXY -.- MI
    HXY -.- HYgX
```

关系：
- H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
- I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
- H(X,Y) = H(X) + H(Y) - I(X;Y)

### Mutual Information（互信息）深度解析

Mutual information（互信息）I(X;Y) 量化知道一个变量能减少多少关于另一个变量的不确定性。

```
I(X;Y) = H(X) - H(X|Y)
       = H(Y) - H(Y|X)
       = H(X) + H(Y) - H(X,Y)
       = sum sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
```

性质：
- I(X;Y) >= 0 始终成立。观察某物永远不会让你丢失信息。
- I(X;Y) = 0 当且仅当 X 和 Y 独立。
- I(X;Y) = I(Y;X)。它是对称的，不像 KL divergence（KL 散度）。
- I(X;X) = H(X)。一个变量与自身共享全部信息。

**用于特征选择的 Mutual information（互信息）。** 在 ML 中，你想要对目标有信息量的特征。Mutual information（互信息）给了你一个 principled 的特征排序方式：

1. 对每个特征 X_i，计算 I(X_i; Y)，其中 Y 是目标变量。
2. 按 MI 得分排序特征。
3. 保留前 k 个特征。

这对特征与目标之间的任何关系都有效——线性、非线性、单调或非单调。Correlation 只捕捉线性关系。MI 捕捉一切。

| 方法 | 检测 | 计算成本 | 处理类别变量？ |
|--------|---------|-------------------|---------------------|
| Pearson correlation | 线性关系 | O(n) | 否 |
| Spearman correlation | 单调关系 | O(n log n) | 否 |
| Mutual information | 任何统计依赖 | 用分箱 O(n log n) | 是 |

### Label Smoothing（标签平滑）与 Cross-entropy（交叉熵）

标准分类使用硬目标：[0, 0, 1, 0]。真实类别概率为 1，其余为 0。Label smoothing（标签平滑）用软目标替换它们：

```
soft_target = (1 - epsilon) * hard_target + epsilon / num_classes
```

当 epsilon = 0.1 且 4 个类别时：
- 硬目标:  [0, 0, 1, 0]
- 软目标:  [0.025, 0.025, 0.925, 0.025]

从信息论角度，label smoothing（标签平滑）增加了目标分布的 entropy（熵）。硬 one-hot 目标的 entropy（熵）为 0——没有不确定性。软目标的 entropy（熵）为正。

为什么这有帮助：
- 防止模型将 logit 推向极端值（无限 logit 才能完美匹配 one-hot 目标下的 cross-entropy（交叉熵））
- 充当 regularization（正则化）：模型不能 100% 自信
- 改善校准：预测概率更好地反映真实不确定性
- 缩小训练与 inference（推理）行为之间的差距

使用 label smoothing（标签平滑）的 cross-entropy（交叉熵）损失变为：

```
L = (1 - epsilon) * CE(hard_target, prediction) + epsilon * H_uniform(prediction)
```

第二项惩罚远离均匀的预测——直接对置信度进行 regularization（正则化）。

### 为什么 Cross-entropy（交叉熵）是 THE 分类损失

三个视角，同一个结论。

**信息论视角。** Cross-entropy（交叉熵）衡量你使用模型分布而非真实分布时浪费了多少 bits。最小化它让你的模型成为现实最高效的编码器。

**最大似然视角。** 对于 N 个训练样本，真实类别为 y_i：

```
Likelihood     = product( q(y_i) )
Log-likelihood = sum( log(q(y_i)) )
Negative log-likelihood = -sum( log(q(y_i)) )
```

最后一行就是 cross-entropy（交叉熵）损失。最小化 cross-entropy（交叉熵）= 最大化训练数据在你模型下的似然。

**梯度视角。** Cross-entropy（交叉熵）关于 logit 的梯度就是 (predicted - true)。干净、稳定、计算快速。这就是它与 softmax（软最大值）完美配对的原因。

### Bits（比特） vs Nats（奈特）

唯一的区别是对数的底。

```
log base 2   -> bits（比特）      (信息论传统)
log base e   -> nats（奈特）      (机器学习惯例)
log base 10  -> hartleys         (很少使用)
```

1 nat = 1/ln(2) bits = 1.4427 bits。PyTorch 和 TensorFlow 默认使用自然对数（nats（奈特））。

### Perplexity（困惑度）

Perplexity（困惑度）是 cross-entropy（交叉熵）的指数。它告诉你模型不确定的有效等概率选择数量。

```
Perplexity（困惑度） = 2^H(P,Q)   (如果使用 bits)
Perplexity（困惑度） = e^H(P,Q)   (如果使用 nats)
```

Perplexity（困惑度）为 50 的语言模型，平均而言，就像必须从 50 个可能的下一个 token 中均匀挑选一样困惑。越低越好。

GPT-2 在常见基准上达到了 perplexity（困惑度）~30。现代模型在充分代表的领域达到了个位数。

## 动手实现

### 第一步：信息内容和 entropy（熵）

```python
import math

def information_content(p, base=2):
    if p <= 0 or p > 1:
        return float('inf') if p <= 0 else 0.0
    return -math.log(p) / math.log(base)

def entropy(probs, base=2):
    return sum(
        p * information_content(p, base)
        for p in probs if p > 0
    )

fair_coin = [0.5, 0.5]
biased_coin = [0.99, 0.01]
fair_die = [1/6] * 6

print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"Fair die entropy:    {entropy(fair_die):.4f} bits")
```

### 第二步：Cross-entropy（交叉熵）和 KL divergence（KL 散度）

```python
def cross_entropy(p, q, base=2):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * (-math.log(qi) / math.log(base))
    return total

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

true_dist = [0.7, 0.2, 0.1]
good_model = [0.6, 0.25, 0.15]
bad_model = [0.1, 0.1, 0.8]

print(f"Entropy of true dist:     {entropy(true_dist):.4f} bits")
print(f"CE (good model):          {cross_entropy(true_dist, good_model):.4f} bits")
print(f"CE (bad model):           {cross_entropy(true_dist, bad_model):.4f} bits")
print(f"KL divergence (good):     {kl_divergence(true_dist, good_model):.4f} bits")
print(f"KL divergence (bad):      {kl_divergence(true_dist, bad_model):.4f} bits")
```

### 第三步：Cross-entropy（交叉熵）作为分类损失

```python
def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

logits = [2.0, 1.0, 0.1]
true_class = 0

probs = softmax(logits)
loss = cross_entropy_loss(true_class, logits)

print(f"Logits:      {logits}")
print(f"Softmax:     {[f'{p:.4f}' for p in probs]}")
print(f"True class:  {true_class}")
print(f"Loss:        {loss:.4f} nats")
print(f"Perplexity:  {math.exp(loss):.2f}")
```

### 第四步：Cross-entropy（交叉熵）等于 negative log-likelihood（负对数似然）

```python
import random

random.seed(42)

n_samples = 1000
n_classes = 3
true_labels = [random.randint(0, n_classes - 1) for _ in range(n_samples)]
model_logits = [[random.gauss(0, 1) for _ in range(n_classes)] for _ in range(n_samples)]

ce_loss = sum(
    cross_entropy_loss(label, logits)
    for label, logits in zip(true_labels, model_logits)
) / n_samples

nll = -sum(
    math.log(softmax(logits)[label])
    for label, logits in zip(true_labels, model_logits)
) / n_samples

print(f"Cross-entropy loss:      {ce_loss:.6f}")
print(f"Negative log-likelihood: {nll:.6f}")
print(f"Difference:              {abs(ce_loss - nll):.2e}")
```

### 第五步：Mutual information（互信息）

```python
def mutual_information(joint_probs, base=2):
    rows = len(joint_probs)
    cols = len(joint_probs[0])

    margin_x = [sum(joint_probs[i][j] for j in range(cols)) for i in range(rows)]
    margin_y = [sum(joint_probs[i][j] for i in range(rows)) for j in range(cols)]

    mi = 0.0
    for i in range(rows):
        for j in range(cols):
            pxy = joint_probs[i][j]
            if pxy > 0:
                mi += pxy * math.log(pxy / (margin_x[i] * margin_y[j])) / math.log(base)
    return mi

independent = [[0.25, 0.25], [0.25, 0.25]]
dependent = [[0.45, 0.05], [0.05, 0.45]]

print(f"MI (independent): {mutual_information(independent):.4f} bits")
print(f"MI (dependent):   {mutual_information(dependent):.4f} bits")
```

## 调用库函数

使用 NumPy，这是你在实践中会使用的方式：

```python
import numpy as np

def np_entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    result = np.zeros_like(p)
    result[mask] = p[mask] * np.log(p[mask])
    return -result.sum()

def np_cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return -(p[mask] * np.log(q[mask])).sum()

def np_kl_divergence(p, q):
    return np_cross_entropy(p, q) - np_entropy(p)

true = np.array([0.7, 0.2, 0.1])
pred = np.array([0.6, 0.25, 0.15])
print(f"Entropy:    {np_entropy(true):.4f} nats")
print(f"Cross-ent:  {np_cross_entropy(true, pred):.4f} nats")
print(f"KL div:     {np_kl_divergence(true, pred):.4f} nats")
```

你从零构建了 `torch.nn.CrossEntropyLoss()` 内部做的事情。现在你知道为什么训练期间损失会下降：模型的预测分布正在接近真实分布，以对数尺度衡量浪费的信息。

## 练习

1. 计算英文字母表在均匀分布假设下的 entropy（熵）（26 个字母）。然后用实际字母频率估计它。哪个更高，为什么？

2. 一个模型对真实类别为 1 的样本输出 logits [5.0, 2.0, 0.5]。手算 cross-entropy（交叉熵）损失，然后用你的 `cross_entropy_loss` 函数验证。什么 logit 会给出零损失？

3. 证明 KL divergence（KL 散度）不对称。选两个分布 P 和 Q，计算 D_KL(P || Q) 和 D_KL(Q || P)。解释它们为何不同。

4. 构建一个计算 token 预测序列 perplexity（困惑度）的函数。给定一个 (true_token_index, predicted_logits) 对列表，返回序列的 perplexity（困惑度）。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Information content（信息内容） | "惊讶度" | 编码一个事件所需的 bits（比特）（或 nats（奈特））数量：-log(p) |
| Entropy（熵） | "随机性" | 分布所有结果的平均惊讶度。衡量不可约的不确定性。 |
| Cross-entropy（交叉熵） | "损失函数" | 使用模型分布 Q 编码来自真实分布 P 的事件时的平均惊讶度。 |
| KL divergence（KL 散度） | "分布之间的距离" | 使用 Q 代替 P 浪费的额外 bits。等于 cross-entropy（交叉熵）减去 entropy（熵）。不对称。 |
| Mutual information（互信息） | "X 和 Y 有多相关" | 知道 Y 减少的关于 X 的不确定性。为零表示独立。 |
| Softmax（软最大值） | "把分数变成概率" | 取指数并归一化。将任何实值向量映射为合法概率分布。 |
| Perplexity（困惑度） | "模型有多困惑" | Cross-entropy（交叉熵）的指数。模型每一步等效选择的词汇量大小。 |
| Bits（比特） | "香农的单位" | 用 log 底 2 衡量的信息。一个 bit 解决一次公平硬币翻转。 |
| Nats（奈特） | "ML 的单位" | 用自然对数衡量的信息。PyTorch 和 TensorFlow 默认使用。 |
| Negative log-likelihood（负对数似然） | "NLL 损失" | 对 one-hot 标签与 cross-entropy（交叉熵）损失相同。最小化它即最大化正确预测的概率。 |

## 延伸阅读

- [Shannon 1948: A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) - 原始论文，至今可读
- [Visual Information Theory (Chris Olah)](https://colah.github.io/posts/2015-09-Visual-Information/) - entropy（熵）和 KL divergence（KL 散度）的最佳可视化解释
- [PyTorch CrossEntropyLoss 文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - 框架如何实现你刚刚构建的内容
