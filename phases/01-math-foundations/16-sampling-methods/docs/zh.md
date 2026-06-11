# 采样方法

> 采样是 AI 探索可能性空间的方式。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 分钟

## Learning Objectives

- 仅使用均匀随机数从零实现 inverse CDF (逆累积分布函数)、rejection sampling (拒绝采样) 和 importance sampling (重要性采样)
- 为语言模型 token 生成构建 temperature sampling (温度采样)、top-k sampling 和 top-p (nucleus) sampling
- 解释 reparameterization trick (重参数化技巧) 以及它为什么能在 VAE 中实现通过采样的 backpropagation (反向传播)
- 运行 Metropolis-Hastings MCMC (马尔可夫链蒙特卡洛) 以从未归一化的目标分布中采样

## The Problem

语言模型处理完你的提示后，产生一个包含 50,000 个 logits 的向量。词汇表中的每个 token 对应一个。现在它必须选一个。怎么选？

如果它总是选概率最高的 token，每个回答都一模一样。确定性的。无聊的。如果它均匀随机地选，输出是乱码。答案存在于这两个极端之间的某个位置，而这个位置由采样控制。

采样不限于文本生成。Reinforcement learning (强化学习) 通过采样轨迹来估计策略梯度。VAE (Variational Autoencoder / 变分自编码器) 通过从学习到的分布中采样并反向传播随机性来学习隐表示。Diffusion models (扩散模型) 通过采样噪声并迭代去噪来生成图像。Monte Carlo (蒙特卡洛) 方法估计没有闭式解的积分。MCMC (Markov Chain Monte Carlo / 马尔可夫链蒙特卡洛) 算法探索不可能枚举的高维后验分布。

每个生成式 AI 系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课从零构建每一个主要采样方法，从均匀随机数开始，到驱动现代 LLM 和生成模型的技术结束。

## The Concept

### 为什么采样重要

采样在 AI 和机器学习中扮演四个基本角色：

**生成。** 语言模型、diffusion models 和 GAN 都通过采样产生输出。采样算法直接控制创造力、连贯性和多样性。Temperature、top-k 和 nucleus sampling 是工程师每天调节的旋钮。

**训练。** Stochastic gradient descent (随机梯度下降) 采样 mini-batch。Dropout (随机失活) 采样要关闭的神经元。数据增强采样随机变换。Importance sampling (重要性采样) 在 reinforcement learning (PPO, TRPO) 中重新加权样本以降低梯度 variance。

**估计。** ML 中的许多量没有闭式解。数据分布上的期望 loss、基于能量的模型的配分函数、贝叶斯推断中的证据。Monte Carlo 估计通过在这些样本上取平均来近似所有这些东西。

**探索。** MCMC 算法探索贝叶斯推断中的后验分布。进化策略采样参数扰动。Thompson sampling 在 bandit 问题中平衡探索与利用。

核心挑战：你只能直接从简单分布（均匀、正态）中采样。对于其他一切，你需要一种将简单样本转换为目标分布样本的方法。

### 均匀随机采样

每个采样方法都从这里开始。均匀随机数生成器在 [0, 1) 中产生值，每个等长子区间具有相等概率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    对于 0 <= a <= b <= 1

性质：
  E[U] = 0.5
  Var(U) = 1/12
```

要从 n 个项的离散集合中均匀采样，生成 U 并返回 floor(n * U)。要从连续范围 [a, b] 中采样，计算 a + (b - a) * U。

关键洞察：单个均匀随机数恰好包含从任何分布中产生一个样本所需的正确随机性。诀窍是找到正确的变换。

### Inverse CDF Method（逆累积分布函数方法 / Inverse Transform Sampling）

Cumulative distribution function (CDF / 累积分布函数) 将值映射到概率：

```
F(x) = P(X <= x)

性质：
  F 非递减
  F(-inf) = 0
  F(+inf) = 1
  F 将实数轴映射到 [0, 1]
```

Inverse CDF 将概率映射回值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。

```
算法：
  1. 生成 u ~ Uniform(0, 1)
  2. 返回 F_inverse(u)

为什么有效：
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**指数分布示例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

对 F(x) = u 求解 x：
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

由于 (1 - U) 和 U 同分布：
  x = -ln(u) / lambda
```

当你能写出闭式 F_inverse 时，这完美工作。对于正态分布，没有闭式 inverse CDF，所以我们使用其他方法（Box-Muller，或数值近似）。

**离散版本：** 对于离散分布，构建 CDF 作为累积和，生成 U，找到累积和首次超过 U 的第一个索引。这就是 Lesson 06 中 `sample_categorical` 的工作原理。

### Rejection Sampling（拒绝采样）

当你无法 invert CDF 但能 up to a constant 评估目标 PDF 时，rejection sampling 有效。

```
目标分布: p(x)  （可评估，可能未归一化）
提案分布: q(x)  （可从中采样）
界限: M 使得 p(x) <= M * q(x) 对所有 x

算法：
  1. 从 q(x) 采样 x
  2. 从 Uniform(0, 1) 采样 u
  3. 如果 u < p(x) / (M * q(x))，接受 x
  4. 否则，拒绝并返回步骤 1

接受率 = 1/M
```

界限 M 越紧，接受率越高。在低维（1-3维）中，rejection sampling 效果很好。在高维中，接受率指数下降，因为大部分提案体积被拒绝。这是拒绝采样的维度灾难。

**示例：从截断正态采样。** 使用截断范围上的均匀提案。包络 M 是该范围内正态 PDF 的最大值。

**示例：从半圆采样。** 在边界矩形内均匀提案。如果点落在半圆内则接受。这就是 Monte Carlo 计算 pi 的方式：接受率等于面积比 pi/4。

### Importance Sampling（重要性采样）

有时你不需要从目标分布 p(x) 中采样。你需要估计 p(x) 下的期望，而你有来自不同分布 q(x) 的样本。

```
目标: 估计 E_p[f(x)] = f(x) * p(x) dx 的积分

改写：
  E_p[f(x)] = f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

其中 w(x) = p(x) / q(x)  是 importance weights。

估计量：
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    其中 x_i ~ q(x)
```

这在 reinforcement learning 中至关重要。在 PPO (Proximal Policy Optimization) 中，你在旧策略 pi_old 下收集轨迹，但想优化新策略 pi_new。Importance weight 是 pi_new(a|s) / pi_old(a|s)。PPO 裁剪这些 weights 以防止新策略偏离旧策略太远。

Importance sampling 估计量的 variance 取决于 q 与 p 的相似程度。如果 q 与 p 非常不同，少数样本获得巨大权重并主导估计。Self-normalized importance sampling 除以权重之和来缓解这个问题：

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation（蒙特卡洛估计）

Monte Carlo 估计通过随机样本的平均来近似积分。大数定律保证收敛。

```
目标: 估计 I = domain D 上 g(x) dx 的积分

方法：
  1. 从 D 均匀采样 x_1, ..., x_N
  2. I ~ (D 的体积 / N) * sum(g(x_i))

误差: O(1 / sqrt(N))   与维度无关
```

误差率与维度无关。这就是为什么 Monte Carlo 方法在高维中占主导，而基于网格的积分是不可能的。

**估计 pi：**

```
从 [-1, 1] x [-1, 1] 均匀采样 (x, y)
统计有多少落在单位圆内: x^2 + y^2 <= 1
pi ~ 4 * (圆内数量) / (总数量)
```

**估计期望：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    其中 x_i ~ p(x)

样本均值收敛到真实期望。
估计量的 variance = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC)：Metropolis-Hastings

MCMC 构建一个 Markov chain，其 stationary distribution (稳态分布) 是目标分布 p(x)。足够步数后，链上的样本（近似地）是来自 p(x) 的样本。

```
目标: p(x)  （已知 up to a normalizing constant）
提案: q(x'|x)  （给定当前状态如何提议下一状态）

Metropolis-Hastings 算法：
  1. 从某个 x_0 开始
  2. 对于 t = 1, 2, ..., T：
     a. 提议 x' ~ q(x'|x_t)
     b. 计算接受率：
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. 以概率 min(1, alpha) 接受：
        - 如果 u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - 否则: x_{t+1} = x_t
  3. 丢弃前 B 个样本（burn-in）
  4. 返回剩余样本
```

对于对称提案（q(x'|x) = q(x|x')），比率简化为 p(x')/p(x)。这是原始的 Metropolis 算法。

**为什么有效。** 接受规则确保 detailed balance (细致平衡)：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。Detailed balance 意味着 p(x) 是链的 stationary distribution。

**实际考虑：**
- Burn-in：在链达到平衡前丢弃早期样本
- Thinning：每 k 个样本保留一个以减少自相关
- 提案尺度：太小则链移动缓慢（接受率高，探索慢）；太大则大部分提案被拒绝（接受率低，卡在原位）
- 高维中高斯提案的最优接受率约为 0.234

### Gibbs Sampling（吉布斯采样）

Gibbs sampling 是 MCMC 在多变量分布上的特例。它不同时提议所有维度的移动，而是每次更新一个变量，从其条件分布中采样。

```
目标: p(x_1, x_2, ..., x_d)

算法：
  对每个迭代 t：
    采样 x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    采样 x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    采样 x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling 要求你能从每个条件分布 p(x_i | x_{-i}) 中采样。这对许多模型来说很简单：
- 贝叶斯网络：条件分布由图结构得出
- 高斯混合：条件分布是高斯
- Ising 模型：每个自旋的条件仅依赖其邻居

接受率总是 1（每个提案都被接受），因为直接从精确条件分布采样自动满足 detailed balance。

**局限性。** 当变量高度相关时，Gibbs sampling 混合缓慢，因为一次更新一个变量无法穿过分布进行大的对角线移动。

### Temperature Sampling（温度采样，用于 LLM）

语言模型为词汇表中的每个 token 输出 logits z_1, ..., z_V。Softmax 将这些转换为概率。Temperature 在 softmax 前重新缩放 logits：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: 标准 softmax（原始分布）
T -> 0:   argmax（确定性，总是选最高 logit）
T -> inf: 均匀（所有 token 等概率）
T < 1.0: 锐化分布（更自信，多样性更低）
T > 1.0: 压平分布（较不自信，多样性更高）
```

**为什么有效。** 将 logits 除以 T < 1 放大了 logits 之间的差异。如果 z_1 = 2 且 z_2 = 1，除以 T = 0.5 得到 z_1/T = 4 和 z_2/T = 2，使差距更大。Softmax 后，最高 logit 的 token 获得大得多的份额。

**实践中：**
- T = 0.0：贪心解码，最适合事实问答
- T = 0.3-0.7：略具创意，适合代码生成
- T = 0.7-1.0：平衡，适合一般对话
- T = 1.0-1.5：创意写作、头脑风暴
- T > 1.5：越来越随机，很少有用

Temperature 不改变哪些 token 是可能的。它改变分配给每个 token 的概率质量。

### Top-k Sampling

Top-k sampling 将候选集限制为概率最高的 k 个 token，然后重新归一化并从该受限集中采样。

```
算法：
  1. 计算所有 V 个 token 的 softmax 概率
  2. 按概率降序排列 token
  3. 只保留前 k 个 token
  4. 重新归一化: p_i' = p_i / sum(p_j for j in top-k)
  5. 从重新归一化的分布中采样

k = 1:  贪心解码
k = V:  无过滤（标准采样）
k = 40: 典型设置，移除不太可能 token 的长尾
```

Top-k 防止模型选择概率极低的 token（错别字、无意义词），这些存在于词汇分布的长尾中。问题：k 是固定的，与上下文无关。当模型很自信（一个 token 有 95% 概率），k = 40 仍允许 39 个替代。当模型不确定（概率分散在 1000 个 token 上），k = 40 切掉了合理的选项。

### Top-p (Nucleus) Sampling

Top-p sampling 动态调整候选集大小。不是保留固定数量的 token，而是保留累积概率超过 p 的最小 token 集。

```
算法：
  1. 计算所有 V 个 token 的 softmax 概率
  2. 按概率降序排列 token
  3. 找到最小的 k 使得前 k 个概率之和 >= p
  4. 只保留这 k 个 token
  5. 重新归一化并采样

p = 0.9:  保留覆盖 90% 概率质量的 token
p = 1.0:  无过滤
p = 0.1:  非常严格，接近贪心
```

当模型自信时，nucleus sampling 保留少数 token（也许 2-3 个）。当模型不确定时，保留很多（也许 200 个）。这种自适应行为是为什么 nucleus sampling 通常比 top-k 产生更好文本的原因。

**常见组合：**
- Temperature 0.7 + top-p 0.9：好的通用设置
- Temperature 0.0（贪心）：适合确定性任务
- Temperature 1.0 + top-k 50：Fan et al. (2018) 原始论文设置

Top-k 和 top-p 可以组合。先应用 top-k，然后在剩余集合上应用 top-p。

### Reparameterization Trick（重参数化技巧，用于 VAE）

VAE (Variational Autoencoder / 变分自编码器) 通过学习将输入编码到隐空间分布，从该分布中采样，再解码样本来回学习。问题：你无法通过采样操作反向传播。

```
标准采样（不可微）：
  z ~ N(mu, sigma^2)

  随机性阻断梯度流。
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick 将随机性与参数分离：

```
重参数化采样：
  epsilon ~ N(0, 1)          （固定随机噪声，无参数）
  z = mu + sigma * epsilon   （参数的确定性函数）

  现在 z 是 mu 和 sigma 的确定性、可微函数。
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  梯度流过 mu 和 sigma。
```

这有效是因为 N(mu, sigma^2) 与 mu + sigma * N(0, 1) 同分布。关键洞察：将随机性移到无参数来源（epsilon），然后将样本表达为参数的可微变换。

**在 VAE 训练循环中：**
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重构输入
5. 反向传播步骤 4, 3, 2, 1（可能，因为步骤 3 可微）

没有 reparameterization trick，VAE 无法使用标准 backpropagation 训练。这一单一洞察使 VAE 变得实用。

### Gumbel-Softmax（可微分类采样）

Reparameterization trick 对连续分布（高斯）有效。对于离散分类分布，我们需要不同的方法。Gumbel-Softmax 提供对分类采样的可微近似。

**Gumbel-Max trick（不可微）：**

```
从具有 log-probabilities log(p_1), ..., log(p_k) 的分类分布中采样：
  1. 为每个类别采样 g_i ~ Gumbel(0, 1)
     （g = -log(-log(u))，其中 u ~ Uniform(0, 1)）
  2. 返回 argmax(log(p_i) + g_i)

这产生精确的分类样本。
```

**Gumbel-Softmax（可微近似）：**

```
用 soft softmax 替换硬 argmax：
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau（温度）控制近似：
  tau -> 0:   趋近 one-hot 向量（硬分类）
  tau -> inf: 趋近均匀 (1/k, 1/k, ..., 1/k)
  tau = 1.0: 软近似
```

Gumbel-Softmax 产生离散样本的连续松弛。输出是概率向量（软 one-hot）而非硬 one-hot。梯度流过 softmax。训练中的前向传播时，你可以使用 "straight-through" 估计器：前向使用硬 argmax，但反向使用软 Gumbel-Softmax 梯度。

**应用：**
- VAE 中的离散隐变量
- 神经架构搜索（选择离散操作）
- Hard attention 机制
- 具有离散动作的 Reinforcement learning

### Stratified Sampling（分层采样）

标准 Monte Carlo 采样可能偶然在样本空间中留下空隙。Stratified sampling 通过将空间分成层并从每层采样来强制均匀覆盖。

```
标准 Monte Carlo：
  从 [0, 1] 均匀采样 N 个点
  某些区域可能有簇，其他有空隙

Stratified sampling：
  将 [0, 1] 分成 N 个等层: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  从每层均匀采样一个点
  x_i = (i + u_i) / N   其中 u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling 的 variance 总是小于或等于标准 Monte Carlo：

```
Var(stratified) <= Var(standard Monte Carlo)

当 f(x) 平滑变化时，改进最大。
对分段常函数，stratified sampling 是精确的。
```

**应用：**
- 数值积分（quasi-Monte Carlo）
- 训练数据划分（确保每折类别平衡）
- 与 stratification 结合的 importance sampling（结合两种技术）
- NeRF (Neural Radiance Fields) 沿相机光线使用 stratified sampling

### 与 Diffusion Models 的联系

Diffusion models 通过采样过程生成图像。Forward process (前向过程) 在 T 步内向图像添加高斯噪声，直到变成纯噪声。Reverse process (反向过程) 学习去噪，逐步恢复原始图像。

```
Forward process（已知）：
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  其中 epsilon ~ N(0, I)

  经过 T 步: x_T ~ N(0, I)  （纯噪声）

Reverse process（学习得到）：
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  其中 z ~ N(0, I)

  每个去噪步骤都是一个采样步骤。
```

与本课方法的联系：
- 每个去噪步骤使用 reparameterization trick（采样噪声，应用确定性变换）
- 噪声 schedule {alpha_t} 控制一种形式的 temperature annealing
- 训练使用 Monte Carlo 估计来近似 ELBO (evidence lower bound)
- Diffusion models 中的 ancestral sampling 是一个 Markov chain（每步仅依赖当前状态）

整个图像生成过程是迭代采样：从噪声开始，每一步采样一个由学习到的去噪模型条件化的稍微少噪声的版本。

## Build It

### Step 1: 均匀和 inverse CDF 采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成 10,000 个指数样本并验证 mean 为 1/lambda。

### Step 2: 拒绝采样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用 rejection sampling 从截断正态分布中抽取样本。通过样本直方图验证形状。

### Step 3: 重要性采样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀提案估计正态分布下的 E[X^2]。与已知答案（mu^2 + sigma^2）比较。

### Step 4: Monte Carlo 估计 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

从双峰分布（两个高斯混合）中采样。可视化链的轨迹。

### Step 6: Gibbs 采样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7: Temperature 采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示 temperature 如何改变一组 token logits 的输出分布。

### Step 8: Top-k 和 top-p 采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9: Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

演示梯度流过重参数化样本但不流过直接采样。

### Step 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    while u == 0:
        u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低 temperature 如何使输出趋近 one-hot 向量。

完整实现和可视化见 `code/sampling.py`。

## Use It

使用 NumPy 和 SciPy，生产级版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大规模 MCMC，使用专用库：
- PyMC：带 NUTS (adaptive HMC) 的完整贝叶斯建模
- emcee：ensemble MCMC sampler
- NumPyro/JAX：GPU 加速 MCMC

你从零构建了这些。现在你知道库调用在做什么。

## Exercises

1. 为 Cauchy 分布实现 inverse CDF 采样。CDF 是 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本并绘制直方图与真实 PDF 对比。注意厚尾（远离中心的极端值）。

2. 使用 rejection sampling 从 Beta(2, 5) 分布中生成样本，使用 Uniform(0, 1) 提案。绘制接受的样本与真实 Beta PDF 对比。理论接受率是多少？

3. 使用 Monte Carlo 估计 sin(x) 从 0 到 pi 的积分，分别用 1,000、10,000 和 100,000 个样本。比较每层的误差。验证误差按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings 以从 2D 分布 p(x, y) 比例于 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 中采样。绘制样本和链轨迹。实验不同的提案标准差。

5. 构建完整的文本生成演示：给定 10 个词的词汇表及其 logits，使用 (a) 贪心、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成 20 个 token 的序列。比较 5 次运行的输出多样性。

## Key Terms

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Sampling | “抽取随机值” | 按照概率分布生成值。所有生成式 AI 背后的机制 |
| Uniform distribution | “所有等可能” | [a, b] 中每个值的概率密度为 1/(b-a)。所有采样方法的起点 |
| Inverse CDF | “概率变换” | F_inverse(U) 将均匀样本转换为已知 CDF 的任意分布的样本。精确且高效 |
| Rejection sampling | “提议并接受/拒绝” | 从简单提案生成，按 target/proposal 比率的比例概率接受。精确但浪费样本 |
| Importance sampling | “重加权样本” | 通过用 p(x)/q(x) 加权每个样本，使用来自 q(x) 的样本估计 p(x) 下的期望。RL 中 PPO 的核心 |
| Monte Carlo | “随机样本取平均” | 将积分近似为样本平均。误差 O(1/sqrt(N)) 与维度无关 |
| MCMC | “收敛的随机游走” | 构建一个 Markov chain 使其 stationary distribution 是目标分布。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | “上坡接受，偶尔下坡” | 提议移动，基于密度比率接受。Detailed balance 确保收敛到目标分布 |
| Gibbs sampling | “一次一个变量” | 每次从条件分布中更新一个变量，固定其他变量。100% 接受率 |
| Temperature | “信心旋钮” | 在 softmax 前将 logits 除以 T。T<1 锐化（更自信），T>1 压平（更多样） |
| Top-k sampling | “保留 k 个最好的” | 将除概率最高的 k 个 token 外全部置零，重新归一化，采样。固定候选集大小 |
| Nucleus sampling (top-p) | “保留可能的那些” | 保留累积概率超过 p 的最小 token 集。自适应候选集大小 |
| Reparameterization trick | “把随机性移到外面” | 写成 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样可微。VAE 训练的核心 |
| Gumbel-Softmax | “软分类采样” | 使用 Gumbel 噪声 + 带温度的 softmax 对分类采样的可微近似 |
| Stratified sampling | “强制覆盖” | 将样本空间分成层，从每层采样。Variance 总是低于朴素 Monte Carlo |
| Burn-in | “预热期” | 在链达到 stationary distribution 前丢弃的初始 MCMC 样本 |
| Detailed balance | “可逆性条件” | p(x) * T(x->y) = p(y) * T(y->x)。p 是 Markov chain stationary distribution 的充分条件 |
| Diffusion sampling | “迭代去噪” | 从噪声开始并应用学习到的去噪步骤生成数据。每步都是一个条件采样操作 |

## Further Reading

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - 原始 Gumbel-Softmax 论文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p) sampling 论文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 引入 reparameterization trick 的 VAE 论文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 将采样与图像生成联系起来
