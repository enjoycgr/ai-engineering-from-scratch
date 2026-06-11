# 从零实现 Self-Attention（自注意力）

> 注意力机制就像一张查询表，每个词都会问"谁对我重要？"——然后自己学会答案。

**类型：** 动手实现
**语言：** Python
**前置知识：** Phase 3（深度学习核心）、Phase 5 Lesson 10（序列到序列模型）
**时间：** 约 90 分钟

## 学习目标

- 仅使用 NumPy 从零实现缩放点积自注意力（scaled dot-product self-attention），包括 Query/Key/Value（查询/键/值）投影和 softmax 加权求和
- 构建多头注意力（multi-head attention）层，实现分头（split heads）、并行计算注意力、拼接结果
- 追踪注意力矩阵如何捕捉 token 间的关系，并解释为什么除以 sqrt(d_k) 可以防止 softmax 饱和
- 应用因果掩码（causal masking）将双向注意力转换为自回归（autoregressive，解码器风格）注意力

## 问题背景

RNN（循环神经网络）逐个处理序列中的 token。当你处理到第 50 个 token 时，来自第 1 个 token 的信息已经被压缩了 50 次。长距离依赖被挤压到一个固定大小的隐藏状态中——这是 LSTM 门控机制也无法完全解决的瓶颈。

2014 年 Bahdanau 的注意力论文给出了解决方案：让解码器（decoder）回顾编码器（encoder）的每个位置，并决定哪些位置对当前步骤重要。但它仍然依附于 RNN。2017 年的 "Attention Is All You Need" 论文提出了一个更尖锐的问题：如果注意力是*唯一*的机制呢？没有循环，没有卷积，只有注意力。

自注意力（self-attention）让序列中的每个位置都能在一个并行步骤中关注所有其他位置。这正是 Transformer 快速、可扩展且占据主导地位的原因。

## 核心概念

### 数据库查询类比

将注意力机制想象为一种"软"数据库查询：

```
传统数据库：
  查询（Query）: "法国的首都"  -->  精确匹配  -->  "巴黎"

注意力机制：
  查询（Query）: "法国的首都"  -->  与所有键（Key）计算相似度  -->  所有值（Value）的加权混合
```

每个 token 生成三个向量：
- **Query (Q，查询)**："我在寻找什么？"
- **Key (K，键)**："我包含什么？"
- **Value (V，值)**："如果我被选中，我提供什么信息？"

查询（Query）与所有键（Key）的点积产生注意力分数（attention scores）。高分意味着"这个键匹配我的查询"。这些分数用于加权值（Value）。输出是所有值的加权和。

### Q、K、V 的计算

每个 token 的嵌入（embedding）通过三个可学习的权重矩阵进行投影：

```
输入嵌入（n 个 token 的序列，每个 d 维）：

  X = [x1, x2, x3, ..., xn]       形状: (n, d)

三个权重矩阵：

  Wq  形状: (d, dk)
  Wk  形状: (d, dk)
  Wv  形状: (d, dv)

投影计算：

  Q = X @ Wq    形状: (n, dk)      每个 token 的查询
  K = X @ Wk    形状: (n, dk)      每个 token 的键
  V = X @ Wv    形状: (n, dv)      每个 token 的值
```

直观地看，对于单个 token：

```
             Wq
  x_i ------[*]------> q_i    "我在寻找什么？"
       |
       |     Wk
       +----[*]------> k_i    "我包含什么？"
       |
       |     Wv
       +----[*]------> v_i    "我提供什么？"
```

### 注意力矩阵

一旦获得所有 token 的 Q、K、V，注意力分数就形成一个矩阵：

```
Scores = Q @ K^T    形状: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- q1 对每个键的注意力程度
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

每一行：一个 token 对整个序列的注意力分布
```

逐行观察：每个查询扫过所有键，每行对所有 token 打分，softmax 将分数转为权重，上下文向量（context vector）就是值的加权混合。

```figure
attention-matrix
```

### 为什么要缩放？

点积的数值随维度 dk 增长。如果 dk = 64，点积可能达到几十，将 softmax 推入梯度消失的区域。解决办法：除以 sqrt(dk)。

```
缩放后的分数 = (Q @ K^T) / sqrt(dk)
```

这能将数值保持在 softmax 产生有效梯度的范围内。

### Softmax 将分数转为权重

Softmax 将原始分数转换为每行的概率分布：

```
q1 的原始分数:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
注意力权重:   [0.52, 0.09, 0.07, 0.14, 0.08]   (总和约 1.0)
```

现在每个 token 都有一组权重，表示它对其他每个 token 的关注程度。

### 值的加权求和

每个 token 的最终输出是所有值向量的加权和：

```
output_i = sum( attention_weight[i][j] * v_j  对所有 j )

对于 token 1：
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 完整流程

```mermaid
flowchart LR
  X["X (输入)"] --> Q["Q = X · Wq"]
  X --> K["K = X · Wk"]
  X --> V["V = X · Wv"]
  Q --> S["Q · Kᵀ / √dk"]
  K --> S
  S --> SM["softmax"]
  SM --> WS["加权求和"]
  V --> WS
  WS --> O["输出"]
```

一行公式：

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 动手实现

### 步骤 1：从零实现 Softmax

Softmax 将原始对数几率（logits）转换为概率。为数值稳定性，先减去最大值。

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 步骤 2：缩放点积注意力

核心函数。接收 Q、K、V 矩阵，返回注意力输出和权重矩阵。

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 步骤 3：带可学习投影的 Self-Attention 类

完整的自注意力模块，包含 Wq、Wk、Wv 权重矩阵，使用类似 Xavier 的方式初始化。

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 步骤 4：在句子上运行

为一个句子创建随机嵌入，观察注意力权重。

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 步骤 5：用 ASCII 热力图可视化注意力

将注意力权重映射为字符，快速可视化。

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 实际应用

PyTorch 的 `nn.MultiheadAttention` 正是我们刚才构建的内容，外加多头拆分和输出投影：

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

关键区别：多头注意力并行运行多个注意力函数，每个都有自己的 Q、K、V 投影，维度为 dk = d_model / n_heads，然后拼接结果。这让模型能同时关注不同类型的关系。

## 产出物

本节课产出：
- `outputs/prompt-attention-explainer.md` — 一个通过数据库查询类比来解释注意力的 prompt

## 练习题

1. 修改 `scaled_dot_product_attention`，使其接受一个可选的掩码矩阵，在 softmax 之前将某些位置设为负无穷（这就是因果/解码器掩码的工作原理）
2. 从零实现多头注意力：将 Q、K、V 拆分为 `n_heads` 份，对每个头分别运行注意力，拼接结果，并通过最终的权重矩阵 Wo 投影
3. 取两个相同长度的不同句子，用同一个 SelfAttention 实例处理它们，比较注意力模式。什么变了？什么没变？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Query (Q，查询) | "问题向量" | 输入的可学习投影，表示这个 token 正在寻找什么信息 |
| Key (K，键) | "标签向量" | 输入的可学习投影，表示这个 token 包含什么信息，用于与查询匹配 |
| Value (V，值) | "内容向量" | 输入的可学习投影，携带实际信息，根据注意力分数进行聚合 |
| Scaled dot-product attention（缩放点积注意力） | "注意力公式" | softmax(QK^T / sqrt(dk)) @ V —— 缩放防止高维下 softmax 饱和 |
| Self-attention（自注意力） | "token 看自己和其他 token" | Q、K、V 都来自同一个序列的注意力，让每个位置都能关注所有其他位置 |
| Attention weights（注意力权重） | "关注程度" | 对位置的概率分布，由缩放点积的 softmax 产生 |
| Multi-head attention（多头注意力） | "并行注意力" | 运行多个具有不同投影的注意力函数，然后拼接结果以获得更丰富的表示 |

## 延伸阅读

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) — 完整架构的最佳可视化教程
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 逐行 PyTorch 实现与讲解
