# Multi-Head Attention (多头注意力)

> 一个注意力头一次学习一种关系。八个头学习八种。头是免费的，多用几个。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 02 (Self-Attention from Scratch，从零实现自注意力)
**时间：** 约 75 分钟

## 问题所在

单个 self-attention (自注意力) 头只计算一个 attention matrix (注意力矩阵)。这个矩阵只能捕捉一种关系——通常是使训练信号损失最小化的那种。如果你的数据同时包含主谓一致、共指消解、长距离语篇和句法分块，单个头会把它们全部混合到一个 softmax 分布中，导致一半信号丢失。

2017 年 Vaswani 论文提出的解决方案：并行运行多个注意力函数，每个都有自己的 Q、K、V 投影，然后将输出拼接起来。每个头在维度为 `d_model / n_heads` 的更小子空间中操作。总参数量保持不变，表达能力却提升了。

multi-head attention (多头注意力) 是 2026 年每个 transformer 的默认配置。唯一的争论在于*多少*个头，以及 keys 和 values 是否共享投影（Grouped-Query Attention (分组查询注意力)、Multi-Query Attention (多查询注意力)、Multi-head Latent Attention (多头潜在注意力)）。

## 核心概念

![Multi-head attention splits, attends, concatenates (多头注意力：拆分、注意力计算、拼接)](../assets/multi-head-attention.svg)

**拆分 (Split)。** 取形状为 `(N, d_model)` 的 `X`。投影为 Q、K、V，每个形状都是 `(N, d_model)`。重塑为 `(N, n_heads, d_head)`，其中 `d_head = d_model / n_heads`。转置为 `(n_heads, N, d_head)`。

**并行计算注意力 (Attend in parallel)。** 在每个头内部运行 scaled dot-product attention (缩放点积注意力)。每个头输出 `(N, d_head)`。各个头在嵌入的不同子空间上操作，在注意力计算过程中互不通信。

**拼接并投影 (Concatenate and project)。** 将头重新堆叠为 `(N, d_model)`，然后乘以一个可学习的输出矩阵 `W_o`，形状为 `(d_model, d_model)`。`W_o` 是头与头之间进行信息混合的地方。

**为什么有效。** 每个头可以专门化，而不必与其他头竞争表示预算。2019–2024 年的探测研究表明了不同的头角色：位置头、关注前一个 token 的头、复制头、命名实体头、induction heads (归纳头，它们构成了 in-context learning (上下文学习) 的基础)。

**2026 年的变体谱系：**

| 变体 | Q 头数 | K/V 头数 | 使用者 |
|---------|---------|-----------|---------|
| Multi-head (MHA，多头注意力) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA，多查询注意力) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA，分组查询注意力) | N | G (例如 N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA，多头潜在注意力) | N | 压缩为低秩 | DeepSeek-V2, V3 |

GQA 是现代默认选择，因为它将 KV-cache (KV 缓存) 内存减少了 `N/G` 倍，同时几乎保持完整质量。MLA 更进一步，将 K/V 压缩到一个 latent space (潜在空间)，然后在计算时投影回来——消耗更多 FLOPs，但节省更多内存。

## 动手构建

### 第一步：从我们已有的单头注意力中拆分出头

取第 02 课的 `SelfAttention`，用 split/concat (拆分/拼接) 对包装它。参见 `code/main.py` 中的 numpy 实现；逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape (重塑) 和一次 transpose (转置)。没有循环。这正是 PyTorch 在 `nn.MultiheadAttention` 底层所做的事情。

### 第二步：每个头运行缩放点积注意力

每个头获得 Q、K、V 的独立切片。注意力变成一个 batched matmul (批量矩阵乘法)：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 就是一个 `bmm` (批量矩阵乘法)。GPU 看到的是单个 batched matmul (批量矩阵乘法)，形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`。增加头是免费的。

### 第三步：Grouped-Query Attention (分组查询注意力) 变体

只有 key 和 value 的投影发生变化。Q 获得 `n_heads` 个组；K 和 V 获得 `n_kv_heads < n_heads` 个组，然后被重复以匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理时这节省了内存，因为只有 `n_kv_heads` 份副本存在于 KV cache (KV 缓存) 中，而不是 `n_heads` 份。Llama 3 70B 使用 64 个 query heads (查询头) 和 8 个 KV heads——缓存缩小了 8 倍。

### 第四步：探测每个头学到了什么

在一个短句子上用 4 个头运行 MHA。对每个头，打印 `(N, N)` 的 attention matrix (注意力矩阵)。即使使用随机初始化，你也会看到不同的头捕捉到不同的结构——这部分是信号，部分是子空间中的旋转对称性。

## 使用它

在 PyTorch 中，一行代码即可：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 中的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention 在 CUDA 上自动分发到 Flash Attention。
# 对于 GQA，传入形状为 (B, n_heads, N, d_head) 的 Q，
# 以及形状为 (B, n_kv_heads, N, d_head) 的 K, V。PyTorch 会自动处理重复。
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**多少个头？** 2026 年生产模型的经验法则：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` 几乎总是 64 或 128。它是一个头能"看到"多少信息的单位。低于 32，头会开始与 scaling factor (缩放因子) `sqrt(d_head)` 冲突；高于 256，你会失去"多个小型专家"的好处。

## 交付它

参见 `outputs/skill-mha-configurator.md`。该 skill 会根据参数预算、序列长度和部署目标，为新 transformer 推荐 head count (头数)、kv-head count (KV 头数) 和 projection strategy (投影策略)。

## 练习

1. **简单。** 取 `code/main.py` 中的 MHA，在 `d_model=64` 固定的情况下将 `n_heads` 从 1 改为 16。绘制一个微小的单层模型在 synthetic copy task (合成复制任务) 上的损失。更多的头有帮助、趋于平稳，还是有害？
2. **中等。** 实现 MQA（一个 KV head 被所有 query heads 共享）。测量与完整 MHA 相比参数量下降了多少。计算在推理时 N=2048 的情况下 KV-cache (KV 缓存) 缩小了多少。
3. **困难。** 实现一个微型版本的 Multi-head Latent Attention (多头潜在注意力)：将 K,V 压缩到秩为 `r` 的 latent (潜在表示)，将 latent 存入 KV cache，在注意力计算时解压。在什么 `r` 值下，缓存内存降到完整 MHA 的 1/8 以下，同时质量保持在验证 ppl 的 1 bit 以内？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Head (头) | "A single attention circuit" (单个注意力电路) | 一个维度为 `d_head = d_model / n_heads` 的 Q/K/V 投影，拥有自己的 attention matrix (注意力矩阵)。 |
| d_head | "Head dimension" (头维度) | 每个头的隐藏宽度；在生产环境中几乎总是 64 或 128。 |
| Split / combine (拆分/拼接) | "Reshape tricks" (重塑技巧) | 围绕注意力计算的 `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose (重塑+转置)。 |
| W_o | "Output projection" (输出投影) | 拼接头之后应用的 `(d_model, d_model)` 矩阵；头是这里混合的。 |
| MQA | "One KV head" (一个 KV 头) | Multi-Query Attention (多查询注意力)：单个共享的 K/V 投影。KV cache (KV 缓存) 最小，有一定质量损失。 |
| GQA | "The default since Llama 2" (Llama 2 以来的默认选择) | Grouped-Query Attention (分组查询注意力)，`n_kv_heads < n_heads`；重复以匹配 Q。 |
| MLA | "DeepSeek's trick" (DeepSeek 的技巧) | Multi-head Latent Attention (多头潜在注意力)：K,V 压缩为低秩 latent (潜在表示)，在注意力计算时解压。 |
| Induction head (归纳头) | "The circuit behind in-context learning" (上下文学习背后的电路) | 一对检测先前出现并复制其后内容的头。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 原始的 multi-head (多头) 规范。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在训练后将 MHA 转换为 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA 以及为什么它在缓存内存上击败 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 从机制角度探究头实际在做什么。
