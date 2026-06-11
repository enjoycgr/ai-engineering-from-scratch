# Attention Mechanism — The Breakthrough (注意力机制 —— 突破性进展)

> 解码器不再眯着眼盯着压缩摘要，而是开始查看整个源序列。此后的一切都是 attention 加工程实现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 分钟

## The Problem (问题)

第 09 课以一个审慎的失败告终。一个 GRU encoder-decoder 在 toy copy task 上，长度 5 时准确率 89%，长度 80 时接近随机。原因是结构性的，不是训练 bug：编码器收集到的每一条信息都必须塞进一个固定大小的 hidden state，而解码器永远看不到其他任何东西。

Bahdanau、Cho 和 Bengio 在 2014 年发表了一个三行修复方案。不再只给解码器最终的 encoder state，而是保留每一个 encoder state。在 decoder 的每一步，计算 encoder states 的加权平均，权重表示"解码器现在需要看 encoder 位置 `i` 多少？"这个加权平均就是 context vector (上下文向量)，并且每个 decoder 步骤都会变化。

这就是全部思想。Transformer 扩展了它。Self-attention (自注意力) 将其应用于单个序列。Multi-head attention (多头注意力) 并行运行它。但 2014 年的版本已经打破了瓶颈，一旦你掌握了它，转向 transformer 主要是工程而非概念上的跨越。

## The Concept (概念)

![Bahdanau attention: decoder queries all encoder states (Bahdanau 注意力：解码器查询所有编码器状态)](../assets/attention.svg)

在每个 decoder 步骤 `t`：

1. 使用前一个 decoder hidden state `s_{t-1}` 作为 **query (查询)**。
2. 将其与每个 encoder hidden state `h_1, ..., h_T` 计算分数。每个 encoder 位置一个标量。
3. 对分数应用 softmax (软最大值) 得到 attention weights (注意力权重) `α_{t,1}, ..., α_{t,T}`，和为 1。
4. Context vector (上下文向量) `c_t = Σ α_{t,i} * h_i`。Encoder states 的加权平均。
5. Decoder 接收 `c_t` 加上前一个输出 token，生成下一个 token。

加权平均是关键。当 decoder 需要将 "Je" 翻译为 "I" 时，它给 encoder 状态中 "Je" 位置的权重高，其他低。当它需要 "not" 时，它给 "pas" 的权重高。Context vector 每一步都会重塑。

## Shapes (the thing that bites everyone) (形状 —— 每个人都会踩的坑)

这是每个 attention 实现第一次都会出错的地方。慢慢读。

| Thing | Shape | Notes |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | 一个向量 |
| Attention score `e_{t,i}` | scalar | 每个 encoder 位置一个 |
| Attention weight `α_{t,i}` | scalar | 对所有 `i` 做 softmax 后 |
| Context vector `c_t` | `(d_h,)` | 与 encoder state 形状相同 |

**Bahdanau (additive) score (Bahdanau 加性分数).** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 形状为 `(d_s,)`，`h_i` 形状为 `(d_h,)`。
- `W_a` 形状为 `(d_attn, d_s)`。`U_a` 形状为 `(d_attn, d_h)`。
- tanh 内部的和形状为 `(d_attn,)`。
- `v_α` 形状为 `(d_attn,)`。与 `v_α` 的内积坍缩为标量。**这就是 `v_α` 的作用。** 它不是魔法。它是将 attention-dim 向量投影为标量分数的投影。

**Luong (multiplicative) score (Luong 乘性分数).** 三种变体：

- `dot`: `e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`。硬约束。如果 encoder 是双向的则跳过。
- `general`: `e_{t,i} = s_t^T * W * h_i`，`W` 形状 `(d_s, d_h)`。移除等维约束。
- `concat`: 本质上是 Bahdanau 形式。由于前两种更便宜，很少使用。

**一个值得指出的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（生成当前词*之前*的 decoder state）。Luong 使用 `s_t`（生成当前词*之后*的状态）。混淆它们会产生极难调试的微妙错误梯度。选择一篇论文并坚持其约定。

## Build It (动手实现)

### Step 1: additive (Bahdanau) attention (加性注意力)

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

对照上面的表格检查形状。`encoder_states` 形状为 `(T_enc, d_h)`。`projected_enc` 形状为 `(T_enc, d_attn)`。`projected_dec` 形状为 `(d_attn,)` 并广播。`combined` 形状为 `(T_enc, d_attn)`。`scores` 形状为 `(T_enc,)`。`weights` 形状为 `(T_enc,)`。`context` 形状为 `(d_h,)`。搞定。

### Step 2: Luong dot and general (Luong 点积与通用注意力)

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每种三行。这就是 Luong 的论文为何能发表的原因。大多数任务上准确率相同，代码量少得多。

### Step 3: a worked numerical example (数值示例)

给定三个 encoder states（大致对应 "cat"、"sat"、"mat"）和一个与第一个对齐最好的 decoder state，attention distribution 集中在位置 0。如果 decoder state 转向与最后一个对齐，attention 移动到位置 2。Context vector 随之跟踪。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行胜出。然后将 decoder state 移近第三个 encoder state 并观察权重变化。就是这样。Attention 是显式的对齐。

### Step 4: why this is the bridge to transformers (为何这是通往 transformer 的桥梁)

将上面的语言翻译为 Q/K/V：

- **Query (查询)** = decoder state `s_{t-1}`
- **Key (键)** = encoder states（我们与之计算分数的对象）
- **Value (值)** = encoder states（我们加权求和的对象）

在经典 attention 中，keys 和 values 是同一个东西。Self-attention (自注意力) 将它们分开：你可以用不同的学习投影对同一个序列做查询，分别得到 K 和 V。Multi-head attention (多头注意力) 用不同的学习投影并行运行。Transformer 将整个过程堆叠多次并丢弃 RNN。

数学是相同的。形状是相同的。从 Bahdanau attention 到 scaled dot-product attention 的教学跳跃主要是符号上的。

## Use It (使用它)

PyTorch 和 TensorFlow 直接内置了 attention。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 transformer attention 层。Query batch 有 5 个位置，key/value batch 有 10 个位置，128 维，8 个 attention head (注意力头)。`output` 是新的 context-augmented queries。`weights` 是可以可视化的 5x10 对齐矩阵。

### When classical attention still matters (经典注意力何时仍然重要)

- **教学。** 单头、单层、基于 RNN 的版本让每个概念都可见。
- **Transformer 放不下的设备端序列任务。**
- **2014-2017 年的任何论文。** 不了解 Bahdanau 的约定你会误读它们。
- **机器翻译中的细粒度对齐分析。** 原始 attention weights 即使在 transformer 模型上也是一种可解释性工具，阅读它们需要知道它们是什么。

### The attention-weight-as-explanation trap (将注意力权重当作解释的陷阱)

Attention weights 看起来可解释。它们是跨位置求和为 1 的权重；你可以绘制它们；高值意味着"关注了这个"。审稿人喜欢它们。

它们并不像看起来那样可解释。Jain 和 Wallace (2019) 表明，在某些任务上，attention distribution 可以被置换并用任意替代替换而不改变模型预测。永远不要在没有消融或反事实检查的情况下将 attention weights 报告为推理的证据。

## Ship It (交付)

Save as `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Exercises (练习)

1. **Easy.** 实现 `softmax` masking (掩码)，使 encoder 中的 padding token 获得零 attention weight。在具有变长序列的 batch 上测试。
2. **Medium.** 为 Luong `general` 形式添加 multi-head attention (多头注意力)。将 `d_h` 分成 `n_heads` 组，每头分别运行 attention，然后拼接。验证单头情况与之前的实现匹配。
3. **Hard.** 在来自第 09 课的 toy copy task 上训练带有 Bahdanau attention 的 GRU encoder-decoder。绘制准确率与序列长度的关系图。与无 attention 的基线比较。你应该看到随着长度增长差距扩大，确认 attention 消除了瓶颈。

## Key Terms (关键术语)

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Attention (注意力) | Looking at things | Value 序列的加权平均，权重由 query-key 相似度计算得出。 |
| Query, Key, Value (查询、键、值) | QKV | 三个投影：Q 提问，K 是匹配对象，V 是返回内容。 |
| Additive attention (加性注意力) | Bahdanau | 前馈分数：`v^T tanh(W q + U k)`。 |
| Multiplicative attention (乘性注意力) | Luong dot / general | 分数为 `q^T k` 或 `q^T W k`。更便宜，大多数任务上准确率相同。 |
| Alignment matrix (对齐矩阵) | The pretty picture | Attention weights 作为 `(T_dec, T_enc)` 网格。阅读它可查看模型关注了什么。 |

## Further Reading (延伸阅读)

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— 原始论文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) —— 三种分数变体及其比较。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) —— 可解释性警告。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) —— 可运行的 PyTorch 逐步教程。
