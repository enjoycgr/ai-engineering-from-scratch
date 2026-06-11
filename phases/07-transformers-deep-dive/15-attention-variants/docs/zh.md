# Attention Variants（注意力变体）—— 滑动窗口、稀疏、差分

> 完整的注意力（Full attention）是一个圆。每个 token 都能看到所有其他 token，而内存为此付出代价。四种变体弯曲了圆的形状，并回收了一半成本。

**类型：** Build（动手构建）
**语言：** Python
**前置知识：** Phase 7 · 02（Self-Attention，自注意力），Phase 7 · 03（Multi-Head，多头注意力），Phase 7 · 12（KV Cache / Flash Attention）
**时间：** ~60 分钟

## The Problem（问题）

完整的注意力在序列长度上的内存和计算成本均为 `O(N²)`。对于支持 128K 上下文的 Llama 3 70B，每层有 160 亿个 attention 条目，乘以 80 层。Flash Attention（第 12 课）隐藏了 `O(N²)` 的激活内存，但并未改变算术成本——每个 token 仍然要关注所有其他 token。

三类变体改变了注意力矩阵本身的拓扑结构：

1. **Sliding window attention（SWA，滑动窗口注意力）。** 每个 token 只关注固定窗口内的邻居，而非完整前缀。内存和计算量降至 `O(N · W)`，其中 `W` 为窗口大小。Gemma 2/3、Mistral 7B 的前几层、Phi-3-Long 均使用了 SWA。
2. **Sparse / block attention（稀疏 / 块注意力）。** 只有选定的 `(i, j)` 对会被计算分数；其余被强制设为零权重。Longformer、BigBird、OpenAI sparse transformer 属于此类。
3. **Differential attention（差分注意力）。** 使用独立的 Q/K 投影计算两个注意力图，然后相减。消除“attention sink（注意力汇聚）”导致的权重泄漏到前几个 token 的问题。Microsoft 的 DIFF Transformer（2024）。

这些变体可以共存。2026 年的前沿模型通常混合使用它们：大多数层使用 SWA-1024，每五层使用全局完整注意力，少数层使用差分头来清理检索效果。Gemma 3 的 5:1 SWA-to-global 比例是当前教科书级的默认配置。

## The Concept（概念）

### Sliding Window Attention（SWA，滑动窗口注意力）

位置 `i` 处的每个 query 只关注 `[i - W, i]`（因果 SWA）或 `[i - W/2, i + W/2]`（双向）范围内的位置。窗口外的 token 在分数矩阵中获得 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

当 `N = 8192`、`W = 1024` 时，分数矩阵期望的非零行数为 1024 × 8192——减少了 8 倍。

**SWA 会缩小 KV cache。** 每层只需保留最后 `W` 个 token 的 K 和 V。对于 Gemma-3 风格的配置（窗口 1024，上下文 128K），KV cache 缩小 128 倍。

**质量代价。** 仅使用 SWA 的 transformer 在长距离检索上表现挣扎。解决方案：将 SWA 层与完整注意力层交错排列。Gemma 3 使用 5:1 的 SWA:global 比例。Mistral 7B 使用了因果 SWA 堆栈，信息通过重叠窗口“向前流动”——每层将有效感受野扩展 `W`，经过 `L` 层后模型可以回看到 `L × W` 个 token。

### Sparse / Block Attention（稀疏 / 块注意力）

预先选择一个 `N × N` 的稀疏模式。三种经典形状：

- **Local + strided（OpenAI sparse transformer）。** 关注最后 `W` 个 token，以及在此之前每隔 `stride` 个 token 的一个。以 `O(N · sqrt(N))` 的计算量同时捕获局部和远程信息。
- **Longformer / BigBird。** 局部窗口 + 少量全局 token（例如 `[CLS]`），它们关注所有 token 并被所有 token 关注 + 随机稀疏连接。在同等质量下实现 2 倍上下文长度。
- **Native Sparse Attention（DeepSeek，2025）。** 学习哪些 `(Q, K)` 块重要；在 kernel 层面跳过零块。兼容 FlashAttention。

稀疏注意力是一个 kernel 工程的故事。数学很简单（给分数矩阵加 mask）；收益来自于从不将零条目加载到 SRAM 中。FlashAttention-3 和 2026 年的 FlexAttention API 使自定义稀疏模式成为 PyTorch 中的一等公民。

### Differential Attention（DIFF Transformer，2024）

常规注意力存在“attention sink（注意力汇聚）”问题：softmax 强制每行和为 1，因此不想特别关注任何内容的 token 会将权重倾倒到第一个 token（或前几个 token）上。这窃取了本应分配给真实内容的容量。

差分注意力通过计算**两个**注意力图并相减来解决此问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个可学习的标量（通常为 0.5–0.8）。A1 捕获真实内容的权重；A2 捕获 sink 分量。相减操作抵消了 sink，将权重重新分配给相关 token。

报告结果（Microsoft 2024）：perplexity 降低 5–10%，在相同训练长度下有效上下文延长 1.5–2 倍，needle-in-haystack（大海捞针）检索更锐利。

### Variant Comparison（变体对比）

| Variant（变体） | Compute（计算量） | KV cache | Quality vs full（相对完整注意力的质量） | Production use（生产使用） |
|---------|---------|----------|-----------------|----------------|
| Full attention（完整注意力） | `O(N²)` | 每层 `O(N)` | 基线 | 每个模型的默认层 |
| SWA (window 1024)（滑动窗口，1024） | `O(N·W)` | 每层 `O(W)` | -0.1 ppl，配合全局层表现良好 | Gemma 2/3, Phi-3-Long |
| Local + strided sparse（局部 + 步进稀疏） | `O(N·√N)` | 混合 | 与 SWA 相近 | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | `O(N)` 近似 | 混合 | 在 2 倍上下文下匹配完整注意力 | 早期长上下文 BERT |
| Native Sparse (DeepSeek-V3.2)（原生稀疏注意力） | `O(N · active fraction)` | `O(N)` | 在 0.05 ppl 以内 | DeepSeek-V3.2, 2025 |
| Differential（差分注意力） | `O(2·N²)` | `O(2N)` | -5% 至 -10% ppl | DIFF Transformer, 2026 年初的模型 |

## Build It（动手构建）

参见 `code/main.py`。我们实现了一个因果 mask 比较器，在玩具序列上并排展示 full（完整）、SWA（滑动窗口）、local+strided（局部+步进）和 differential（差分）注意力。

### Step 1: full causal mask（完整因果 mask，基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

第 07 课的基线。下三角矩阵；对角线上方权重为零。

### Step 2: sliding window causal mask（滑动窗口因果 mask）

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

一个参数——`window`。当 `window >= n` 时，恢复为完整因果注意力。当 `window = 1` 时，每个 token 只关注自身。

### Step 3: local + strided sparse mask（局部 + 步进稀疏 mask）

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集的局部窗口，加上从序列开头起每隔 `stride` 个 token 的一个。感受野随着额外层数以对数步增长。

### Step 4: differential attention（差分注意力）

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

两次注意力计算，用一个可学习的混合系数相减。在代码中，我们比较了 single（单次）与 differential（差分）的 attention-sink 热力图，观察 sink 的坍缩。

### Step 5: KV cache sizes（KV cache 大小）

在 `N = 131072` 时，打印每种变体每层的 cache 大小。SWA 和稀疏变体降低 10–100 倍。差分注意力翻倍。有意识地支付你的内存账单。

## Use It（实际使用）

2026 年的生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 混合使用 SWA（window=1024）和全局层，比例为 5:1。
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 中的 FlexAttention 接受一个 mask 函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

这会编译为自定义 Triton kernel。对于常见模式，速度在 FlashAttention-3 的 10% 以内，而且 mask 函数是一个 Python callable。

**何时选择每种变体：**

- **Pure full attention（纯完整注意力）** —— 每层上下文不超过 ~16K，或检索质量至关重要时。
- **SWA + global mix（SWA + 全局混合）** —— 长上下文（>32K），训练和推理受内存限制。2026 年 >32K 时的默认选择。
- **Sparse block attention（稀疏块注意力）** —— 自定义 kernel，自定义模式。保留给专门的工作负载（检索、音频）。
- **Differential attention（差分注意力）** —— 任何 attention-sink 污染会造成伤害的工作负载（长上下文 RAG、needle-in-haystack）。

## Ship It（交付）

参见 `outputs/skill-attention-variant-picker.md`。该 skill 根据目标上下文长度、检索需求和训练/推理计算配置，为新模型选择注意力拓扑。

## Exercises（练习）

1. **Easy（简单）。** 运行 `code/main.py`。验证 `window=4` 的 SWA 将每行最后 4 个 token 之外的所有内容置零。验证 `window=n` 在比特级别上复现完整因果注意力。
2. **Medium（中等）。** 在第 07 课 capstone 之上实现 `window=1024` 的因果 SWA。在 tinyshakespeare 上训练 1,000 步。与完整注意力相比，val loss 下降多少？peak memory 降低多少？
3. **Hard（困难）。** 在 capstone 模型中实现 Gemma-3 风格的 5:1 层混合（5 层 SWA，1 层全局）。在相同参数量下，与纯 SWA 和纯全局基线比较 loss、memory 和生成质量。
4. **Hard（困难）。** 实现每个头可学习 `λ` 的差分注意力。在合成检索任务上训练（一根 needle，2,000 个 distractor）。在相同参数量下，与单次注意力基线比较检索准确率。

## Key Terms（关键术语）

| Term（术语） | What people say（人们常说） | What it actually means（实际含义） |
|------|-----------------|-----------------------|
| Sliding window attention (SWA)（滑动窗口注意力） | "Local attention"（局部注意力） | 每个 query 关注其最后 `W` 个 token；KV cache 缩小至 `O(W)`。 |
| Effective receptive field（有效感受野） | "How far back the model sees"（模型能回看多远） | 在 `L` 层 SWA 堆栈中，窗口为 `W`，最多可达 `L × W` 个 token。 |
| Longformer / BigBird | "Local + global + random"（局部 + 全局 + 随机） | 带有少量始终参与注意的全局 token 的稀疏模式；早期长上下文方法。 |
| Native Sparse Attention（原生稀疏注意力） | "DeepSeek's kernel trick"（DeepSeek 的 kernel 技巧） | 学习块级稀疏性；在 kernel 层面跳过零块，同时保持质量。 |
| Differential attention（差分注意力） | "Two maps, one subtracts"（两个图，一个相减） | DIFF Transformer：从第一个注意力图中减去可学习的 `λ` 倍的第二个注意力图，以抵消 attention sink。 |
| Attention sink（注意力汇聚） | "Weight bleeds to token 0"（权重泄漏到 token 0） | Softmax 归一化强制每行和为 1；无信息量的 query 将权重倾倒到位置 0。 |
| FlexAttention | "Mask-as-Python"（用 Python 写 mask） | PyTorch 2.5+ API，将任意 mask 函数编译成 FlashAttention 形状的 kernel。 |
| Layer type mix（层类型混合） | "5:1 SWA-to-global"（5:1 SWA 对全局） | 在堆栈中交错排列稀疏和完整注意力层，以在降低内存的同时保持质量。 |

## Further Reading（延伸阅读）

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) —— 滑动窗口 + 全局 token 的经典论文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) —— 局部 + 全局 + 随机。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) —— OpenAI 的局部+步进模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) —— 1:1 SWA:global 混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) —— 5:1 混合且 window=1024，现为教科书级默认配置。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) —— DIFF Transformer 论文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) —— DeepSeek-V3.2 的学习稀疏性注意力。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) —— Use It 中 mask-as-callable 模式的 API 参考。
