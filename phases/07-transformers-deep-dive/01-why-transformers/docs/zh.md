# 为什么是 Transformer —— RNN 的问题

> RNN 逐个处理 token，Transformer 一次性处理所有 token。这一单一架构赌注改变了 2017 年后深度学习中的每一条扩展曲线。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 3（深度学习核心）、Phase 5 · 09（序列到序列）、Phase 5 · 10（注意力机制）
**时间：** 约 45 分钟

## 问题所在

2017 年之前，世界上所有最先进的序列模型——语言、翻译、语音——都是循环神经网络（recurrent neural network, RNN）。LSTM 和 GRU 在相当于 ImageNet 级别的翻译基准测试中统治了半个十年。它们是人们唯一可用的工具。

但它们有三个致命弱点。顺序计算意味着你无法沿时间轴并行化：token `t+1` 需要 token `t` 的隐藏状态。一个 1,024 token 的序列意味着在单个 GPU 上需要 1,024 个串行步骤，而 GPU 每个周期可以执行 1,000,000 次浮点运算。训练的实际耗时与序列长度呈线性关系，而硬件却是为并行计算设计的。

梯度消失意味着 50 个 token 之前的信息已经经过了 50 次非线性压缩。门控循环单元（gated recurrent unit, GRU；long short-term memory, LSTM）缓解了挤压，但从未消除它。长程依赖——"我去年夏天在去京都的飞机上读的那本书是……"—— routinely 失败。

固定宽度的隐藏状态意味着编码器在解码器看到任何内容之前，将整个源序列压缩成单个向量。无论源序列是 5 个 token 还是 500 个 token，瓶颈的形状都相同。

2017 年的论文 "Attention Is All You Need" 提出了一个激进的想法：完全抛弃循环。让每个位置并行地关注其他每个位置。用一次大型矩阵乘法代替 1,024 次顺序矩阵乘法。

其结果到 2026 年主导了每一个模态。语言（GPT-5、Claude 4、Llama 4）、视觉（ViT、DINOv2、SAM 3）、音频（Whisper）、生物学（AlphaFold 3）、机器人学（RT-2）。同一个模块，不同的输入。

## 核心概念

![RNN 顺序计算 vs Transformer 并行注意力](../assets/rnn-vs-transformer.svg)

**循环作为瓶颈。** RNN 计算 `h_t = f(h_{t-1}, x_t)`。每一步都依赖于前一步。你无法在 `h_4` 之前计算 `h_5`。在拥有 10,000+ 并行核心的现代 GPU 上，这在长序列上浪费了 99% 的硅片资源。

**注意力作为广播。** Self-attention（自注意力）同时为每一对 `(i, j)` 计算 `output_i = sum_j(a_ij * v_j)`。整个 N×N 注意力矩阵在一次批处理 matmul（矩阵乘法）中填满。没有任何步骤依赖于另一个步骤。GPU 非常喜欢它。

**加速不是常数。** 它是 `O(N)` 串行深度和 `O(1)` 串行深度之间的差异。在实践中，在匹配的硬件上，当 N=512 时，transformer 每个 epoch 的训练速度快 5–10 倍，并且随着序列长度的增加，差距会扩大，直到你遇到 attention 的 `O(N²)` 内存墙（Flash Attention 后来修复了这个问题——见第 12 课）。

**Transformer 的代价。** Attention 内存按 `O(N²)` 缩放。对于 2K 上下文，没问题。对于 128K 上下文，你需要滑动窗口（sliding window）、RoPE（Rotary Positional Embedding，旋转位置编码）外推、Flash Attention 分块或线性注意力变体。循环在时间上和内存上都是 `O(N)`；transformer 用时间换内存，然后通过并行化赢回时间。

**归纳偏置（inductive bias）的转变。** RNN 假设局部性和近因性（locality and recency）。Transformer 不做任何假设——每一对都是注意力的候选。这就是为什么 transformer 需要更多数据才能训练好，但一旦有了足够的数据就能扩展得更远。Chinchilla（2022）将这一点形式化：给定足够的 token，transformer 总是能在同等参数量下击败 RNN。

## 动手实现

这里没有神经网络——我们通过数值模拟核心瓶颈，让你在自己的笔记本电脑上感受到差距。

### 步骤 1：测量串行深度

参见 `code/main.py`。我们构建两个函数。一个将序列编码为加法链（串行，像 RNN）。一个将其编码为并行归约（广播，像 attention）。相同的数学，不同的依赖图。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # 无法并行化：h 依赖于前一个 h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # 每个 x 都是独立的
```

我们对长度高达 100,000 的序列进行计时。RNN 版本是 O(N) 且单 CPU 流水线。即使在纯 Python 中，attention 风格的归约在长度 ≥ 1,000 时也能击败它，因为 Python 的 `sum()` 是用 C 实现的，迭代时没有每步的解释器开销。

### 步骤 2：计算理论运算量

两种算法都执行 N 次加法。差异在于*依赖深度*：在下一步开始之前必须顺序发生多少操作。RNN 深度 = N。Attention 深度 = 树形归约的 log(N)，或并行扫描的 1。决定 GPU 时间的是深度，而不是操作数。

### 步骤 3：长序列上的经验扩展

我们打印一个计时表，使 O(N) 差距可见。在 2026 年的 Mac 笔记本电脑上，1,000 以下的序列太快而无法测量。100,000 的序列显示出清晰的线性扫描。将其扩展到 16,384 token 的 transformer 和等效的 12 层 LSTM，你就会明白为什么 2016 年训练的实际耗时是一个阻碍。

## 何时使用

2026 年仍然选择 RNN 的情况：

| 场景 | 选择 |
|-----------|------|
| 流式推理，逐个 token，恒定内存 | RNN 或 state-space model（SSM，如 Mamba、RWKV） |
| 超长序列（>1M token），attention 内存爆炸 | Linear attention、Mamba 2、Hyena |
| 没有 matmul 加速器的边缘设备 | Depthwise-separable RNN 仍然在 FLOPs/watt 上获胜 |
| 其他情况（训练、批处理推理、上下文高达 128K） | Transformer |

State-space model（SSM，状态空间模型）如 Mamba 本质上是具有结构化参数化的 RNN，使其兼具两者优点：`O(N)` 扫描内存，通过 selective scan（选择性扫描）实现并行训练。它们恢复了 transformer 90% 的质量，并具有更好的长上下文扩展性。2026 年大多数前沿实验室训练混合 SSM+transformer 模型（例如 Jamba、Samba）——循环并未消亡，它是一个组件。

## 交付

参见 `outputs/skill-architecture-picker.md`。该技能根据长度、吞吐量和训练预算约束为新序列问题选择架构。对于超过 1B token 的训练运行，它应该始终拒绝推荐纯 RNN，除非说明了权衡。

## 练习

1. **简单。** 从 `code/main.py` 中获取 `rnn_style`，将标量隐藏状态替换为长度 64 的隐藏状态向量。重新测量。隐藏状态维度的串行开销增长了多少？
2. **中等。** 在纯 Python 中实现 parallel prefix-sum（并行前缀和，Hillis-Steele scan）。验证它在长度 1024 时产生与串行扫描相同的数值输出。计算深度。
3. **困难。** 将 attention 风格的归约移植到 GPU 上的 PyTorch。在序列长度从 64 到 65,536 的范围内对两者进行计时。绘制并解释曲线形状。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Recurrence（循环） | "RNN 是顺序的" | 步骤 `t` 依赖于步骤 `t-1` 的计算，强制沿时间轴串行执行。 |
| Serial depth（串行深度） | "图有多深" | 依赖操作的最长链；即使在无限硬件上也限制实际耗时。 |
| Attention（注意力） | "让 token 互相查看" | 加权求和 `sum_j a_ij v_j`，其中 `a_ij` 来自位置 i 和 j 之间的相似度分数。 |
| Context window（上下文窗口） | "模型能看到多少" | Attention 层可以接收的位置数量；二次内存成本在此缩放。 |
| Inductive bias（归纳偏置） | "架构中内置的假设" | 关于数据样子的先验；CNN 假设平移不变性，RNN 假设近因性。 |
| State-space model（状态空间模型） | "有代数支撑的 RNN" | 通过结构化状态空间矩阵实现并行训练的循环参数化。 |
| Quadratic bottleneck（二次瓶颈） | "为什么上下文这么贵" | Attention 内存 = 序列长度的 `O(N²)`；Flash Attention 隐藏了常数，而不是缩放。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 杀死主流 NLP 中循环的论文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— attention 的诞生地， bolted onto an RNN。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) —— 原始 LSTM 论文，供记录。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) —— 现代对 transformer 的循环回应。
