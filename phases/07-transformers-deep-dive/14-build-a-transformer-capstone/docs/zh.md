# 从零构建 Transformer —— 终极项目

> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**前置条件：** 第 7 阶段 · 第 01 至 13 课。不要跳过。
**时间：** 约 120 分钟

## 问题描述

你已经读完了每篇论文。你已经实现了注意力机制、多头拆分、位置编码、编码器和解码器块、BERT 和 GPT 的损失函数、MoE、KV 缓存。现在，让它们在一个真实任务上协同工作。

终极项目：在一个字符级语言建模任务上，端到端训练一个小型的仅解码器（decoder-only）Transformer。它阅读莎士比亚。它生成新的莎士比亚。它足够小，可以在笔记本电脑上不到 10 分钟内完成训练。它足够正确，以至于换上一个更大的数据集并延长训练时间，就能得到一个真正的语言模型（LM）。

这是本课程的 "nanoGPT"。它并非原创 —— Karpathy 2023 年的 nanoGPT 教程是每个学生至少会写一次的参考实现。我们借鉴了其结构，并围绕我们已学的内容进行了重构。

## 核心概念

![从零构建 Transformer 的模块图](../assets/capstone.svg)

架构说明：

```
输入 token (B, N)
   │
   ▼
token embedding + positional embedding  ◀── 第 04 课（RoPE 选项）
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── 第 05 课
│  MultiHeadAttention (causal)      │  ◀── 第 03 课 + 第 07 课（causal mask）
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 第 05 课
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head（与 token embedding 绑定）
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── 第 07 课
```

### 我们提供的代码

- `GPTConfig` —— 统一配置所有超参数的地方。
- `MultiHeadAttention` —— 带有因果掩码（causal mask）的批处理注意力，包含可选的 Flash-style 路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` —— 现代前馈网络（FFN）。
- `Block` —— 预归一化（pre-norm）、残差连接包裹的注意力 + FFN。
- `GPT` —— embedding、堆叠的 block、LM head、generate()。
- 训练循环：AdamW、余弦学习率（cosine LR）、梯度裁剪（gradient clipping）。
- 基于莎士比亚文本的字符级分词器（char-level tokenizer）。

### 我们不提供的代码

- RoPE —— 在第 04 课中进行了概念性实现。这里为了简单起见，我们使用可学习的位置嵌入（learned positional embeddings）。练习部分要求你将其替换为 RoPE。
- 生成过程中的 KV 缓存 —— 每个生成步骤都会重新计算整个前缀的注意力。更慢但更简单。练习部分要求你添加 KV 缓存。
- Flash Attention —— PyTorch 2.0+ 会在输入匹配时自动调度；我们使用 `F.scaled_dot_product_attention`。
- MoE —— 每个 block 只有一个 FFN。你在第 11 课中了解了 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，一个 4 层、4 头、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上训练 2,000 步：

- 训练损失（Training loss）从 ~4.2（随机）收敛到 ~1.5，大约需要 6 分钟。
- 采样输出看起来具有莎士比亚的风格：古旧词汇、换行、类似 "ROMEO:" 的专有名词出现。
- 验证损失（Val loss，文本最后 10% 作为留出集）与训练损失紧密跟踪；在此规模/预算下没有过拟合。

## 构建步骤

本节课使用 PyTorch。安装 `torch`（CPU 版本即可）。参见 `code/main.py`。该脚本处理：

- 如果缺失则下载 `tinyshakespeare.txt`（或读取本地副本）。
- 字节级字符分词器（Byte-level char tokenizer）。
- 90/10 的训练/验证划分。
- 在支持的硬件上使用 bf16 自动混合精度（autocast）的训练循环。
- 训练完成后的采样。

### 第一步：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 个唯一字符。极小的词表。适合 4 字节的 vocab_size。没有 BPE，没有分词器带来的麻烦。

### 第二步：模型

参见 `code/main.py`。Block 是第 05 课的标准结构 —— 预归一化（pre-norm）、RMSNorm、SwiGLU、因果多头注意力（causal MHA）。4/4/128 配置的参数量：约 800K。

### 第三步：训练循环

获取长度为 256 的 token 窗口的随机批次。前向传播。Shift-by-one 交叉熵。反向传播。AdamW 更新。记录日志。重复。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 第四步：采样

给定一个提示（prompt），重复进行前向传播，从 top-p logits 中采样，追加，继续。在 500 个 token 后停止。

### 第五步：查看输出

经过 2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亚。但具有莎士比亚的风格。对于约 800K 参数和笔记本电脑上 6 分钟的训练来说，这是一个明显的胜利。

## 如何使用

这个终极项目是一个参考架构。三个扩展方向，可以将其变为真实可用的模型：

1. **替换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词表大小从 65 跃升至约 50,000。模型容量需要相应扩大以作补偿。
2. **在更大的语料库上训练。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在单个 A100 上，100 亿（10B）token 训练一个 125M 参数的 GPT 大约需要 24 小时。
3. **添加 RoPE + KV 缓存 + Flash Attention。** 下面的练习将带你逐步实现每一个。

最终你会得到一个 1.25 亿参数的 GPT，能够生成流利的英文。不是前沿模型。但同样的代码路径 —— 只是规模更大 —— 正是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用于训练研究检查点（research checkpoints）的代码。

## 交付

参见 `outputs/skill-transformer-review.md`。该 skill 会审查一个从零实现的 Transformer，检查其在之前所有 13 节课中的正确性。

## 练习

1. **简单。** 运行 `code/main.py`。验证你训练的模型在最后一步的验证损失（validation loss）低于 2.0。将 `max_steps` 从 2,000 改为 5,000 —— 验证损失是否继续改善？
2. **中等。** 将可学习的位置嵌入（learned positional embeddings）替换为 RoPE。在 `MultiHeadAttention` 内部将旋转应用于 Q 和 K。训练并验证验证损失至少一样低。
3. **中等。** 在采样循环中实现 KV 缓存。使用和不使用缓存分别生成 500 个 token。在笔记本电脑上， wall-clock 时间应提高 5–20 倍。
4. **困难。** 为模型添加第二个 head，用于预测下一个之后的 token（MTP —— Multi-Token Prediction，来自 DeepSeek-V3）。联合训练。它有帮助吗？
5. **困难。** 将每个 block 中的单个 FFN 替换为 4 专家的 MoE。路由（Router）+ top-2 路由。观察在匹配激活参数（matched active parameters）的情况下验证损失如何变化。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| nanoGPT | "Karpathy 的教程仓库" | 最小的仅解码器 Transformer 训练代码，约 300 行；标准参考实现。 |
| tinyshakespeare | "标准的玩具语料库" | 约 1.1 MB 的文本；自 2015 年以来每个字符级语言模型教程都在使用它。 |
| Tied embeddings | "共享输入/输出矩阵" | LM head 的权重 = token embedding 矩阵的转置；节省参数，提升质量。 |
| bf16 autocast | "训练精度技巧" | 前向/反向传播使用 bf16，优化器状态保持 fp32；自 2021 年以来的标准做法。 |
| Gradient clipping | "阻止梯度爆炸" | 将全局梯度范数上限设为 1.0；防止训练崩溃。 |
| Cosine LR schedule | "2020 年后的默认方案" | 学习率线性上升（warmup），然后以余弦形状衰减至峰值的 10%。 |
| MFU | "模型浮点运算利用率（Model FLOP Utilization）" | 实际达到的 FLOPs / 理论峰值；2026 年，40%（稠密模型）、30%（MoE）是不错的成绩。 |
| Val loss | "留出集损失（Held-out loss）" | 模型从未见过的数据上的交叉熵；过拟合检测器。 |

## 延伸阅读

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典的带注释实现。
