# 用于文本的 CNN 与 RNN

> 卷积学习 n-gram。循环网络记忆。两者都被注意力机制取代。两者在受限硬件上仍然重要。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 分钟

## The Problem

TF-IDF 和 Word2Vec 生成的平面向量忽略了词序。基于它们构建的分类器无法区分 `dog bites man` 和 `man bites dog`。有时词序本身就是信号。

在 transformer 出现之前，有两类架构填补了这一空白。

**用于文本的卷积网络（TextCNN）。** 在词 embedding（词嵌入）序列上应用 1D 卷积。宽度为 3 的滤波器是一个可学习的 trigram 检测器：它跨越三个词并输出一个分数。堆叠不同宽度（2、3、4、5）以检测多尺度模式。通过 max-pooling（最大池化）得到固定大小的表示。扁平、并行、快速。

**循环网络（RNN、LSTM、GRU）。** 逐个处理词，维护一个向前传递信息的隐藏状态。顺序的、带记忆的、支持可变输入长度。从 2014 年到 2017 年主导了序列建模，然后注意力机制出现了。

本课构建两者，然后指出促使注意力机制诞生的那个失败点。

## The Concept

**TextCNN** (Kim, 2014)。词先被嵌入。宽度为 `k` 的 1D 卷积在连续的 `k`-gram embedding 上滑动一个滤波器，生成一个特征图（feature map）。对该特征图进行全局 max-pooling（全局最大池化）以挑选最强激活。将来自多个滤波器宽度的 max-pooled 输出拼接起来。送入分类头。

为什么有效。一个滤波器就是一个可学习的 n-gram。Max-pooling 是位置不变的，因此 "not good" 在评论开头或中间都会触发相同的特征。三种滤波器宽度，每种 100 个滤波器，相当于 300 个可学习的 n-gram 检测器。训练是并行的；没有时间上的顺序依赖。

**RNN。** 在每个时间步 `t`，隐藏状态 `h_t = f(W * x_t + U * h_{t-1} + b)`。跨时间共享 `W`、`U`、`b`。时间 `T` 的隐藏状态是整个前缀的摘要。对于分类，对 `h_1 ... h_T` 进行池化（max、mean 或 last）。

普通 RNN 存在梯度消失（vanishing gradient）问题。**LSTM** 增加了门控机制来决定遗忘什么、存储什么、输出什么，从而在长序列上稳定梯度。**GRU** 将 LSTM 简化为两个门；参数量更少，性能相似。

**双向 RNN（Bidirectional RNNs）** 同时运行一个前向和一个后向 RNN，拼接它们的隐藏状态。每个词的表示都能看到左右两侧的上下文。对标注任务至关重要。

## Build It

### Step 1: PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 将 `[batch, seq_len, embed_dim]` 重塑为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道（channels）。无论输入长度如何，池化后的输出都是固定大小的。

### Step 2: LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对序列进行 max-pool，而不是只取最后一个状态。对于分类，max-pooling 通常优于取最后一个隐藏状态，因为长序列末尾的信息往往会主导最后一个状态。

### Step 3: 梯度消失演示（直觉）

一个没有门控的普通 RNN 无法学习长距离依赖。考虑一个玩具任务：预测序列中是否出现过 token `A`。如果 `A` 在位置 1，而序列长度为 100，那么来自损失的梯度必须反向流经 99 次循环权重的乘法。如果权重小于 1，梯度就会消失；如果大于 1，就会爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# 当 weight=0.9，跨越 100 步时：
#   0.9 ^ 100 ≈ 2.7e-5
# 从第 100 步到第 1 步的梯度实际上为零。
```

LSTM 通过**细胞状态（cell state）**修复了这个问题，该状态以仅含加法交互的方式流经网络（遗忘门对其进行乘法缩放，但梯度仍能沿"高速公路"流动）。GRU 用更少的参数实现了类似效果。两者都能让你在 100+ 步的序列上稳定训练。

### Step 4: 为什么这仍然不够

即使有了 LSTM，三个问题仍然存在。

1. **顺序瓶颈。** 在长度为 1000 的序列上训练 RNN 需要 1000 个串行的前向/反向步骤。无法跨时间并行化。
2. **编码器-解码器中的固定大小上下文向量。** 解码器只能看到编码器的最终隐藏状态，该状态被压缩在整个输入之上。长输入会丢失细节。第 09 课直接讨论这一点。
3. **远距离依赖的准确率天花板。** LSTM 优于普通 RNN，但在传播跨越 200+ 步的特定信息时仍然困难。

注意力机制解决了所有三个问题。Transformer 完全抛弃了循环。第 10 课是转折点。

## Use It

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都是生产就绪的。训练代码是标准的。

Hugging Face 提供预训练 embedding，你可以直接把它们作为输入层接入：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

适用场景检查清单。

- **边缘 / 端侧推理。** 使用 GloVe embedding 的 TextCNN 比 transformer 小 10-100 倍。如果你的部署目标是手机，这就是你的技术栈。
- **流式 / 在线分类。** RNN 逐个处理 token；transformer 需要完整序列。对于实时流入的文本，LSTM 仍然胜出。
- **用于基线的极小模型。** 在新任务上快速迭代。在 CPU 上 5 分钟就能训练一个 TextCNN。
- **数据有限的序列标注。** BiLSTM-CRF（第 06 课）对于 1k-10k 标注句子的 NER 来说，仍然是生产级架构。

其他所有情况都用 transformer。

## Ship It

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: 根据给定约束集选择文本编码器架构。
phase: 5
lesson: 08
---

给定约束（任务、数据量、延迟预算、部署目标、计算预算），输出：

1. 编码器架构：TextCNN、BiLSTM、BiLSTM-CRF、transformer fine-tune（transformer 微调），或"使用预训练 transformer 作为冻结编码器 + 小型分类头"。
2. Embedding 输入：随机初始化、冻结的 GloVe / fastText，或上下文相关的 transformer embedding。
3. 五行训练配方：optimizer、learning rate（学习率）、batch size（批量大小）、epoch（轮次）、regularization（正则化）。
4. 一个监控信号。对于 RNN/CNN 模型：缺少注意力机制意味着它们会遗漏长距离依赖；检查按长度划分的准确率。对于 transformer：如果学习率过高，fine-tuning（微调）会崩溃；检查训练损失。

当数据少于约 500 条标注样本时，拒绝推荐 fine-tuning transformer，除非先证明 TextCNN / BiLSTM 基线已经饱和。标记边缘部署（手机、微控制器、浏览器）需要把架构决策放在一切之前。
```

## Exercises

1. **Easy.** 在一个 3 类玩具数据集上训练 TextCNN（你自己构造数据）。验证滤波器宽度 (2, 3, 4) 在平均 F1 上优于单一宽度 (3)。
2. **Medium.** 为 LSTM 分类器实现 max-pool、mean-pool 和 last-state pooling。在一个小数据集上比较；记录哪种池化胜出并假设原因。
3. **Hard.** 构建一个 BiLSTM-CRF NER 标注器（结合第 06 课和本课）。在 CoNLL-2003 上训练。与第 06 课的纯 CRF 基线以及 BERT fine-tune 比较。报告训练时间、内存和 F1。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| TextCNN | CNN for text | 在词 embedding 上堆叠 1D 卷积并进行全局 max-pool。Kim (2014)。 |
| RNN | Recurrent net | 每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | Gated RNN | 增加输入/遗忘/输出门 + 细胞状态。能在长序列上稳定训练。 |
| GRU | Simpler LSTM | 两个门而不是三个。准确率相似，参数量更少。 |
| Bidirectional | Both directions | 前向 + 后向 RNN 拼接。每个词都能看到其上下文两侧。 |
| Vanishing gradient | Training signal dies | 普通 RNN 中反复乘以小于 1 的权重，使早期步骤的梯度实际上变为零。 |

## Further Reading

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 论文。八页。可读性强。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 论文。出乎意料地清晰。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 让 LSTM 变得人人可懂的那些图。
