# Sequence-to-Sequence Models (序列到序列模型)

> 两个 RNN 假装成翻译器。它们遇到的瓶颈正是 attention 存在的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 分钟

## The Problem (问题)

分类将变长序列映射到单个标签。翻译将变长序列映射为另一个变长序列。输入和输出使用不同的词表，可能是不同的语言，长度也不一定对等。

seq2seq 架构（Sutskever, Vinyals, Le, 2014）用一个刻意简单的配方解决了这个问题。两个 RNN。一个读取源句子并生成固定大小的 context vector (上下文向量)。另一个读取该向量并逐 token 生成目标句子。与你在第 08 课写的代码相同，只是以不同方式拼接在一起。

这值得研究有两个原因。首先，context-vector bottleneck (上下文向量瓶颈) 是 NLP 中最具教学价值的失败案例。它驱动了 attention 和 transformer 所做的一切。其次，训练方法（teacher forcing (教师强制)、scheduled sampling (计划采样)、推理时的 beam search (束搜索)）仍然适用于包括 LLM 在内的每一个现代生成系统。

## The Concept (概念)

**Encoder (编码器).** 读取源句子的 RNN。它的最终 hidden state (隐藏状态) 就是 **context vector (上下文向量)** —— 对整个输入的固定大小摘要。 supposedly 不会丢失任何信息。

**Decoder (解码器).** 另一个从 context vector 初始化的 RNN。每一步它接收先前生成的 token 作为输入，并在目标词表上产生一个分布。采样或取 argmax 来选择下一个 token。将其反馈回输入。重复直到生成 `<EOS>` token 或达到最大长度。

**Training (训练):** 每个 decoder 步骤的 cross-entropy loss (交叉熵损失)，在序列上求和。通过两个网络进行标准的 backpropagation through time (随时间反向传播)。

**Teacher forcing (教师强制).** 训练期间，decoder 在步骤 `t` 的输入是位置 `t-1` 的 *ground-truth (真实)* token，而不是 decoder 自己之前的预测。这稳定了训练；没有它，早期错误会级联，模型永远无法学习。在 inference (推理) 时，必须使用模型自己的预测，因此始终存在训练/推理分布差距。这个差距称为 **exposure bias (暴露偏差)**。

**The bottleneck (瓶颈).** 编码器学到的关于源句子的一切信息都必须被压缩进那一个 context vector。长句丢失细节。罕见词被模糊。重排（chat noir vs. black cat）必须靠记忆，而非计算。

Attention (lesson 10) 通过让 decoder 查看 *每一个* encoder hidden state，而不只是最后一个，直接修复了这个问题。这就是全部要点。

## Build It (动手实现)

### Step 1: an encoder (编码器)

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状为 `[batch, seq_len, hidden_dim]` —— 每个输入位置一个 hidden state。`hidden` 的形状为 `[1, batch, hidden_dim]` —— 最终步骤。第 08 课说"对 outputs 做池化用于分类"。这里我们将最后一个 hidden state 作为 context vector，并忽略每步的 outputs。

### Step 2: a decoder (解码器)

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder 每次调用一步。输入：一批单 token 和当前 hidden state。输出：下一个 token 的词表 logits 和更新后的 hidden state。

### Step 3: training loop with teacher forcing (使用教师强制的训练循环)

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

两个值得指出的参数。`ignore_index=0` 跳过对 padding token 的损失计算。`teacher_forcing_ratio` 是在每一步使用真实 token 与模型预测的概率。从 1.0（完全 teacher forcing）开始，在训练过程中退火到约 0.5，以缩小 exposure-bias (暴露偏差) 差距。

### Step 4: inference loop (greedy) (推理循环 —— 贪心解码)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding (贪心解码) 在每一步选择概率最高的 token。它可能会偏离：一旦提交了一个 token，就无法撤回。**Beam search (束搜索)** 保持前 `k` 个部分序列活跃，并在最后选择得分最高的完整序列。Beam width (束宽) 3-5 是标准设置。

### Step 5: the bottleneck, demonstrated (演示瓶颈)

在 toy copy task (复制任务) 上训练模型：源 `[a, b, c, d, e]`，目标 `[a, b, c, d, e]`。增加序列长度。观察准确率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个 GRU hidden state 无法无损记忆 40 个 token 的输入。信息存在于每个 encoder 步骤中，但 decoder 只能看到最后一个状态。Attention 直接修复了这一点。

## Use It (使用它)

PyTorch 提供了 `nn.Transformer` 和基于 `nn.LSTM` 的 seq2seq 模板。Hugging Face 的 `transformers` 库提供了完整的 encoder-decoder 模型（BART, T5, mBART, NLLB），在数十亿 token 上训练。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代 encoder-decoder 用 transformer 替代了 RNN。高层结构（encoder、decoder、逐 token 生成）与 2014 年的 seq2seq 论文完全相同。每个块内部的机制不同。

### When to still reach for RNN-based seq2seq (何时仍应使用基于 RNN 的 seq2seq)

对于新项目，几乎从不。特定例外：

- 流式翻译，你需要以有界内存逐 token 消费输入。
- 设备端文本生成，transformer 的内存成本过高。
- 教学。理解 encoder-decoder bottleneck 是理解 transformer 为何获胜的最快路径。

### Exposure bias and its mitigations (暴露偏差及其缓解方法)

- **Scheduled sampling (计划采样).** 在训练期间退火 teacher forcing ratio，使模型学会从自己的错误中恢复。
- **Minimum risk training (最小风险训练).** 在句子级 BLEU 分数上训练，而不是 token 级 cross-entropy。更接近你真正想要的。
- **Reinforcement learning fine-tuning (强化学习微调).** 用指标奖励序列生成器。现代 LLM RLHF 中使用。

这三种方法仍然适用于基于 transformer 的生成。

## Ship It (交付)

Save as `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Exercises (练习)

1. **Easy.** 实现 toy copy task。在输入-输出对（目标等于源）上训练 GRU seq2seq。测量长度 5、10、20 时的准确率。复现瓶颈现象。
2. **Medium.** 添加 beam width 为 3 的 beam search 解码。在小型平行语料库上测量与 greedy 相比的 BLEU。记录 beam search 获胜的地方（通常是最后几个 token）以及没有区别的地方。
3. **Hard.** 在 10k 对 paraphrase 数据集上 fine-tune (微调) `facebook/bart-base`。比较 fine-tuned 模型的 beam-4 输出与基础模型在 held-out 输入上的结果。报告 BLEU 并挑选 10 个定性示例。

## Key Terms (关键术语)

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Encoder (编码器) | Input RNN | 读取源句子。生成每步 hidden states 和最终的 context vector。 |
| Decoder (解码器) | Output RNN | 从 context vector 初始化。逐 token 生成目标 token。 |
| Context vector (上下文向量) | The summary | 最终 encoder hidden state。固定大小。Attention 解决的瓶颈。 |
| Teacher forcing (教师强制) | Use true tokens | 训练时传入 ground-truth 的前一个 token。稳定学习。 |
| Exposure bias (暴露偏差) | Train/test gap | 在真实 token 上训练的模型从未练习过从自己的错误中恢复。 |
| Beam search (束搜索) | Better decoding | 每一步保持前 k 个部分序列活跃，而不是贪心提交。 |

## Further Reading (延伸阅读)

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) —— 原始 seq2seq 论文。四页。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) —— 引入了 GRU 和 encoder-decoder 框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) —— attention 论文。本课后立即阅读。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) —— 可构建的 seq2seq + attention 代码。
