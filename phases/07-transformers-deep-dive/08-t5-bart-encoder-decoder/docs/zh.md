# T5、BART —— 编码器-解码器模型（Encoder-Decoder Models）

> 编码器负责理解，解码器负责生成。把它们组合起来，你就得到了一个为输入→输出任务而生的模型：翻译、摘要、改写、语音转录。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 7 · 06（BERT）、Phase 7 · 07（GPT）
**时间：** 约 45 分钟

## 问题背景

仅解码器（decoder-only）的 GPT 和仅编码器（encoder-only）的 BERT 分别针对不同的目标对 2017 年的原始架构做了裁剪。但许多任务天然就是输入→输出形式的：

- 翻译：英语 → 法语。
- 摘要：5,000 token 的文章 → 200 token 的摘要。
- 语音识别：音频 token → 文本 token。
- 结构化抽取：散文 → JSON。

对于这类任务，编码器-解码器（encoder-decoder）架构是最自然的选择。编码器为源输入生成一个稠密表示（dense representation），解码器在每一步生成都通过交叉注意力（cross-attention）关注该表示。训练时，输出端采用逐位移（shift-by-one）的方式，损失函数与 GPT 相同，只是以编码器输出为条件。

两篇论文定义了现代编码器-解码器范式：

1. **T5**（Raffel 等人，2019）。《Text-to-Text Transfer Transformer》。将每一个 NLP 任务都重新定义为文本输入、文本输出。单一架构、单一词表、单一损失函数。预训练任务是掩码跨度预测（span corruption）：在输入中破坏若干跨度，让解码器在输出中还原它们。
2. **BART**（Lewis 等人，2019）。《Bidirectional and Auto-Regressive Transformer》。去噪自编码器（denoising autoencoder）：以多种方式破坏输入（打乱、掩码、删除、旋转），让解码器重建原始文本。

到了 2026 年，编码器-解码器架构仍在以下场景活跃：

- Whisper（语音 → 文本）。
- Google 的翻译技术栈。
- 某些具有明确“上下文+编辑”结构的代码补全 / 修复模型。
- Flan-T5 及其变体，用于结构化推理任务。

仅解码器模型抢走了聚光灯，但编码器-解码器从未退场。

## 核心概念

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### 前向循环

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

关键在于：编码器对每个输入只运行一次。解码器以自回归（autoregressive）方式运行，但每一步都交叉关注**同一个**编码器输出。对于长输入，缓存编码器输出是一个免费的加速手段。

### T5 预训练 —— 跨度破坏（span corruption）

从输入中随机选取若干跨度（平均长度 3 个 token，共覆盖约 15% 的文本）。每个跨度替换为一个唯一的哨兵（sentinel）token：`<extra_id_0>`、`<extra_id_1>` 等。解码器只输出被破坏的跨度，并带上对应的哨兵前缀：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

相比预测整个序列，这种信号更经济。在 T5 论文的消融实验中，它与 MLM（BERT）和 prefix-LM（UniLM）相比具有竞争力。

### BART 预训练 —— 多噪声去噪（multi-noise denoising）

BART 尝试了五种噪声函数：

1. Token 掩码（token masking）。
2. Token 删除（token deletion）。
3. 文本填充（text infilling）：掩码一个跨度，解码器需推断正确的长度和内容。
4. 句子置换（sentence permutation）。
5. 文档旋转（document rotation）。

将文本填充与句子置换结合时，下游任务表现最佳。解码器始终重建原始完整序列。BART 的输出是整个序列，而不仅仅是被破坏的跨度，因此预训练计算成本高于 T5。

### 推理（Inference）

与 GPT 相同的自回归生成方式。贪心搜索（greedy）、束搜索（beam search）、top-p 采样均适用。翻译和摘要通常采用束搜索（宽度 4–5），因为这类任务的输出分布比聊天更窄。

### 2026 年如何选择各变体

| 任务 | 是否使用编码器-解码器？ | 原因 |
|------|------------------------|------|
| 翻译 | 是，通常使用 | 明确的源序列；固定的输出分布；束搜索效果好 |
| 语音转文本 | 是（Whisper） | 输入模态与输出不同；编码器处理音频特征 |
| 聊天 / 推理 | 否，仅解码器 | 没有持久的“输入”——对话本身就是序列 |
| 代码补全 | 通常不用 | 长上下文下的仅解码器模型更优；如 Qwen 2.5 Coder 等代码模型均为仅解码器 |
| 摘要 | 两者皆可 | BART、PEGASUS 曾优于早期的仅解码器基线；现代仅解码器大语言模型已能匹敌 |
| 结构化抽取 | 两者皆可 | T5 很简洁，因为“文本→文本”可吸收任意输出格式 |

自 2022 年以来的趋势：仅解码器模型逐渐接管了编码器-解码器曾经擅长的任务，原因有三：(a) 经过指令微调的仅解码器大语言模型可通过提示泛化到任何任务；(b) 单一架构更易于扩展；(c) RLHF 假设模型是解码器。编码器-解码器仍保有其优势领域：输入模态不同（语音、图像）或束搜索质量至关重要的场景。

## 动手实现

参见 `code/main.py`。我们在一个小型语料库上实现了 T5 风格的跨度破坏——这是本课最有用的单一环节，因为它在之后的每一个编码器-解码器预训练方案中都会出现。

### 步骤 1：跨度破坏

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目标格式遵循 T5 约定：`<sent0> span0 <sent1> span1 ...`。被破坏的输入将未改动的 token 与跨度位置的哨兵 token 交错排列。

### 步骤 2：验证往返一致性

给定被破坏的输入和目标，重建原始句子。如果你的破坏是可逆的，那么前向传播就是良定义的。这是一个快速的健康检查——真实训练从不这样做，但测试成本很低，能捕获跨度记录中的差一错误（off-by-one bugs）。

### 步骤 3：BART 噪声

五个函数：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。组合其中两个并展示结果。

## 使用现成模型

HuggingFace 参考代码：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的 trick：任务名称直接写入输入文本。同一个模型可以处理数十种任务，因为每个任务都是文本输入、文本输出。到了 2026 年，这一模式已被指令微调的仅解码器模型所泛化，但 T5 是首创者。

## 交付技能

参见 `outputs/skill-seq2seq-picker.md`。该技能根据输入-输出结构、延迟和质量目标，为新任务选择编码器-解码器或仅解码器方案。

## 练习

1. **简单。** 运行 `code/main.py`，对一句 30 token 的话应用跨度破坏，验证将非哨兵源 token 与解码后的目标跨度拼接后能否还原原文。
2. **中等。** 实现 BART 的 `text_infill` 噪声：将随机跨度替换为单个 `<mask>` token，解码器需推断正确的跨度长度及内容。展示一个示例。
3. **困难。** 在微型英语 → 猪拉丁语（pig-Latin）语料库（200 对）上微调 `flan-t5-small`，在留出的 50 对测试集上测量 BLEU。与在相同数据、相同计算量下微调 `Llama-3.2-1B` 进行对比。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| Encoder-decoder（编码器-解码器） | "Seq2seq transformer" | 两个堆叠结构：双向编码器处理输入，带交叉注意力的因果解码器生成输出。 |
| Cross-attention（交叉注意力） | "Where source talks to target" | 解码器的 Q × 编码器的 K/V。编码器信息进入解码器的唯一通道。 |
| Span corruption（跨度破坏） | "T5's pretraining trick" | 将随机跨度替换为哨兵 token；解码器输出这些跨度。 |
| Denoising objective（去噪目标） | "BART's game" | 对输入施加噪声函数，训练解码器重建干净序列。 |
| Sentinel token（哨兵 token） | "The `<extra_id_N>` placeholder" | 特殊 token，用于在源端标记被破坏的跨度，并在目标端重新标记。 |
| Flan | "Instruction-tuned T5" | 在超过 1,800 个任务上微调的 T5；使编码器-解码器在指令跟随方面具备竞争力。 |
| Beam search（束搜索） | "Decoding strategy" | 每步保留 top-k 条部分序列；翻译/摘要的标准策略。 |
| Teacher forcing（强制教学） | "Training-time input" | 训练时向解码器输入真实的上一个输出 token，而非采样得到的 token。 |

## 延伸阅读

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) —— T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) —— BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) —— Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) —— Whisper，2026 年最具代表性的编码器-解码器模型。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) —— 参考实现。
