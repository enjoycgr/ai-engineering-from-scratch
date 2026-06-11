# BERT — Masked Language Modeling (掩码语言建模)

> GPT 预测下一个词。BERT 预测缺失的词。一句话的差别 —— 却造就了半个多世纪以来所有与 embedding (嵌入) 相关的成果。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 分钟

## The Problem (问题背景)

2018 年，每一个 NLP 任务 —— 情感分析、命名实体识别 (NER)、问答 (QA)、文本蕴含 —— 都需要在自己的标注数据上从头训练一个模型。没有一个预训练好的"理解英语"的 checkpoint 可以拿来 fine-tune (微调)。ELMo (2018) 展示了可以用双向 LSTM 预训练 contextual embedding (上下文嵌入)；它有帮助，但泛化能力有限。

BERT (Devlin et al. 2018) 提出了一个问题：如果我们拿一个 transformer encoder (编码器)，用互联网上的每一句话来训练它，并强迫它从两侧的上下文中预测缺失的词，会怎样？然后你只需要在下游任务上 fine-tune (微调) 一个 head (输出层)。参数效率上的突破令人震惊。

结果是：在 18 个月内，BERT 及其变体（RoBERTa、ALBERT、ELECTRA）统治了当时存在的每一个 NLP 排行榜。到 2020 年，地球上每一个搜索引擎、内容审核管道和语义搜索系统的内部都有一个 BERT。

2026 年，encoder-only (仅编码器) 模型仍然是分类、检索和结构化提取的正确工具 —— 它们每 token 的运行速度比 decoder (解码器) 快 5–10 倍，而且它们的 embedding (嵌入) 是每个现代检索系统的骨干。ModernBERT (2024 年 12 月) 将架构推进到了 8K 上下文，使用 Flash Attention + RoPE + GeGLU。

## The Concept (核心概念)

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### The training signal (训练信号)

以这句话为例：`the quick brown fox jumps over the lazy dog`。

随机 mask (掩码) 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型在 mask 的位置预测原始 token。因为 encoder (编码器) 是 bidirectional (双向的)，预测位置 1 的 `[MASK]` 时可以利用位置 2+ 的 `brown fox jumps`。这正是 GPT 做不到的事情。

### The BERT mask rules (BERT 掩码规则)

在被选中用于预测的 15% 的 token 中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为随机 token。
- 10% 保持不变。

为什么不总是用 `[MASK]`？因为 `[MASK]` 在 inference (推理) 时永远不会出现。如果在训练时 100% 的 mask 位置都是 `[MASK]`，就会在 pre-training (预训练) 和 fine-tuning (微调) 之间造成 distribution shift (分布偏移)。10% 的随机替换 + 10% 的保持不变让模型保持"诚实"。

### Next Sentence Prediction (NSP) —— 以及为什么它被弃用

原始 BERT 还训练了 NSP (下一句预测)：给定两个句子 A 和 B，预测 B 是否紧跟在 A 之后。RoBERTa (2019) 通过消融实验表明 NSP 不仅无益，反而有害。现代 encoder (编码器) 都跳过了它。

### What changed in 2026: ModernBERT

2024 年的 ModernBERT 论文用 2026 年的原语重建了模型块：

| Component | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

与 2018 年的架构不同，它是原生支持 Flash Attention 的。在 8K 序列长度下，inference (推理) 速度比 DeBERTa-v3 快 2–3 倍，同时 GLUE 分数更高。

### Use cases that still pick an encoder in 2026 (2026 年仍选择编码器的用例)

| Task | Why encoder beats decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings (检索 / 语义搜索嵌入) | Bidirectional context (双向上下文) = 每个 token 更好的 embedding (嵌入) 质量 |
| Classification (sentiment, intent, toxicity) (分类：情感、意图、毒性) | One forward pass (一次前向传播)；没有 generation (生成) 开销 |
| NER / token labeling (命名实体识别 / token 标注) | Per-position output (逐位置输出)，原生 bidirectional (双向) |
| Zero-shot entailment (NLI) (零样本蕴含) | Classifier head (分类器头) 放在 encoder (编码器) 之上 |
| Reranker for RAG (RAG 重排序器) | Cross-encoder (交叉编码器) 打分，比 LLM reranker 快 10 倍 |

## Build It (动手实现)

### Step 1: masking logic (掩码逻辑)

参见 `code/main.py`。函数 `create_mlm_batch` 接收一个 token ID 列表、vocab size (词表大小) 和 mask probability (掩码概率)。返回 input IDs（应用了 mask 的）和 labels（仅在 mask 的位置有值，其余为 -100 —— PyTorch 的 ignore index (忽略索引) 约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Step 2: run MLM prediction on a tiny corpus (在微型语料上运行 MLM 预测)

在一个包含 20 个词、200 句话的词表上训练一个 2 层 encoder + MLM head。不计算梯度 —— 我们只进行 forward-pass (前向传播) 的 sanity check (合理性检查)。完整训练需要 PyTorch。

### Step 3: compare mask types (对比掩码类型)

展示 three-way rule (三分规则) 如何让模型在没有 `[MASK]` 的情况下仍然可用。在一个未 mask 的句子和一个 mask 的句子上分别预测。两者都应该产生合理的 token distribution (分布)，因为模型在训练时见过这两种模式。

### Step 4: fine-tune head (微调输出头)

在一个玩具情感数据集上，将 MLM head 替换为 classification head (分类头)。只训练 head；encoder (编码器) 是 frozen (冻结) 的。这是每一个 BERT 应用都遵循的模式。

## Use It (如何使用)

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models 就是 fine-tuned (微调过的) BERT。** `sentence-transformers` 模型如 `all-MiniLM-L6-v2` 是用 contrastive loss (对比损失) 训练的 BERT。Encoder (编码器) 是同一个。变的是 loss function (损失函数)。

**Cross-encoder rerankers (交叉编码器重排序器) 也是 fine-tuned BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做 pair-classification (成对分类)。Query 和 doc 之间的 bidirectional attention (双向注意力) 正是 cross-encoder 相比 bi-encoder (双编码器) 具有质量优势的原因。

**When not to pick BERT in 2026 (2026 年什么时候不选 BERT)。** 任何 generative (生成式) 任务。Encoder (编码器) 没有合理的方式来自回归地生成 token。另外：任何参数量低于 1B 的场景，小 decoder (解码器) 可以在保持更高灵活性的同时达到相当的质量（Phi-3-Mini、Qwen2-1.5B）。

## Ship It (交付)

参见 `outputs/skill-bert-finetuner.md`。该 skill 为一个新的分类或提取任务 scope 了一个 BERT fine-tune (微调) 方案（backbone (骨干网络) 选择、head spec (输出层规范)、数据、eval (评估)、stopping (停止条件)）。

## Exercises (练习)

1. **Easy (简单)。** 运行 `code/main.py` 并打印 10,000 个 token 上的 mask distribution (分布)。确认约 15% 被选中，其中约 80% 变为 `[MASK]`。
2. **Medium (中等)。** 实现 whole-word masking (整词掩码)：如果一个词被 tokenize (分词) 成多个 subword (子词)，则一起 mask 所有 subword 或都不 mask。测量这是否能提高 500 句话语料上的 MLM accuracy (准确率)。
3. **Hard (困难)。** 在一个公开数据集的 10,000 句话上训练一个 tiny (2-layer, d=64) BERT。Fine-tune (微调) `[CLS]` token 用于 SST-2 情感分类。与同等参数量的 decoder-only (仅解码器) baseline 对比 —— 谁赢了？

## Key Terms (关键术语)

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" | Training signal (训练信号)：随机将 15% 的 token 替换为 `[MASK]`，预测原始 token。 |
| Bidirectional | "Looks both ways" | Encoder attention (编码器注意力) 没有 causal mask (因果掩码) —— 每个位置都能看到所有其他位置。 |
| `[CLS]` | "The pooler token" | 一个特殊 token，加在每个序列的开头；它的最终 embedding (嵌入) 被用作 sentence-level representation (句子级表示)。 |
| `[SEP]` | "Segment separator" | 分隔成对序列（例如 query/doc、句子 A/B）。 |
| NSP | "Next sentence prediction" | BERT 的第二个 pretraining (预训练) 任务；RoBERTa 证明它无用，2019 年后被弃用。 |
| Fine-tuning | "Adapt to a task" | 保持 encoder (编码器) 大部分 frozen (冻结)；在顶部训练一个小的 head (输出层) 用于下游任务。 |
| Cross-encoder | "A reranker" | 一个 BERT，将 query 和 doc 同时作为输入，输出 relevance score (相关性分数)。 |
| ModernBERT | "2024 refresh" | 用 RoPE、RMSNorm、GeGLU、alternating local/global attention (交替局部/全局注意力)、8K 上下文重建的 Encoder (编码器)。 |

## Further Reading (延伸阅读)

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) —— 原始论文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) —— 如何正确训练 BERT；干掉了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) —— replaced-token detection (替换 token 检测) 在同等计算量下击败 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) —— ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) —— 标准 encoder (编码器) 参考实现。
