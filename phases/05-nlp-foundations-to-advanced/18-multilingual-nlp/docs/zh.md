# Multilingual NLP

> 一个模型，100 多种语言，其中大多数没有训练数据。跨语言迁移 (cross-lingual transfer) 是 2020 年代的实用奇迹。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 分钟

## The Problem

English 有数十亿标注样本。Urdu 有数千。Maithili 几乎没有。任何面向全球受众的实用 NLP 系统都必须能在长尾语言上工作，而这些语言往往没有任务特定的训练数据。

多语言模型 (multilingual models) 通过同时在多种语言上训练一个模型来解决这个问题。共享的表示 (shared representation) 让模型能够将高资源语言学到的技能迁移到低资源语言。在 English 情感分析上微调 (fine-tune) 模型，它就能在 Urdu 上产生 surprisingly good 的情感预测。这就是 zero-shot cross-lingual transfer，它重塑了 NLP 向世界交付的方式。

本课介绍 tradeoffs、canonical models，以及让刚接触多语言工作的团队绊倒的一个决定：为迁移选择源语言 (source language)。

## The Concept

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**共享词表 (Shared vocabulary)。** 多语言模型使用在所有目标语言文本上训练的 SentencePiece 或 WordPiece tokenizer。词表是共享的：相同的子词单元在相关语言中表示相同的词素 (morpheme)。English 和 Italian 中的 `anti-` 获得相同的 token。

**共享表示 (Shared representation)。** 在多种语言上进行掩码语言建模 (masked language modeling) 预训练的 transformer 学习到，不同语言中语义相似的句子产生相似的隐藏状态 (hidden states)。mBERT、XLM-R 和 NLLB 都表现出这一点。English 中 "cat" 的 embedding 聚集在 French 中 "chat" 和 Spanish 中 "gato" 附近，完整句子的 embedding 也是如此。

**Zero-shot transfer。** 在一种语言（通常是 English）的标注数据上微调 (fine-tune) 模型。在 inference 时，在模型支持的任何其他语言上运行它。不需要目标语言标注。对于类型学上相关的语言结果较强，对于差异大的语言结果较弱。

**Few-shot fine-tuning。** 添加 100-500 条目标语言的标注样本。分类任务上的准确率跃升至 English 基线的 95-98%。这是多语言 NLP 中性价比最高的单一杠杆。

## The models

| Model | Year | Coverage | Notes |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | 在 Wikipedia 上训练。第一个实用的多语言语言模型 (multilingual LM)。低资源语言上较弱。 |
| XLM-R | 2019 | 100 languages | 在 CommonCrawl 上训练（比 Wikipedia 大得多）。设定了跨语言基线 (cross-lingual baseline)。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 languages | XLM-R 配备 1M-token 词表（对比 250k）。低资源语言上更好。 |
| mT5 | 2020 | 101 languages | 用于多语言生成的 T5 架构。 |
| NLLB-200 | 2022 | 200 languages | Meta 的翻译模型；包含 55 种低资源语言。 |
| BLOOM | 2022 | 46 languages + 13 programming | 多语言训练的开放 176B LLM。 |
| Aya-23 | 2024 | 23 languages | Cohere 的多语言 LLM。在 Arabic、Hindi、Swahili 上表现强劲。 |

按用例选择。分类任务使用 XLM-R-base 作为合理的默认选择。生成任务根据翻译还是开放生成选择 mT5 或 NLLB。LLM 风格的工作搭配 Aya-23 或 Claude，使用显式的多语言提示词。

## The source-language decision (2026 research)

大多数团队默认使用 English 作为微调 (fine-tuning) 源语言。最近的研究（2026）表明这通常是错误的。

语言相似性比原始语料库大小更能预测迁移质量。对于 Slavic 目标语言，German 或 Russian 通常胜过 English。对于 Indic 目标语言，Hindi 通常胜过 English。**qWALS** 相似性指标（2026，基于 World Atlas of Language Structures 特征）量化了这一点。**LANGRANK**（Lin et al., ACL 2019）是一个独立的早期方法，根据语言相似性、语料库大小和遗传相关性对候选源语言进行排名。

实用规则：如果你的目标语言有一个类型学上接近的高资源亲属语言，先尝试在该语言上微调，然后与 English 微调进行比较。

## Build It

### Step 1: zero-shot cross-lingual classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，相同的 API。在 NLI 数据上训练的 XLM-R 通过 entailment trick 很好地迁移到分类任务。

### Step 2: multilingual embedding space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻译在 embedding space 中距离很近。不同的 English 句子距离更远。这就是跨语言检索 (cross-lingual retrieval)、聚类 (clustering) 和相似度计算能够工作的原因。

### Step 3: few-shot fine-tuning strategy

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100-500 条目标语言样本，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的默认值。更高的学习率会导致多语言对齐 (multilingual alignment) 崩溃，你会得到一个 English-only 的模型。

## Evaluation that actually works

- **在留出集上的 per-language accuracy。** 不要聚合。聚合会隐藏长尾问题。
- **与 monolingual baseline 对比。** 对于数据足够的语言，从头训练的 monolingual model 有时能击败多语言模型。测试一下。
- **Entity-level tests。** 目标语言中的命名实体。多语言模型对远离 Latin 的脚本通常有弱的 tokenization。
- **Cross-lingual consistency。** 两种语言中的相同含义应该产生相同的预测。测量这个差距。

## Use It

2026 年的技术栈：

| Task | Recommended |
|-----|-------------|
| Classification, 100 languages | XLM-R-base (~270M) fine-tuned |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Translation, 200 languages | `facebook/nllb-200-distilled-600M` (see lesson 11) |
| Generative multilingual | Claude, GPT-4, Aya-23, mT5-XXL |
| Low-resource language NLP | XLM-V 或在相关高资源语言上的领域特定微调 |

如果性能重要，始终为目标语言的微调预留预算。Zero-shot 只是一个起点，不是最终答案。

### The tokenization tax（低资源语言的问题所在）

多语言模型在所有语言之间共享一个 tokenizer。该词表在由 English、French、Spanish、Chinese、German 主导的语料库上训练。对于主导集之外的任何语言，三种 tax 悄然叠加：

- **Fertility tax。** 低资源语言文本 tokenize 成比 English 多得多的每词 token。一句 Hindi 句子可能需要 equivalent English 句子的 3-5 倍 token。这 3-5 倍消耗了你的 context window、训练效率和延迟。
- **Variant recovery tax。** 每个拼写错误、变音符号变体、Unicode 规范化不匹配或大小写变化在 embedding space 中都成为冷启动的无关序列。模型无法学习母语者认为显而易见的正字法对应关系。
- **Capacity spillover tax。** Tax 1 和 2 消耗了 context position、layer depth 和 embedding dimension。留给实际推理的空间系统性地小于高资源语言从同一模型中获得的空间。

实际症状：你的模型在 Hindi 上训练正常，loss curve 看起来正确，eval perplexity 看起来合理，而生产输出却微妙地错误。形态学 (morphology) 在句子中间崩溃。罕见的屈折变化 (inflections) 仍然无法恢复。**你无法通过数据扩展来摆脱一个 broken tokenizer。**

缓解措施：选择对目标语言有良好覆盖的 tokenizer（XLM-V 的 1M-token 词表是直接修复）；在训练前在留出目标文本上验证 tokenization fertility；对真正长尾的脚本使用 byte-level fallback（SentencePiece `byte_fallback=True`，GPT-2 风格的 byte-level BPE），这样永远不会出现 OOV。

## Ship It

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## Exercises

1. **Easy.** 在 English、French、Hindi 和 Arabic 上各运行 10 句 zero-shot classification pipeline。报告每种语言的准确率。你应该会看到 French 很强，Hindi 不错，Arabic 变化较大。
2. **Medium.** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在小型混合语言语料库上构建一个跨语言检索器 (cross-lingual retriever)。用 English 查询，检索任何语言的文档。测量 recall@5。
3. **Hard.** 对 Hindi 分类任务比较 English-source 和 Hindi-source 微调。在两种方案下各使用 500 条目标语言样本进行 few-shot fine-tuning。报告哪种源语言产生更好的 Hindi 准确率以及差距多少。这是 LANGRANK 论点的微缩版。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Multilingual model | One model, many languages | 跨语言共享词表和参数。 |
| Cross-lingual transfer | Train on one language, run on another | 在源语言上微调，在没有目标语言标注的目标语言上评估。 |
| Zero-shot | No target-language labels | 不在目标语言上微调就进行迁移。 |
| Few-shot | Small target labels | 100-500 条目标语言样本用于微调。 |
| mBERT | First multilingual LM | 在 Wikipedia 上预训练的 104-language BERT。 |
| XLM-R | Standard cross-lingual baseline | 在 CommonCrawl 上预训练的 100-language RoBERTa。 |
| NLLB | Meta's 200-language MT | No Language Left Behind。包含 55 种低资源语言。 |

## Further Reading

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 论文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 开启跨语言迁移研究线的分析论文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 论文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多语言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 源语言论文。
