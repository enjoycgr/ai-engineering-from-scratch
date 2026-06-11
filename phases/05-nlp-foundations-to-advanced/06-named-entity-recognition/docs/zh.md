# Named Entity Recognition（命名实体识别）

> 把名字抽出来。听起来简单，直到你遇到有歧义的边界、嵌套实体和领域术语。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 分钟

## The Problem（问题）

"Apple sued Google over its iPhone search deal in the US." 五个实体：Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (也许), US (GPE)。一个好的 NER 系统能提取出所有这些实体并给出正确的类型。一个差的系统会漏掉 iPhone，把 Apple 水果和 Apple 公司搞混，并将 "US" 标记为 PERSON。

NER 是每个结构化抽取 pipeline 背后的主力。简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人回复的 grounding（ grounding）、法律合同抽取。你几乎看不到它；但你总是依赖它。

本节课沿着经典路径（rule-based（基于规则）、HMM、CRF）走向现代路径（BiLSTM-CRF，然后 transformers）。每一步都解决了前一步的特定局限性。这个模式本身就是本节课的教训。

## The Concept（概念）

**BIO tagging（BIO 标注）**（或 BILOU）将实体抽取转化为序列标注问题。为每个 token 标注 `B-TYPE`（实体开始）、`I-TYPE`（实体内部）或 `O`（任何实体之外）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Multi-token entities（多 token 实体）链式组合：`New B-GPE`, `York I-GPE`, `City I-GPE`。理解 BIO 的模型可以抽取任意跨度。

架构演进：

- **Rule-based（基于规则）。** Regex + gazetteer（地名词典）查找。对已知实体 precision（精确率）高，对新实体 coverage（覆盖率）为零。
- **HMM（隐马尔可夫模型）。** Emission probability（发射概率）给定标签的 token 概率，transition probability（转移概率）标签到标签的概率。Viterbi decode（维特比解码）。在标注数据上训练。
- **CRF（条件随机场）。** 像 HMM，但是 discriminative（判别式）的，因此你可以混合任意特征（word shape（词形）、capitalization（大小写）、neighboring words（邻近词））。在 2026 年，它仍然是低资源部署的经典生产主力。
- **BiLSTM-CRF。** 用 neural features（神经特征）代替 hand-crafted features（手工特征）。LSTM 双向读取句子，CRF 层在顶部强制一致的 tag sequences（标签序列）。
- **Transformer-based（基于 Transformer）。** Fine-tune（微调）BERT，使用 token-classification head（token 分类头）。最佳准确率。最高计算成本。

## Build It（动手实现）

### Step 1: BIO tagging helpers（BIO 标注辅助函数）

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Step 2: hand-crafted features（手工特征）

对于经典（非神经）NER，features（特征）就是关键。有用的特征：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 返回 `xXxxxx`。`word_shape("USA-2024")` 返回 `XXX-dddd`。Capitalization patterns（大小写模式）对 proper nouns（专有名词）是高信号的。

### Step 3: a simple rule-based + dictionary baseline（简单的基于规则 + 词典基线）

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级 gazetteers 有数百万条目，从 Wikipedia 和 DBpedia 抓取。Coverage 很好。Disambiguation（消歧）（`Apple` 公司 vs 水果）很糟糕。这就是统计模型获胜的原因。

### Step 4: the CRF step (sketch, not full impl)（CRF 步骤——概述，非完整实现）

没有概率论基础，50 行代码的完整 CRF 并不具有启发性。改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 和 L2 regularization（正则化）。`all_possible_transitions=True` 让模型学习非法序列（例如 `I-ORG` 在 `O` 之后）是不太可能的，这就是 CRF 如何在不写约束的情况下强制 BIO consistency（一致性）。

### Step 5: what a BiLSTM-CRF adds（BiLSTM-CRF 增加了什么）

Features 变成 learned（学习得到）的。Inputs：token embeddings（token 嵌入）（GloVe 或 fastText）。LSTM 从左到右和从右到左读取。Concatenated hidden states（拼接的隐藏状态）经过 CRF output layer（输出层）。CRF 仍然强制 tag-sequence consistency（标签序列一致性）；LSTM 用 learned features 替代 hand-crafted features。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于 CRF 层，使用 `torchcrf.CRF`（pip install pytorch-crf）。相比 hand-crafted CRF 的提升是可测量的，但除非你拥有数万条标注句子，否则比你想象的要小。

## Use It（使用它）

spaCy 开箱即用地提供生产级 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被标记为 `ORG` 而不是 `PRODUCT`——spaCy 的小模型 product-entity coverage 较弱。大模型（`en_core_web_lg`）表现更好。Transformer 模型（`en_core_web_trf`）表现还要更好。

Hugging Face 用于基于 BERT 的 NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 将连续的 B-X、I-X token 合并为一个 span。没有它，你会得到 token-level labels 并需要自己合并。

### LLM-based NER (the 2026 option)（基于 LLM 的 NER——2026 年的选择）

Zero-shot 和 few-shot LLM NER 现在在许多领域与 fine-tuned models 具有竞争力，而且在标注数据稀缺时明显更好。

- **Zero-shot prompting（零样本提示）。** 给 LLM 一个实体类型列表和一个示例 schema。要求 JSON 输出。开箱即用；在新领域上准确率中等。
- **ZeroTuneBio-style prompting。** 将任务分解为 candidate extraction（候选抽取）→ meaning explanation（含义解释）→ judgment（判断）→ re-check（复核）。一个 multi-stage prompt（多阶段提示）（非 one-shot）在 biomedical NER 上大幅提升准确率。同样的模式适用于法律、金融和科学领域。
- **Dynamic prompting with RAG（基于 RAG 的动态提示）。** 为每次 inference call 从一个小的标注种子集中检索最相似的标注示例；动态构建 few-shot prompt。在 2026 年的 benchmark 中，这比 static prompting 将 GPT-4 biomedical NER F1 提升了 11-12%。
- **Per-entity-type decomposition（按实体类型分解）。** 对于长文档，一次调用抽取所有实体类型会随着长度增长而丢失 recall。为每种实体类型运行一次抽取 pass。更高的 inference 成本，大幅更高的准确率。这是 clinical notes（临床笔记）和法律合同的标准模式。

2026 年的生产建议：在收集训练数据之前，先从一个 LLM zero-shot baseline 开始。通常 F1 已经足够好，你根本不需要 fine-tune。

### Where classical NER still wins（经典 NER 仍然获胜的地方）

即使有了 LLM，经典 NER 在以下情况仍然获胜：

- 延迟预算低于 50ms。
- 你有数千个标注示例，需要 98%+ F1。
- 领域有稳定的 ontology（本体），pretrained CRF 或 BiLSTM 能很好地迁移。
- 监管约束要求 on-prem、non-generative model（非生成式模型）。

### Where it falls apart（它崩溃的地方）

- **Domain shift（领域漂移）。** CoNLL 训练的 NER 在法律合同上表现比 gazetteer 还差。在你的领域上 fine-tune。
- **Nested entities（嵌套实体）。** "Bank of America Tower" 同时是 ORG 和 FACILITY。标准 BIO 无法表示重叠的 span。你需要 nested NER（多遍或基于 span 的模型）。
- **Long entities（长实体）。** "United States Federal Deposit Insurance Corporation." Token-level models 有时会拆分它。使用 `aggregation_strategy` 或后处理。
- **Sparse types（稀疏类型）。** 医学 NER 标签如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型完全不知道。Scispacy 和 BioBERT 是那里的起点。

## Ship It（交付）

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Exercises（练习）

1. **Easy.** 实现 `bio_to_spans`（`spans_to_bio` 的逆操作）并在 10 个句子上验证 round-trip consistency（往返一致性）。
2. **Medium.** 在上述 sklearn-crfsuite CRF 上训练 CoNLL-2003 English NER 数据集。使用 `seqeval` 报告 per-entity F1。典型结果：~84 F1。
3. **Hard.** 在 domain-specific NER 数据集（医学、法律或金融）上 fine-tune `distilbert-base-cased`。与 spaCy 小模型对比。记录 data leakage checks（数据泄漏检查）并写下让你惊讶的地方。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NER | Extract names | 用类型（PERSON、ORG、GPE、DATE……）标注 token span。 |
| BIO | Tagging scheme | `B-X` 开始，`I-X` 继续，`O` 在外部。 |
| BILOU | Better BIO | 添加 `L-X`（最后一个）、`U-X`（单个）以获得更清晰的边界。 |
| CRF | Structured classifier | 建模标签之间的 transitions（转移），而不仅仅是 emissions（发射）。强制有效序列。 |
| Nested NER | Overlapping entities | 一个 span 与它的子 span 是不同的实体。BIO 无法表达这个。 |
| Entity-level F1 | Proper NER metric | 预测的 span 必须与真实 span 完全匹配。Token-level F1 会高估准确率。 |

## Further Reading（延伸阅读）

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) —— BiLSTM-CRF 论文。经典之作。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) —— 介绍了成为标准的 token-classification 模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) —— `Doc.ents` 和 `Span` 每个属性的实用参考。
- [seqeval](https://github.com/chakki-works/seqeval) —— 正确的指标库。始终使用它。
