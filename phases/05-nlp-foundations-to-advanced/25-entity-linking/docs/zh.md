# 实体链接与消歧 (Entity Linking & Disambiguation)

> NER 找到了 "Paris"。实体链接需要决定：Paris, France？Paris Hilton？Paris, Texas？Paris (the Trojan prince)？如果没有链接，你的知识图谱将始终充满歧义。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 分钟

## 问题定义 (The Problem)

一句话写道："Jordan beat the press." 你的 NER 将 "Jordan" 标注为 PERSON。很好。但到底是 *哪个* Jordan？

- Michael Jordan（篮球）？
- Michael B. Jordan（演员）？
- Michael I. Jordan（伯克利机器学习教授 —— 是的，在 ML 论文中这种混淆真实存在）？
- Jordan（国家）？
- Jordan（希伯来语名字）？

实体链接 (entity linking, EL) 将每个 mention 解析到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或你的领域知识库。两个子任务：

1. **候选生成 (candidate generation)。** 给定 "Jordan"，哪些 KB 条目是合理的？
2. **消歧 (disambiguation)。** 给定上下文，哪个候选是正确的？

两个步骤都是可学习的。都有基准测试。组合管道已经稳定了十年 —— 变化的是消歧器的质量。

## 核心概念 (The Concept)

![实体链接管道：mention → 候选 → 消歧后的实体](../assets/entity-linking.svg)

**候选生成 (candidate generation)。** 给定 mention 的表面形式 ("Jordan")，在别名索引 (alias index) 中查找候选。Wikipedia 别名词典覆盖了大多数命名实体："JFK" → John F. Kennedy, Jacqueline Kennedy, JFK airport, JFK (movie)。典型索引每个 mention 返回 10-30 个候选。

**消歧：三种方法。**

1. **先验 + 上下文 (Milne & Witten, 2008)。** `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、无需训练。
2. **基于嵌入 (ESS / REL / BLINK)。** 编码 mention + 上下文。编码每个候选的描述。选择最大余弦相似度。2020-2024 年的默认方法。
3. **生成式 (GENRE, 2021; 基于 LLM, 2023+)。** 逐 token 解码实体的规范名称。约束在一个有效实体名称的 trie 上，因此输出保证是有效的 KB id。

**端到端 vs 管道 (pipeline)。** 现代模型（ELQ、BLINK、ExtEnD、GENRE）在一个前向传播中运行 NER + 候选生成 + 消歧。管道系统仍在生产中占主导，因为你可以更换组件。

### 两项评估指标

- **Mention recall（候选生成）。** 正确 KB 条目出现在候选列表中的黄金 mention 比例。整个管道的下限。
- **消歧准确率 / F1。** 给定正确候选的情况下，top-1 正确的频率。

两者都要报告。一个在 80% 候选召回率上达到 99% 消歧准确率的系统，其实是一个 80% 的管道。

## 动手实现 (Build It)

### 步骤 1：从 Wikipedia 重定向构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 别名数据：约 1800 万 (alias, entity) 对。从 Wikidata dumps 下载。存储为倒排索引。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重叠只是一个 toy 示例。用嵌入 (embedding) 上的余弦相似度替换（参见 `code/main.py` 步骤 2 的 transformer 版本）。

### 步骤 3：基于嵌入的方法（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引时，每个 KB 实体只嵌入一次。在查询时，mention + 上下文嵌入一次，与候选池做点积，选择最大值。

### 步骤 4：生成式实体链接（概念）

GENRE 逐字符解码实体的 Wikipedia 标题。约束解码 (constrained decoding)（参见第 20 课）确保只能输出有效的标题。与基于 KB 的 trie 紧密集成。现代后继是 REL-GEN 和基于 LLM prompt 的 EL 配合结构化输出。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单（Outlines `choice`），这是在 2026 年部署最简单的 EL 管道。

### 步骤 5：在 AIDA-CoNLL 上评估

AIDA-CoNLL 是标准的 EL 基准测试：1,393 篇 Reuters 文章，34k mentions，Wikipedia 实体。报告 in-KB 准确率 (`P@1`) 和 out-of-KB NIL 检测率。

## 常见陷阱 (Pitfalls)

- **NIL 处理 (NIL handling)。** 有些 mention 不在 KB 中（新兴实体、不知名人物）。系统必须预测 NIL 而不是猜错实体。单独测量。
- **Mention 边界错误 (mention boundary errors)。** 上游 NER 漏掉部分跨度（"Bank of America" 只标注为 "Bank"）。EL 召回率下降。
- **流行度偏差 (popularity bias)。** 训练系统过度预测高频实体。ML 论文中提到 "Michael I. Jordan" 时经常链接到篮球 Jordan。
- **跨语言 EL (cross-lingual EL)。** 将中文文本中的 mention 映射到英文 Wikipedia 实体。需要多语言编码器或翻译步骤。
- **KB 陈旧性 (KB staleness)。** 新公司、事件、人物不在去年的 Wikipedia dump 中。生产管道需要刷新循环。

## 实际应用 (Use It)

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语 + Wikipedia | BLINK 或 REL |
| 跨语言，KB = Wikipedia | mGENRE |
| LLM 友好，少量 mentions/天 | 用候选列表 + 约束 JSON prompt Claude/GPT-4 |
| 领域特定 KB（医疗、法律） | 自定义 BERT + KB 感知检索 + 在领域 AIDA 风格数据集上微调 (fine-tuning) |
| 极低延迟 | 仅精确匹配先验 (Milne-Witten baseline) |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年可投入生产的模式：NER → coref → 每个 mention 的 EL → 将聚类折叠为每个聚类一个规范实体。输出：文档中每个实体一个 KB id，而不是每个 mention 一个。

## 交付物 (Ship It)

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习 (Exercises)

1. **简单。** 在 `code/main.py` 中实现先验+上下文消歧器，处理 10 个歧义 mention（Paris, Jordan, Apple）。手工标注正确实体。测量准确率。
2. **中等。** 用 sentence transformer 编码 50 个歧义 mention。嵌入每个候选的描述。比较基于嵌入的消歧与 Jaccard 上下文重叠。
3. **困难。** 构建一个 1k 实体的领域 KB（例如你公司的员工 + 产品）。实现端到端 NER + EL。在 100 个 held-out 句子上测量精确率和召回率。

## 关键术语 (Key Terms)

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| Entity linking (EL) | 链接到 Wikipedia | 将 mention 映射到唯一的 KB 条目。 |
| Candidate generation | 可能是谁？ | 为 mention 返回合理的 KB 条目短列表。 |
| Disambiguation | 选择正确的一个 | 使用上下文对候选打分，选择胜者。 |
| Alias index | 查找表 | 从表面形式 → 候选实体的映射。 |
| NIL | 不在 KB 中 | 明确预测没有 KB 条目匹配。 |
| KB | 知识库 | Wikidata、Wikipedia、DBpedia 或你的领域 KB。 |
| AIDA-CoNLL | 基准测试 | 1,393 篇带有黄金实体链接的 Reuters 文章。 |

## 延伸阅读 (Further Reading)

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基础性的先验+上下文方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入的主力方法。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 带约束解码的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 基准测试论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源生产栈。
