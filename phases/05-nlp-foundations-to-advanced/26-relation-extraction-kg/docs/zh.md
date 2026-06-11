# 关系抽取与知识图谱构建 (Relation Extraction & Knowledge Graph Construction)

> NER 找到了实体。实体链接将它们锚定。关系抽取找到它们之间的边。知识图谱是节点、边及其来源 (provenance) 的总和。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 分钟

## 问题定义 (The Problem)

分析师读到："Tim Cook became CEO of Apple in 2011." 四个事实：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取 (relation extraction, RE) 将自由文本转换为结构化三元组 `(subject, relation, object)`。跨语料库聚合就得到了知识图谱 (knowledge graph, KG)。聚合并查询后，你就拥有了用于 RAG、分析或合规审计的推理基础。

2026 年的问题：LLM 热情地抽取关系。过于热情了。它们会幻觉 (hallucinate) 出源文本不支持的三元组。没有来源 (provenance)，你无法区分真实三元组和看似合理的虚构。2026 年的答案是 AEVS 风格的锚定-验证管道。

## 核心概念 (The Concept)

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式 (Triple form)。** `(subject_entity, relation_type, object_entity)`。关系来自封闭本体 (closed ontology)（Wikidata 属性、FIBO、UMLS）或开放集合（OpenIE 风格，任意关系）。

**三种抽取方法。**

1. **规则 / 模式匹配。** Hearst 模式："X such as Y" → `(Y, isA, X)`。加上手工正则。脆弱、精确、可解释。
2. **监督分类器。** 给定句子中的两个实体 mention，从固定集合预测关系。在 TACRED、ACE、KBP 上训练。2015–2022 年的标准方法。
3. **生成式 LLM。** 提示模型输出三元组。开箱即用。需要来源 (provenance)，否则会幻觉出看似合理的垃圾。

**AEVS (Anchor-Extraction-Verification-Supplement, 2026)。** 当前的幻觉缓解框架：

- **锚定 (Anchor)。** 用精确位置标识每个实体跨度和关系短语跨度。
- **抽取 (Extract)。** 生成与锚定跨度关联的三元组。
- **验证 (Verify)。** 将每个三元组元素匹配回源文本；拒绝任何不受支持的内容。
- **补充 (Supplement)。** 覆盖性检查确保没有锚定跨度被遗漏。

幻觉大幅下降。需要更多计算，但是可审计的。

**开放 vs 封闭的权衡。**

- **封闭本体 (Closed ontology)。** 固定属性列表（例如 Wikidata 的 11,000+ 属性）。可预测。可查询。难以发明新关系。
- **开放信息抽取 (Open IE)。** 任何动词短语都可以成为关系。高召回率 (recall)。低精确率 (precision)。查询混乱。

生产知识图谱通常混合使用：开放 IE 用于发现，然后在合并到主图之前将关系规范化为封闭本体。

## 动手实现 (Build It)

### 步骤 1：基于模式的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

参见 `code/main.py` 中的完整 toy 抽取器。Hearst 模式仍在领域特定管道中部署，因为它们可调试。

### 步骤 2：监督关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一个 seq2seq 关系抽取器：文本进，三元组出，已经是 Wikidata 属性 id。在远程监督 (distant-supervision) 数据上微调。标准的开源权重基线。

### 步骤 3：带锚定的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

根据源文本验证每个返回的跨度。拒绝任何 `text[start:end] != triple_entity` 的情况。这是 AEVS "验证"步骤的最小形式。

### 步骤 4：规范化为封闭本体

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 丢弃未映射的开放关系或路由到人工审核
```

规范化 (canonicalization) 通常占工程工作的 60-80%。为此预留预算。

### 步骤 5：构建小型图谱并查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个 RAG-over-KG 系统的原子。使用 RDF 三元组存储（Blazegraph、Virtuoso）、属性图（Neo4j）或向量增强图存储来扩展。

## 常见陷阱 (Pitfalls)

- **RE 之前先做共指消解 (coreference before RE)。** "He founded Apple" — RE 需要知道 "he" 是谁。先运行 coref（第 24 课）。
- **实体规范化 (entity canonicalization)。** "Apple Inc" 和 "Apple" 必须解析到同一个节点。先做实体链接（第 25 课）。
- **幻觉三元组 (hallucinated triples)。** LLM 会输出文本不支持的三元组。强制执行跨度验证。
- **关系规范化漂移 (relation canonicalization drift)。** 开放 IE 关系不一致（"was born in," "came from," "is a native of"）。折叠为规范 id，否则图谱无法查询。
- **时间错误 (temporal errors)。** "Tim Cook is CEO of Apple" — 现在为真，2005 年为假。许多关系是时间受限的。使用限定符 (qualifiers)（Wikidata 中的 `P580` 开始时间, `P582` 结束时间）。
- **领域不匹配 (domain mismatch)。** REBEL 在 Wikipedia 上训练。法律、医疗和科学文本通常需要领域微调 (domain-fine-tuned) 的 RE 模型。

## 实际应用 (Use It)

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 快速生产，通用领域 | REBEL 或 LlamaPred 配合 Wikidata 规范化 |
| 领域特定（生物医学、法律） | SciREX 风格领域微调 + 自定义本体 |
| LLM 提示，审计输出 | AEVS 管道：锚定 → 抽取 → 验证 → 补充 |
| 高容量新闻信息抽取 | 基于模式 + 监督混合 |
| 从零构建 KG | 开放 IE + 手动规范化检查 |
| 时序 KG | 带限定符抽取（开始/结束时间、时间点） |

集成模式：NER → coref → 实体链接 → 关系抽取 → 本体映射 → 图谱加载。每个阶段都是潜在的质量门。

## 交付物 (Ship It)

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习 (Exercises)

1. **简单。** 在 5 篇新闻文章句子上运行 `code/main.py` 中的模式抽取器。手工检查精确率 (precision)。
2. **中等。** 在相同句子上使用 REBEL（或小型 LLM）。比较三元组。哪个抽取器精确率更高？召回率 (recall) 更高？
3. **困难。** 构建 AEVS 管道：用 LLM 抽取 + 针对源文本验证跨度。在 50 个 Wikipedia 风格句子上测量验证步骤前后的幻觉率 (hallucination rate)。

## 关键术语 (Key Terms)

| 术语 | 通常说法 | 实际含义 |
|------|-----------------|-----------------------|
| Triple | Subject-relation-object | KG 的原子单元：`(s, r, o)` 元组。 |
| Open IE | 抽取任何关系 | 开放词汇关系短语；高召回率，低精确率。 |
| Closed ontology | 固定模式 | 有限的关系类型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | 规范化一切 | 将表面名称 / 关系映射到规范 id。 |
| AEVS | 有根据的抽取 | 锚定-抽取-验证-补充管道 (Anchor-Extraction-Verification-Supplement, 2026)。 |
| Provenance | 真相来源链接 | 每个三元组携带文档 id + 字符跨度指向其来源。 |
| Distant supervision | 廉价标签 | 将文本与现有 KG 对齐以创建训练数据。 |

## 延伸阅读 (Further Reading)

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — 远程监督论文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 主力方法。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 联合信息抽取。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年幻觉缓解设计。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 规范图谱查询。
