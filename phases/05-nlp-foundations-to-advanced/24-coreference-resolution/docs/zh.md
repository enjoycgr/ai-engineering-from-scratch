# 指代消解（Coreference Resolution）

> "She called him. He did not answer. The doctor was at lunch." 三个指代，两个人，没有人被命名。指代消解要弄清楚谁是谁。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 分钟

## 问题所在

从一篇 300 字的文章中提取 Apple Inc. 的每一次提及。当文章说 "Apple" 时很容易。当文章说 "the company"、"they"、"Cupertino's technology giant" 或 "Jobs's firm" 时就很难了。如果不将这些提及解析为同一实体，你的 NER 流水线会遗漏 60-80% 的提及。

指代消解（Coreference resolution）将指代同一现实世界实体的所有表达链接到一个簇中。它是表层 NLP（NER、句法分析）与下游语义（信息抽取、问答、摘要、知识图谱）之间的粘合剂。

2026 年为何重要：

- 摘要："The CEO announced..." 与 "Tim Cook announced..." — 摘要应该说出 CEO 的名字。
- 问答："Who did she call?" 需要解析 "she"。
- 信息抽取：一个知识图谱中同时存在 "PER1 founded Apple" 和 "Jobs founded Apple" 是错误的。
- 多文档信息抽取：合并关于同一事件的多篇文章中的提及是跨文档指代消解。

## 概念

![指代聚类：mentions → entities](../assets/coref.svg)

**任务。** 输入：一篇文档。输出：提及（spans）的聚类，每个簇指代一个实体。

**提及类型。**

- **命名实体（Named entity）。** "Tim Cook"
- **名词性（Nominal）。** "the CEO"、"the company"
- **代词性（Pronominal）。** "he"、"she"、"they"、"it"
- **同位语（Appositive）。** "Tim Cook, Apple's CEO,"

**架构。**

1. **基于规则（Hobbs, 1978）。** 使用语法规则的句法树代词解析。良好的基线。在代词上 surprisingly hard to beat。
2. **提及对分类器（Mention-pair classifier）。** 对每一对提及 (m_i, m_j)，预测它们是否共指。通过传递闭包聚类。2016 年前的标准方法。
3. **提及排序（Mention-ranking）。** 对每个提及，对候选先行词排序（包括"无先行词"）。选择最高分。
4. **基于 Span 的端到端（Lee et al., 2017）。** Transformer 编码器。枚举所有长度受限的候选 span。预测提及分数。为每个 span 预测先行词概率。贪婪聚类。现代默认方法。
5. **生成式（2024+）。** 提示 LLM："列出这段文本中的每个代词及其先行词。" 在简单案例上表现良好，在长文档和罕见指代上挣扎。

**评估指标。** 五个标准指标（MUC、B³、CEAF、BLANC、LEA），因为没有一个单一指标能捕捉聚类质量。报告前三个的平均值作为 CoNLL F1。2026 年 CoNLL-2012 上的最先进水平：~83 F1。

**已知困难案例。**

- 指代数页前引入的实体的定指描述。
- 桥接回指（"the wheels" → 之前提到的汽车）。
- 中文和日语中的零回指。
- 预指（Cataphora，代词在指代对象之前）："When **she** walked in, Mary smiled."

## 动手实现

### 第 1 步：预训练神经指代消解（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 实验模型
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在更长的文档上，你会得到类似：
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 第 2 步：基于规则的代词解析器（教学用）

参见 `code/main.py` 中的纯标准库实现：

1. 提取提及：命名实体（大写 spans）、代词（字典查找）、定指描述（"the X"）。
2. 对每个代词，查看前 K 个提及并按以下标准打分：
   - 性别/数一致（启发式）
   - 新近性（越近越好）
   - 句法角色（主语优先）
3. 链接得分最高的先行词。

无法与神经模型竞争。但它展示了搜索空间以及端到端模型必须做出的决策。

### 第 3 步：使用 LLM 进行指代消解

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

两种失败模式需注意。第一，LLM 过度合并（"him" 和 "her" 指代两个不同的人）。第二，LLM 在长文档中静默丢弃提及。始终用 span offset 检查验证。

### 第 4 步：评估

标准 conll-2012 脚本计算 MUC、B³、CEAF-φ4 并报告平均值。对于内部评估，从在你标注的测试集上的 span-level precision 和 recall 开始，然后添加 mention-linking F1。

## 常见陷阱

- **单例爆炸（Singleton explosion）。** 某些系统将每个提及报告为其自己的簇。B³ 对此较宽容。MUC 会惩罚。始终检查所有三个指标。
- **长上下文中的代词。** 在超过 2,000 token 的文档上性能下降约 15 F1。谨慎分块。
- **性别假设。** 硬编码的性别规则在非二元指代对象、组织、动物上失效。使用学习模型或中性打分。
- **LLM 在长文档上的漂移。** 单次 API 调用无法可靠地聚类跨越 50+ 段落的提及。使用滑动窗口 + 合并。

## 如何使用

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语，单文档 | `en_coreference_web_trf` (spaCy-experimental) 或 AllenNLP neural coref |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 上训练的 SpanBERT / XLM-R |
| 跨文档事件共指 | 专用端到端模型（2025–26 SOTA） |
| 快速 LLM 基线 | 使用结构化输出指代提示的 GPT-4o / Claude |
| 生产对话系统 | 基于规则的兜底 + 神经主模型 + 关键槽位人工审核 |

2026 年实际落地的集成模式：先运行 NER，再运行指代消解，将指代簇合并到 NER 实体中。下游任务看到的是每个簇一个实体，而不是每个 surface mention 一个实体。

## 交付

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: 选择指代消解方法、评估计划和集成策略。
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

给定用例（单文档 / 多文档、领域、语言），输出：

1. 方法。基于规则 / 神经 span-based / LLM 提示 / 混合。一句话理由。
2. 模型。如果是神经模型，给出具体 checkpoint 名称。
3. 集成。操作顺序：tokenize → NER → coref → 下游任务。
4. 评估。在留出集上测试 CoNLL F1（MUC + B³ + CEAF-φ4 平均值）+ 在 20 篇文档上进行人工簇审核。

拒绝没有滑动窗口合并的、对超过 2,000 token 文档使用纯 LLM 指代消解的方案。拒绝任何没有 mention-level precision-recall 报告的流水线。标记部署在人口统计学多样化文本中的性别启发式系统。
```

## 练习

1. **简单。** 在 5 个手工编写的段落上运行 `code/main.py` 中的基于规则解析器。针对 ground truth 测量 mention-link 准确率。
2. **中等。** 在新闻文章上使用预训练神经指代模型。将簇与你自己的手工标注比较。哪里失败了？
3. **困难。** 构建指代增强的 NER 流水线：先 NER，再通过指代簇合并。在 100 篇文章上测量相比纯 NER 的实体覆盖率提升。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Mention | 一个指代 | 指代实体的文本 span（名字、代词、名词短语）。 |
| Antecedent | "it" 指代什么 | 后一个提及与其共指的较早提及。 |
| Cluster | 该实体的提及 | 全部指代同一现实世界实体的提及集合。 |
| Anaphora | 回指 | 后提及指代先提及（"he" → "John"）。 |
| Cataphora | 预指 | 先提及指代后提及（"When he arrived, John..."）。 |
| Bridging | 隐式指代 | "I bought a car. The wheels were bad."（那辆车的轮子。） |
| CoNLL F1 | 排行榜上的数字 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 经典教科书章节。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 基于 span 的端到端。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 改进指代消解的预训练。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 基准测试。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典。
