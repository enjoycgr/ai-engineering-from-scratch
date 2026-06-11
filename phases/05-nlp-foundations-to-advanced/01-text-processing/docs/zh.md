# 文本处理 —— 分词、词干提取、词形还原

> 语言是连续的。模型是离散的。预处理是桥梁。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 分钟

## 问题所在

模型无法直接阅读 "The cats were running."。它读取的是整数。

每个 NLP 系统都会面临同样的三个问题：单词从哪里开始；单词的词根是什么；我们如何在需要时将 "run"、"running"、"ran" 视为同一事物，在不需要时又将它们视为不同事物。

分词出错，模型就会从垃圾中学习。如果你的 tokenizer 将 `don't` 视为一个 token，却将 `do n't` 视为两个，训练分布就会分裂。如果你的 stemmer 将 `organization` 和 `organ` 折叠为同一个词干，主题建模就会失效。如果你的 lemmatizer 需要词性上下文但你没有传递，动词就会被当作名词处理。

本节课从零构建三个预处理步骤，然后展示 NLTK 和 spaCy 如何完成同样的工作，让你了解其中的权衡。

## 核心概念

三个操作。每个都有其作用和失效模式。

**分词 (Tokenization)** 将字符串拆分为 token。"Token" 故意保持模糊，因为合适的粒度取决于任务。经典 NLP 使用词级别。Transformer 使用子词级别。无空格语言使用字符级别。

**词干提取 (Stemming)** 用规则砍掉后缀。快速、激进、简单粗暴。`running -> run`。`organization -> organ`。后者就是它的失效模式。

**词形还原 (Lemmatization)** 使用语法知识将单词还原为词典形式。更慢、更准确，需要查表或形态分析器。`ran -> run`（需要知道 "ran" 是 "run" 的过去式）。`better -> good`（需要知道比较级形式）。

经验法则：当速度重要且可以容忍噪声时（搜索索引、粗略分类），使用 stemming。当语义重要时（问答、语义搜索、任何用户会阅读的内容），使用 lemmatization。

## 动手实现

### 第一步：正则表达式分词器

最简单但有用的分词器在非字母数字字符处分割，同时将标点符号保留为独立的 token。不完美，也不是最终方案，但一行代码就能运行。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三个模式按优先级排列。带有可选内部撇号的单词（`don't`、`it's`）。纯数字。任何单个非空白非字母数字字符作为独立 token（标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需要注意的失效模式。`3pm` 被拆分为 `['3', 'pm']`，因为我们在字母串和数字串之间交替。对大多数任务来说已经足够好。URL、电子邮件、标签都会断裂。对于生产环境，在通用模式之前添加特定模式。

### 第二步：Porter 词干提取器（仅步骤 1a）

完整的 Porter 算法有五阶段规则。仅步骤 1a 就覆盖了最常见的英语后缀，并展示了该模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

从上到下阅读规则。`ies -> i` 规则就是 `ponies -> poni` 而不是 `pony` 的原因。真正的 Porter 有步骤 1b 可以修复它。规则相互竞争。前面的规则优先。顺序比任何单个规则都重要。

### 第三步：基于查找表的词形还原器

真正的词形还原需要形态学知识。一个可行的教学版本使用小型词元表和回退机制。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是关键的教学时刻。`watched` 不在我们的表中，而且我们的回退只处理 `ing`。真正的词形还原覆盖 `ed`、不规则动词、比较级形容词、发音变化的复数（`children -> child`）。这就是生产系统使用 WordNet、spaCy 的 morphologizer 或完整形态分析器的原因。

### 第四步：将它们串联起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的部分是 POS 标注器。Phase 5 · 07（POS 标注）会构建一个。目前，将所有内容默认为 `NOUN` 并承认这一限制。

## 使用现有工具

NLTK 和 spaCy 都提供了生产级版本。各需几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 处理缩略形式、Unicode、你的正则表达式遗漏的边界情况。`PorterStemmer` 运行全部五个阶段。`WordNetLemmatizer` 需要将 POS 标签从 NLTK 的 Penn Treebank 方案转换为 WordNet 的缩写集。上面的转换连接是大多数教程跳过的那部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个流程隐藏在 `nlp(text)` 之后。分词、POS 标注和词形还原全部运行。大规模处理时比 NLTK 更快。开箱即用更准确。代价是你不能轻易替换单个组件。

### 如何选择

| 场景 | 选择 |
|-----------|------|
| 教学、研究、替换组件 | NLTK |
| 生产环境、多语言、速度重要 | spaCy |
| Transformer 流程（反正你会用模型的 tokenizer） | 使用 `tokenizers` / `transformers`，跳过经典预处理 |

### 没人告诉你的两个失效模式

大多数教程只教算法就停了。有两件事会在真实的预处理流程中咬你一口，而且几乎从未被提及。

**可复现性漂移。** NLTK 和 spaCy 在不同版本之间会改变分词和词形还原器的行为。spaCy 2.x 产生 `['do', "n't"]` 的内容，在 3.x 中可能产生 `["don't"]`。你的模型是在一种分布上训练的。inference 现在在另一种分布上运行。准确率悄然下降，没人知道原因。在 `requirements.txt` 中锁定库版本。编写一个预处理回归测试，冻结 20 个样本句子的预期分词结果。每次升级时运行它。

**训练/推理不匹配。** 训练时使用激进的预处理（小写化、停用词移除、词干提取），部署时处理原始用户输入，看着性能暴跌。这是生产 NLP 中最常见的单一故障。如果你在训练时进行预处理，必须在推理时运行完全相同的函数。将预处理作为函数封装在模型包内，而不是作为 serving 团队重写的 notebook 单元格。

## 交付物

一个可复用的 prompt，帮助工程师在不阅读三本教科书的情况下选择预处理策略。

保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: 为 NLP 任务推荐分词、词干提取和词形还原配置。
phase: 5
lesson: 01
---

你提供经典 NLP 预处理建议。给定任务描述，你输出：

1. 分词选择（regex、NLTK word_tokenize、spaCy 或 transformer tokenizer）。解释原因。
2. 是否进行 stemming、lemmatization、两者都做，或都不做。解释原因。
3. 具体的库调用。命名函数。如果涉及 NLTK，引用 POS 标签转换。
4. 用户应该测试的一个失效模式。

拒绝为用户可见的文本推荐 stemming。拒绝在没有 POS 标签的情况下推荐 lemmatization。标记非英语输入需要不同的流程。
```

## 练习

1. **简单。** 扩展 `tokenize` 以将 URL 保留为单个 token。测试：`tokenize("Visit https://example.com today.")` 应该产生一个 URL token。
2. **中等。** 实现 Porter 步骤 1b。如果一个单词包含元音且以 `ed` 或 `ing` 结尾，则移除它。处理双辅音规则（`hopping -> hop`，而不是 `hopp`）。
3. **困难。** 构建一个使用 WordNet 作为查找表但在 WordNet 没有条目时回退到你的 Porter stemmer 的词形还原器。在标注语料库上测量准确率，与纯 WordNet 和纯 Porter 进行比较。

## 关键术语

| 术语 | 人们通常说的 | 实际含义 |
|------|-------------|-----------------------|
| Token | 一个单词 | 模型消费的任何单位。可以是词、子词、字符或字节。 |
| Stem | 单词的词根 | 基于规则的后缀剥离结果。不总是真实存在的单词。 |
| Lemma | 词典形式 | 你会去查词典的形式。需要语法上下文才能正确计算。 |
| POS tag | 词性 | 如 NOUN、VERB、ADJ 等类别。需要准确进行词形还原。 |
| Morphology | 词形规则 | 单词如何根据时态、数、格改变形式。词形还原依赖于此。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) —— 原始论文，五页，仍然是最清晰的解释。
- [spaCy 101 —— 语言特征](https://spacy.io/usage/linguistic-features) —— 真实流程是如何连接的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) —— 你还没想到过的分词边界情况。
