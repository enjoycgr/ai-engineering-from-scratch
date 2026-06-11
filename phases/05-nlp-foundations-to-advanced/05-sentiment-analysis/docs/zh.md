# Sentiment Analysis（情感分析）

> 最经典的 NLP 任务。关于经典文本分类，你需要知道的大部分内容都会在这里出现。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 分钟

## The Problem（问题）

"The food was not great." 是正面还是负面？

Sentiment（情感分析）听起来很简单。评论者说了他们喜欢或不喜欢某样东西。给句子打上标签。它之所以成为最经典的 NLP 任务，是因为每一个看起来简单的案例背后都隐藏着一个困难的案例。Negation（否定）会翻转含义。Sarcasm（讽刺）会反转含义。"Not bad at all" 尽管包含两个负面编码的词，却是正面的。Emoji 比周围文本携带更多信号。Domain vocabulary（领域词汇）很重要（`tight` 在音乐评论中与在时尚评论中的含义不同）。

Sentiment 是经典 NLP 的一个实践实验室。如果你理解为什么每一个 naive baseline（朴素基线）都有特定的 failure mode（失效模式），你就能理解为什么每一个更丰富的模型被发明出来。本节课从零开始构建一个 Naive Bayes（朴素贝叶斯）基线，添加 logistic regression（逻辑回归），并指出使生产级 sentiment 成为 compliance-grade problem（合规级问题）的那些陷阱。

## The Concept（概念）

Classical sentiment（经典情感分析）是一个两步配方。

1. **Represent（表示）。** 将文本转换为特征向量。BoW（词袋）、TF-IDF 或 n-grams（n 元语法）。
2. **Classify（分类）。** 在标注样本上拟合一个 linear model（线性模型）（Naive Bayes、logistic regression、SVM）。

Naive Bayes 是能工作的最简模型。假设每个特征在给定标签下是独立的。从计数中估计 `P(word | positive)` 和 `P(word | negative)`。在 inference（推理）时，将概率相乘。"naive" independence assumption（独立性假设）错得离谱，但结果却出奇地好。原因在于：在稀疏文本特征和中等数据量的情况下，分类器更关心每个词倾向于哪一侧，而不是具体的联合概率。

Logistic regression 修复了独立性假设。它为每个特征学习一个权重，包括负权重。`not good` 作为一个 bigram（二元语法）特征会得到负权重。而 Naive Bayes 无法对它从未标注过的 bigram 做到这一点。

## Build It（动手实现）

### Step 1: a real mini-dataset（一个真实的微型数据集）

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

故意做得很小。实际工作中使用数万个样本（IMDb、SST-2、Yelp polarity）。数学原理完全相同。

### Step 2: multinomial Naive Bayes from scratch（从零实现多项式朴素贝叶斯）

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing（加性平滑）（alpha=1.0）就是 Laplace smoothing（拉普拉斯平滑）。没有它，某个类别中未出现的词概率为零，log 会爆炸。实践中常用 `alpha=0.01`。`alpha=1.0` 是教学默认值。

### Step 3: logistic regression from scratch（从零实现逻辑回归）

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization（L2 正则化）在这里很重要。文本特征是稀疏的；没有 L2，模型会 memorizes training examples（记住训练样本）。从 `0.01` 开始并调参。

### Step 4: handling negation (the failure mode)（处理否定——失效模式）

考虑 "not good" 和 "not bad"。一个 BoW 分类器看到的是 `{not, good}` 和 `{not, bad}`，从训练集中出现更多的那个来学习。一个 bigram 分类器看到的是 `not_good` 和 `not_bad`，将它们作为不同的特征来学习。这通常就足够了。

当你没有 bigram 时，一个更粗糙但有效的修复方法是：**negation scoping（否定范围标注）**。在否定词之后、下一个标点符号之前，给所有 token 加上 `NOT_` 前缀。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在 `good` 和 `NOT_good` 是不同的特征。分类器可以给它们赋予相反的权重。三行预处理代码，在 sentiment benchmark 上带来可测量的准确率提升。

### Step 5: evaluation metrics that matter（重要的评估指标）

如果类别不平衡，仅靠 Accuracy（准确率）是误导性的。真实的 sentiment 语料库通常是 70-80% 正面或 70-80% 负面；一个恒定为多数类的分类器可以达到 80% 的准确率，但毫无价值。报告以下每一项：

- **Per-class precision and recall（每类精确率和召回率）。** 每个类别一对。Macro-average（宏平均）它们，得到一个尊重类别平衡的单一数字。
- **Macro-F1 (primary metric for imbalanced data)（Macro-F1——不平衡数据的主要指标）。** 每类 F1 分数的均值，等权重。当类别不平衡时，用它代替 accuracy。
- **Weighted-F1 (alternative)（Weighted-F1——替代方案）。** 与 macro 相同，但按类别频率加权。当不平衡本身具有业务意义时，与 Macro-F1 一起报告。
- **Confusion matrix（混淆矩阵）。** 原始计数。在信任任何标量指标之前，始终检查它；它揭示了模型混淆了哪一对类别。
- **Per-class error samples（每类错误样本）。** 每个类别抽取 5 个错误预测。阅读它们。没有什么能替代阅读实际的错误。

对于严重不平衡的数据（> 95-5 比例），报告 **AUROC** 和 **AUPRC** 而不是 accuracy。AUPRC 对少数类更敏感，而这通常是你关心的（垃圾邮件、欺诈、罕见情感）。

**Common bug to avoid（需要避免的常见错误）。** 在不平衡数据上报告 micro-F1 而不是 macro-F1 会得到一个看起来很高的数字，因为它被多数类主导。Macro-F1 迫使你看到少数类的表现。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Use It（使用它）

scikit-learn 用六行代码就能正确实现。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

有三点需要注意。`stop_words=None` 保留否定词。`ngram_range=(1, 2)` 添加 bigram，使 `not_good` 成为一个特征。`sublinear_tf=True` 减弱重复词的影响。这三个参数是将 SST-2 上 75% 准确率的基线提升到 85% 准确率的关键。

### When to reach for a transformer（何时该用 transformer）

- Sarcasm detection（讽刺检测）。Classical models 在这里完全失败。
- 情感在文档中间发生转变的长评论。
- Aspect-based sentiment（基于方面的情感分析）。"Camera was great but battery was terrible." 你需要将情感归因于特定方面。只有 Transformers 或 structured output models 能做到。
- 非英语、低资源语言。Multilingual BERT 免费提供 zero-shot baseline。

如果你需要以上任何一种，跳到 phase 7（transformers 深入）。否则，TF-IDF 上的 Naive Bayes 或 logistic regression，加上 bigram 和否定处理，就是你 2026 年的生产基线。

### The reproducibility trap (again)（可复现性陷阱——再次出现）

重新训练 sentiment 模型是常规操作。重新评估它们却不是。论文中报告的 accuracy 数字使用特定的 split、特定的预处理、特定的 tokenizer。如果你在不使用相同 pipeline 的情况下将你的新模型与基线比较，你会得到误导性的差异。始终在你的 pipeline 上重新生成基线，而不是使用论文中的数字。

## Ship It（交付）

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Exercises（练习）

1. **Easy.** 在 scikit-learn pipeline 中添加 `apply_negation` 作为预处理步骤，并在小型 sentiment 数据集上测量 F1 的变化。
2. **Medium.** 实现 class-weighted logistic regression（类别加权逻辑回归）（向 scikit-learn 传递 `class_weight="balanced"`，或自己推导梯度）。在合成的 90-10 类别不平衡上测量效果。
3. **Hard.** 通过在 sentiment 模型的残差上训练第二个分类器来构建一个 sarcasm detector（讽刺检测器）。记录你的实验设置。当你的准确率低于 chance（2 类讽刺的 chance-level 约为 50%，大多数首次尝试都会落在这里）时警告读者。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Polarity | Positive or negative | 二元标签；有时扩展到 neutral 或 fine-grained（5 星）。 |
| Aspect-based sentiment | Per-aspect polarity | 将情感归因于文本中提到的特定实体或属性。 |
| Negation scoping | Reversing nearby tokens | 在 "not" 之后给 token 加上 `NOT_` 前缀，直到遇到标点符号。 |
| Laplace smoothing | Adding 1 to counts | 防止 Naive Bayes 中出现零概率特征。 |
| L2 regularization | Shrinking weights | 向 loss 中添加 `lambda * sum(w^2)`。对稀疏文本特征至关重要。 |

## Further Reading（延伸阅读）

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) —— 奠基性综述。很长，但前四节涵盖了所有经典内容。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) —— 这篇论文证明了 bigrams + Naive Bayes 在短文本上很难被击败。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— `CountVectorizer`、`TfidfVectorizer` 以及你会调整的每个参数的参考文档。
