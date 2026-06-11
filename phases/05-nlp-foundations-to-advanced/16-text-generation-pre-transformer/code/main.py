import math
import random
from collections import Counter, defaultdict


def tokenize(text):
    """将文本转为小写，拆分词元，保留标点作为独立词元。"""
    return text.lower().replace(".", " .").replace(",", " ,").split()


def train_bigrams(sentences):
    """
    从句子列表中统计 bigram、unigram 和 unigram 上下文集合。

    返回:
        bigrams: (prev, w) -> 计数的 Counter
        unigrams: w -> 计数的 Counter
        unigram_contexts: w -> 出现在 w 之前的不同词的集合
    """
    bigrams = Counter()
    unigrams = Counter()
    unigram_contexts = defaultdict(set)
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)
    return bigrams, unigrams, unigram_contexts


def kneser_ney_prob(bigrams, unigram_contexts, context_totals, unique_follow, total_unique_bigrams, prev, w, discount=0.75):
    """
    计算单个 bigram 的 Kneser-Ney 平滑概率。

    参数:
        bigrams: (prev, w) 的计数 Counter
        unigram_contexts: w -> 前驱词集合
        context_totals: prev -> 所有 bigram 计数的总和
        unique_follow: prev -> 以 prev 开头的不同后续词集合
        total_unique_bigrams: 所有不同 bigram 的总数
        prev: 前驱词
        w: 目标词
        discount: 绝对折扣 D

    返回:
        平滑后的 P(w | prev)
    """
    count = bigrams.get((prev, w), 0)
    denom = context_totals.get(prev, 0)
    continuation = len(unigram_contexts.get(w, set())) / max(total_unique_bigrams, 1)
    if denom == 0:
        return continuation or 1e-9
    first = max(count - discount, 0) / denom
    lam = discount * len(unique_follow[prev]) / denom
    return first + lam * continuation


def laplace_prob(bigrams, unigrams, vocab_size, prev, w):
    """计算 Laplace（加一）平滑后的 bigram 概率。"""
    num = bigrams.get((prev, w), 0) + 1
    den = unigrams.get(prev, 0) + vocab_size
    return num / den


def perplexity(prob_fn, sentences):
    """
    在测试句子上计算困惑度（perplexity）。

    参数:
        prob_fn: 接收 (prev, w) 返回概率的函数
        sentences: 分词后的句子列表

    返回:
        困惑度值（越低越好）
    """
    total_log = 0.0
    total = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log += math.log(max(p, 1e-12))
            total += 1
    return math.exp(-total_log / total)


def sample_sentence(prob_fn, vocab, max_len=15, seed=0):
    """
    按概率比例采样生成一个句子，直到遇到 </s> 或达到最大长度。

    参数:
        prob_fn: 接收 (prev, w) 返回概率的函数
        vocab: 词表列表
        max_len: 最大生成长度
        seed: 随机数种子

    返回:
        生成的词元列表（不含 <s> 和 </s>）
    """
    rng = random.Random(seed)
    tokens = ["<s>"]
    for _ in range(max_len):
        probs = [(w, prob_fn(tokens[-1], w)) for w in vocab if w != "<s>"]
        total = sum(p for _, p in probs)
        r = rng.random() * total
        acc = 0.0
        for w, p in probs:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            return tokens[1:-1]
    return tokens[1:]


def main():
    raw = [
        "the cat sat on the mat .",
        "the cat ran across the room .",
        "the dog sat by the window .",
        "a cat chased the mouse .",
        "the dog ran after the cat .",
        "a mouse hid under the table .",
        "the cat watched the birds .",
        "the dog chased the ball .",
        "a bird sat on the branch .",
        "the cat slept on the couch .",
    ]
    train = [tokenize(s) for s in raw]
    test = [tokenize("the cat sat on the couch ."), tokenize("the dog watched the mouse .")]

    bigrams, unigrams, unigram_contexts = train_bigrams(train)
    vocab = list(unigrams.keys())
    vocab_size = len(vocab)
    context_totals = Counter()
    unique_follow = defaultdict(set)
    for (prev, w), c in bigrams.items():
        context_totals[prev] += c
        unique_follow[prev].add(w)
    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())

    def kn(prev, w):
        return kneser_ney_prob(bigrams, unigram_contexts, context_totals, unique_follow, total_unique_bigrams, prev, w)

    def lap(prev, w):
        return laplace_prob(bigrams, unigrams, vocab_size, prev, w)

    print(f"vocab size: {vocab_size}")
    print(f"train sentences: {len(train)}")
    print(f"test sentences: {len(test)}")
    print()
    print(f"perplexity (Laplace):     {perplexity(lap, test):.2f}")
    print(f"perplexity (Kneser-Ney):  {perplexity(kn, test):.2f}")
    print()
    print("=== sampled sentences (Kneser-Ney, 3 seeds) ===")
    for seed in [1, 7, 42]:
        sentence = sample_sentence(kn, vocab, seed=seed)
        print(f"  seed={seed}: {' '.join(sentence)}")


if __name__ == "__main__":
    main()
