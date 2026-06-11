import re

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokenize(text):
    """将文本拆分为小写 token 列表。"""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_vocab(docs):
    """为语料库构建 word -> index 映射。"""
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def skipgram_pairs(docs, window=2):
    """为 skip-gram 生成所有 (center, context) 对。"""
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i != j:
                    pairs.append((center, doc[j]))
    return pairs


def sigmoid(x):
    """数值稳定的 sigmoid，带裁剪防止溢出。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def init_embeddings(vocab_size, dim, seed):
    """随机初始化中心矩阵 W 和上下文矩阵 W'。"""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime


def train_pair(W, W_prime, c_idx, ctx_idx, neg_indices, lr):
    """使用 negative sampling (负采样) 更新单个 (center, context) 对。"""
    v_c = W[c_idx]
    u_pos = W_prime[ctx_idx]
    u_negs = W_prime[neg_indices]

    pos_err = sigmoid(v_c @ u_pos) - 1.0
    neg_errs = sigmoid(u_negs @ v_c)

    grad_center = pos_err * u_pos + neg_errs @ u_negs
    W_prime[ctx_idx] -= lr * pos_err * v_c
    for i, neg_idx in enumerate(neg_indices):
        W_prime[neg_idx] -= lr * neg_errs[i] * v_c
    W[c_idx] -= lr * grad_center


def train(docs, dim=16, window=2, k_neg=5, epochs=200, lr=0.05, seed=0):
    """在 tokenized 文档上训练 skip-gram embedding。"""
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    W, W_prime = init_embeddings(vocab_size, dim, seed)
    pairs = skipgram_pairs(docs, window=window)
    rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            neg_candidates = rng.integers(0, vocab_size, size=k_neg * 2)
            negs = [int(n) for n in neg_candidates if n != ctx_idx and n != c_idx][:k_neg]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W


def nearest(vocab, W, target_vec, topk=5, exclude=None):
    """通过余弦相似度返回与 target_vec 最接近的词。"""
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if int(i) in exclude:
            continue
        out.append((inv_vocab[int(i)], float(sims[i])))
        if len(out) == topk:
            break
    return out


def main():
    # 小型玩具语料库，重复 20 次以放大信号。
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "a cat chased a mouse",
        "a dog chased a cat",
        "the kitten slept on the mat",
        "the puppy slept on the rug",
        "cats and dogs are pets",
        "kittens and puppies are young",
        "cats chase mice",
        "dogs chase squirrels",
    ] * 20

    docs = [tokenize(s) for s in corpus]
    vocab, W = train(docs, dim=16, window=2, k_neg=5, epochs=120, lr=0.05, seed=42)

    # 打印几个词的最近邻以验证几何结构。
    for word in ["cat", "dog", "sat", "chased"]:
        idx = vocab[word]
        top = nearest(vocab, W, W[idx], topk=4, exclude={idx})
        print(f"nearest to {word}:")
        for w, s in top:
            print(f"  {w:12s} {s:.3f}")
        print()


if __name__ == "__main__":
    main()
