"""T5 跨度破坏（span corruption）+ BART 去噪噪声函数。

仅使用标准库。展示编码器-解码器模型如何将任意输入转换为
有监督的（corrupted_input -> clean_spans）训练对。
"""

import random


def sentinel(i):
    return f"<extra_id_{i}>"


def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """T5-style span corruption（T5 风格的跨度破坏）。

    返回 (corrupted_source, decoder_target)，均为 token（字符串）列表。
    """
    if rng is None:
        rng = random.Random()
    n = len(tokens)
    n_mask = max(1, int(round(n * mask_rate)))
    n_spans = max(1, int(round(n_mask / mean_span)))
    # 选取不重叠的跨度起始位置。
    positions = list(range(n))
    rng.shuffle(positions)
    starts = []
    used = [False] * n
    span_lengths = []
    remaining = n_mask
    for _ in range(n_spans):
        if remaining <= 0:
            break
        # 随机选取一个尚未使用且有足够空间的起始点
        random_order = list(range(n))
        rng.shuffle(random_order)
        chosen_start = None
        for start in random_order:
            if used[start]:
                continue
            # 跨度长度
            length = max(1, int(rng.gauss(mean_span, 1.0)))
            length = min(length, remaining, n - start)
            if length < 1:
                continue
            if any(used[i] for i in range(start, start + length)):
                continue
            chosen_start = start
            for i in range(start, start + length):
                used[i] = True
            starts.append(start)
            span_lengths.append(length)
            remaining -= length
            break
        if chosen_start is None:
            break

    ordered = sorted(zip(starts, span_lengths), key=lambda x: x[0])

    source = []
    target = []
    prev_end = 0
    for idx, (start, length) in enumerate(ordered):
        source.extend(tokens[prev_end:start])
        source.append(sentinel(idx))
        target.append(sentinel(idx))
        target.extend(tokens[start:start + length])
        prev_end = start + length
    source.extend(tokens[prev_end:])
    target.append(sentinel(len(ordered)))  # 结尾哨兵（closing sentinel）
    return source, target


def round_trip(source, target):
    """用目标中的对应跨度替换源中的哨兵，重建原始序列。"""
    # 将目标解析为 sentinel->span 映射
    spans = {}
    current_key = None
    current_span = []
    for tok in target:
        if tok.startswith("<extra_id_"):
            if current_key is not None:
                spans[current_key] = current_span
            current_key = tok
            current_span = []
        else:
            current_span.append(tok)
    # 目标中的最后一个哨兵后面没有跨度（它是结尾标记）。
    out = []
    for tok in source:
        if tok.startswith("<extra_id_"):
            out.extend(spans.get(tok, []))
        else:
            out.append(tok)
    return out


def token_mask(tokens, rate=0.15, rng=None, mask_token="<mask>"):
    if rng is None:
        rng = random.Random()
    return [mask_token if rng.random() < rate else t for t in tokens]


def token_delete(tokens, rate=0.15, rng=None):
    if rng is None:
        rng = random.Random()
    return [t for t in tokens if rng.random() >= rate]


def text_infill(tokens, rate=0.15, mean_span=3.0, rng=None, mask_token="<mask>"):
    """BART text infill（文本填充）：用单个 mask 掩码跨度；解码器推断长度。"""
    if rng is None:
        rng = random.Random()
    out = []
    i = 0
    n = len(tokens)
    budget = int(n * rate)
    while i < n:
        if budget > 0 and rng.random() < 0.3:
            span_len = max(1, min(int(rng.gauss(mean_span, 1.0)), budget, n - i))
            out.append(mask_token)
            budget -= span_len
            i += span_len
        else:
            out.append(tokens[i])
            i += 1
    return out


def sentence_permute(sentences, rng=None):
    if rng is None:
        rng = random.Random()
    sents = list(sentences)
    rng.shuffle(sents)
    return sents


def document_rotate(tokens, rng=None):
    if rng is None:
        rng = random.Random()
    if len(tokens) <= 1:
        return tokens
    pivot = rng.randrange(1, len(tokens))
    return tokens[pivot:] + tokens[:pivot]


def main():
    rng = random.Random(42)

    sentence = (
        "the quick brown fox jumps over the lazy dog a stitch in time saves nine "
        "language models learn statistical patterns subword tokenization helps rare words"
    ).split()

    print("=== T5 span corruption ===")
    source, target = corrupt_spans(sentence, mask_rate=0.20, mean_span=3.0, rng=rng)
    print("corrupted source:")
    print("  " + " ".join(source))
    print()
    print("decoder target:")
    print("  " + " ".join(target))
    print()
    reconstructed = round_trip(source, target)
    print("reconstruction matches original:",
          "YES" if reconstructed == sentence else "NO")
    if reconstructed != sentence:
        print("  reconstructed: " + " ".join(reconstructed))

    print()
    print("=== BART noise functions ===")
    print("original: " + " ".join(sentence[:14]))
    print()
    print("token mask:     " + " ".join(token_mask(sentence[:14], rate=0.2, rng=random.Random(1))))
    print("token delete:   " + " ".join(token_delete(sentence[:14], rate=0.2, rng=random.Random(2))))
    print("text infill:    " + " ".join(text_infill(sentence[:14], rate=0.3, rng=random.Random(3))))

    sentences = [
        ["the", "quick", "brown", "fox"],
        ["a", "stitch", "in", "time"],
        ["language", "models", "learn", "patterns"],
    ]
    perm = sentence_permute(sentences, rng=random.Random(4))
    print("sentence permute:")
    for s in perm:
        print("  " + " ".join(s))

    print()
    print("document rotate: " + " ".join(document_rotate(sentence[:14], rng=random.Random(5))))


if __name__ == "__main__":
    main()
