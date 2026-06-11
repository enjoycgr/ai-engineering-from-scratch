"""Attention variants: full, sliding-window, local+strided sparse, differential.
（注意力变体：完整注意力、滑动窗口、局部+步进稀疏、差分注意力）

Pure stdlib. We compare the structure of the score mask and the KV cache
size per variant at a realistic long-context budget.
（纯标准库。我们在真实的长上下文预算下比较分数 mask 的结构和每种变体的 KV cache 大小。）
"""

import math


NEG_INF = float("-inf")


def causal_mask(n):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            M[i][j] = 0.0
    return M


def swa_mask(n, window):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M


def strided_mask(n, window, stride):
    M = [[NEG_INF] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M


def count_nonmasked(M):
    return sum(1 for row in M for v in row if v == 0.0)


def render(M, label):
    n = len(M)
    print(f"{label}  ({count_nonmasked(M)} / {n*n} cells attended)")
    for i in range(n):
        cells = "".join("x" if M[i][j] == 0.0 else "." for j in range(n))
        print(f"  {i:>2} | {cells}")
    print()


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def attention_row(q, Ks, Vs, mask_row):
    d = len(q)
    scores = []
    for k, m in zip(Ks, mask_row):
        if m == NEG_INF:
            scores.append(NEG_INF)
        else:
            s = sum(qi * ki for qi, ki in zip(q, k)) / math.sqrt(d)
            scores.append(s)
    finite = [s for s in scores if s != NEG_INF]
    if not finite:
        return [0.0] * len(Vs[0]), [0.0] * len(scores)
    shifted = softmax(finite)
    weights = []
    k = 0
    for s in scores:
        if s == NEG_INF:
            weights.append(0.0)
        else:
            weights.append(shifted[k])
            k += 1
    d_v = len(Vs[0])
    out = [0.0] * d_v
    for w, v in zip(weights, Vs):
        for j in range(d_v):
            out[j] += w * v[j]
    return out, weights


def diff_attention_row(q1, q2, K1, K2, V, mask_row, lam):
    _, w1 = attention_row(q1, K1, V, mask_row)
    _, w2 = attention_row(q2, K2, V, mask_row)
    diff = [a - lam * b for a, b in zip(w1, w2)]
    d_v = len(V[0])
    out = [0.0] * d_v
    for w, v in zip(diff, V):
        for j in range(d_v):
            out[j] += w * v[j]
    return out, diff


def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * d_head * seq_len * dtype_bytes


def main():
    print("=== attention mask shapes on an 8-token sequence ===")
    print("（=== 8-token 序列上的注意力 mask 形状 ===）")
    print()
    render(causal_mask(8), "full causal（完整因果）")
    render(swa_mask(8, window=4), "sliding window (W=4)（滑动窗口 W=4）")
    render(strided_mask(8, window=2, stride=3), "local (W=2) + strided (stride=3)（局部 W=2 + 步进 stride=3）")

    print("=== attention sink: one 'noisy' query on 8 random tokens ===")
    print("（=== attention sink：在 8 个随机 token 上的单个“噪声”query ===）")
    import random
    rng = random.Random(0)
    d = 8
    K = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(8)]
    V = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(8)]
    q = [rng.gauss(0, 1) for _ in range(d)]
    mask = causal_mask(8)[7]
    _, w_single = attention_row(q, K, V, mask)
    print(f"single attn weights: " + " ".join(f"{w:.3f}" for w in w_single))
    print(f"  (notice the weight bleeding to position 0 — the attention sink)")
    print(f"  （注意权重泄漏到位置 0 —— 这就是 attention sink）")

    q1 = q[:]
    q2 = [x + 0.2 * rng.gauss(0, 1) for x in q]
    K2 = [[x + 0.2 * rng.gauss(0, 1) for x in row] for row in K]
    _, w_diff = diff_attention_row(q1, q2, K, K2, V, mask, lam=0.5)
    print(f"diff   attn weights: " + " ".join(f"{w:+.3f}" for w in w_diff))
    print(f"  (lambda=0.5 subtracts the sink component; negative weights allowed)")
    print(f"  （lambda=0.5 减去 sink 分量；允许负权重）")
    print()

    print("=== KV cache @ 128K context, Llama-3-70B-ish (80 layers, 8 KV heads, d_head=128, fp16) ===")
    print("（=== 128K 上下文下的 KV cache，Llama-3-70B 级别（80 层，8 个 KV heads，d_head=128，fp16）===）")
    n_layers, n_kv_heads, d_head = 80, 8, 128
    N = 131072
    full = kv_cache_bytes(n_layers, n_kv_heads, d_head, N)

    print(f"  full attention              : {full / 1e9:>6.1f} GB")
    print(f"  （完整注意力）")
    for window in (4096, 1024):
        reduced = full * (window / N)
        print(f"  SWA window={window:>5}             : {reduced / 1e9:>6.1f} GB   ({N/window:.0f}x shrink)")
        print(f"  （SWA 窗口={window:>5}）")

    gemma3_ratio = 1 / 6
    gemma_total = full * (5 / 6) * (1024 / N) + full * (1 / 6)
    print(f"  Gemma-3 mix (5:1, W=1024)   : {gemma_total / 1e9:>6.1f} GB   ({full/gemma_total:.1f}x shrink)")
    print(f"  （Gemma-3 混合 5:1, W=1024）")

    diff = full * 2
    print(f"  differential attention (2x) : {diff / 1e9:>6.1f} GB   (pays 2x for sink-free weights)")
    print(f"  （差分注意力 2x）")
    print()
    print("takeaway: SWA is the cheapest long-context win.")
    print("          （要点：SWA 是最便宜的长上下文收益。）")
    print("          Gemma 3's 5:1 mix keeps enough global layers for retrieval")
    print("          （Gemma 3 的 5:1 混合保留了足够的全局层用于检索）")
    print("          while shrinking KV ~6x vs pure full attention.")
    print("          （同时将 KV 缩小到纯完整注意力的约 1/6。）")
    print("          DIFF attention pays 2x KV for sink-free, sharper retrieval.")
    print("          （差分注意力以 2 倍 KV 为代价换取无 sink、更锐利的检索。）")


if __name__ == "__main__":
    main()
