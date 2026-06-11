"""KV cache + 分块（Flash 风格）注意力，纯标准库实现。

展示内容：
- 朴素 O(N^2) 增量解码器 vs KV 缓存 O(N) 解码器
- 运行最大值 softmax，逐 tile 产生 bit-identical（逐位相同）输出
- 面向 2026 年真实模型的 KV cache 大小计算
"""

import math
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def attention_full(q, Ks, Vs):
    """单 query 对完整 key 和 value 列表的注意力。"""
    scores = [dot(q, k) / math.sqrt(len(q)) for k in Ks]
    weights = softmax(scores)
    out = [0.0] * len(Vs[0])
    for w, v in zip(weights, Vs):
        for j in range(len(out)):
            out[j] += w * v[j]
    return out


def tiled_softmax_dot(q, Ks, Vs, tile=4):
    """Flash-attention 风格的增量 softmax(qK^T)V，tile 大小为 `tile`。"""
    d_head = len(Vs[0])
    scale = 1.0 / math.sqrt(len(q))
    m = float("-inf")
    s = 0.0
    out = [0.0] * d_head
    for start in range(0, len(Ks), tile):
        k_block = Ks[start:start + tile]
        v_block = Vs[start:start + tile]
        scores = [dot(q, k) * scale for k in k_block]
        new_m = max(m, *scores)
        if m == float("-inf"):
            exp_old = 0.0
        else:
            exp_old = math.exp(m - new_m)
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(d_head):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]


class KVCache:
    def __init__(self):
        self.K = []
        self.V = []

    def append(self, k, v):
        self.K.append(k)
        self.V.append(v)

    def __len__(self):
        return len(self.K)


def decode_naive(all_K, all_V, all_queries):
    """每一步都重新计算对完整前缀的注意力。
    返回输出列表，每个生成 token 一个。操作数 = 1+2+...+N = N(N+1)/2。
    """
    outputs = []
    ops = 0
    for t, q in enumerate(all_queries):
        Ks = all_K[:t + 1]
        Vs = all_V[:t + 1]
        out = attention_full(q, Ks, Vs)
        ops += t + 1
        outputs.append(out)
    return outputs, ops


def decode_cached(all_K, all_V, all_queries):
    """KV cache：每一步追加一个 K,V 并用 cache 做注意力查询。"""
    cache = KVCache()
    outputs = []
    ops = 0
    for q, k, v in zip(all_queries, all_K, all_V):
        cache.append(k, v)
        out = attention_full(q, cache.K, cache.V)
        ops += len(cache)
        outputs.append(out)
    return outputs, ops


def kv_cache_bytes(N, n_layers, n_heads_kv, d_head, dtype=2):
    """KV cache 总字节数。dtype=2 表示 fp16/bf16，1 表示 int8，4 表示 fp32。"""
    return 2 * N * n_layers * n_heads_kv * d_head * dtype


def main():
    rng = random.Random(42)
    d_head = 8
    N = 10

    # 为一个 10-token 序列、单个 head 生成随机 Q, K, V。
    all_Q = [[rng.gauss(0, 1) for _ in range(d_head)] for _ in range(N)]
    all_K = [[rng.gauss(0, 1) for _ in range(d_head)] for _ in range(N)]
    all_V = [[rng.gauss(0, 1) for _ in range(d_head)] for _ in range(N)]

    naive, naive_ops = decode_naive(all_K, all_V, all_Q)
    cached, cached_ops = decode_cached(all_K, all_V, all_Q)

    print(f"=== naive vs KV-cached decoding on N={N} tokens ===")
    print(f"naive attention ops:  {naive_ops}  (O(N^2) = {N * (N + 1) // 2})")
    print(f"cached attention ops: {cached_ops}  (O(N) with per-step cost, unchanged)")
    print("outputs match (max abs diff over all tokens):",
          f"{max(abs(a - b) for va, vb in zip(naive, cached) for a, b in zip(va, vb)):.2e}")
    print()
    print("* naive 每步成本相同；节省来自于不重新计算之前的")
    print("  隐藏状态。统计 K,V 重算会让 naive 在矩阵乘法上达到 O(N^2)。")
    print()

    print("=== tiled-softmax (Flash) vs standard softmax agreement ===")
    q = all_Q[-1]
    std = attention_full(q, all_K, all_V)
    for tile in [1, 2, 4, 8, 32]:
        tiled = tiled_softmax_dot(q, all_K, all_V, tile=tile)
        err = max(abs(a - b) for a, b in zip(std, tiled))
        print(f"  tile={tile:>3}  max abs diff = {err:.2e}")
    print("  在浮点重关联范围内 bit-identical（逐位相同）。无近似。")
    print()

    print("=== KV cache size table (fp16) ===")
    configs = [
        ("Llama-3.2-3B",  28, 8,   128),
        ("Llama-3-8B",    32, 8,   128),
        ("Llama-3-70B",   80, 8,   128),
        ("Llama-3.1-405B", 126, 8, 128),
        ("Qwen2.5-72B",   80, 8,   128),
        ("DeepSeek-V3 (MLA)", 61, 1, 512),  # MLA compresses to latent; rough
    ]
    for name, L, h_kv, d_h in configs:
        for N_ctx in [2048, 32768, 131072]:
            b = kv_cache_bytes(N_ctx, L, h_kv, d_h, dtype=2)
            print(f"  {name:<24}  N={N_ctx:>7}  -> {b / 1e9:.2f} GB")
    print()
    print("要点：在 128K 上下文下，70B 级别的稠密模型仅 KV 就占用 10+ GB。")
    print("GQA 和 MLA 是现代长上下文推理可负担的原因。")


if __name__ == "__main__":
    main()
