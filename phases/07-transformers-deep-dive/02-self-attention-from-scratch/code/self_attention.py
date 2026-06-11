import numpy as np


def softmax(x):
    # 为数值稳定性，先减去每行的最大值
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    # 缩放点积注意力：Q @ K^T / sqrt(dk)，softmax，然后 @ V
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights


class SelfAttention:
    # 单头自注意力，包含可学习的 Q/K/V 投影
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale_qk = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale_qk, (d_model, dk))
        self.Wk = rng.normal(0, scale_qk, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        # 通过权重矩阵投影得到 Q、K、V
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        return scaled_dot_product_attention(Q, K, V)


class MultiHeadSelfAttention:
    # 多头自注意力：并行运行多个注意力头，拼接后通过 Wo 投影
    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.dv = d_model // n_heads
        self.heads = [
            SelfAttention(d_model, self.dk, self.dv, seed=seed + i)
            for i in range(n_heads)
        ]
        rng = np.random.default_rng(seed + n_heads)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.Wo = rng.normal(0, scale, (n_heads * self.dv, d_model))

    def forward(self, X):
        head_outputs = []
        all_weights = []
        for head in self.heads:
            out, w = head.forward(X)
            head_outputs.append(out)
            all_weights.append(w)
        concatenated = np.concatenate(head_outputs, axis=-1)
        output = concatenated @ self.Wo
        return output, all_weights


def print_attention_matrix(weights, tokens):
    # 以表格形式打印注意力权重矩阵
    print(f"\n{'':>6}", end="")
    for token in tokens:
        print(f"{token:>6}", end="")
    print()
    for i, token in enumerate(tokens):
        print(f"{token:>6}", end="")
        for j in range(len(tokens)):
            print(f"{weights[i][j]:6.3f}", end="")
        print()


def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    # 使用 ASCII 字符将注意力权重可视化为热力图
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()
    w_max = weights.max()
    for i in range(len(tokens)):
        print(f"{tokens[i]:>6}", end="")
        for j in range(len(tokens)):
            level = int(weights[i][j] * (len(chars) - 1) / w_max)
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()


if __name__ == "__main__":
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    n_tokens = len(sentence)
    d_model = 16
    dk = 8
    dv = 8

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (n_tokens, d_model))

    print("=" * 60)
    print("SELF-ATTENTION FROM SCRATCH")
    print("=" * 60)

    print(f"\nSentence: {' '.join(sentence)}")
    print(f"Tokens: {n_tokens}, d_model: {d_model}, dk: {dk}, dv: {dv}")
    print(f"Input shape: {X.shape}")

    attn = SelfAttention(d_model, dk, dv, seed=42)
    output, weights = attn.forward(X)

    print(f"\nOutput shape: {output.shape}")
    print("\nAttention weights:")
    print_attention_matrix(weights, sentence)

    print("\nASCII heatmap (darker = higher attention):")
    ascii_heatmap(weights, sentence)

    print("\n" + "=" * 60)
    print("MULTI-HEAD SELF-ATTENTION")
    print("=" * 60)

    n_heads = 2
    mha = MultiHeadSelfAttention(d_model, n_heads, seed=42)
    mha_output, head_weights = mha.forward(X)

    print(f"\nHeads: {n_heads}")
    print(f"Output shape: {mha_output.shape}")

    for h, hw in enumerate(head_weights):
        print(f"\nHead {h + 1} attention weights:")
        print_attention_matrix(hw, sentence)

    print("\n" + "=" * 60)
    print("SOFTMAX DEMO")
    print("=" * 60)

    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print(f"\nLogits:  {logits}")
    print(f"Softmax: {probs.round(4)}")
    print(f"Sum:     {probs.sum():.4f}")

    large_logits = np.array([100.0, 200.0, 300.0])
    probs_large = softmax(large_logits)
    print(f"\nLarge logits:  {large_logits}")
    print(f"Softmax:       {probs_large.round(4)}")
    print(f"Sum:           {probs_large.sum():.4f}")
    print("(Numerically stable - no overflow)")
