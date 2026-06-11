"""玩具视觉自回归 (VAR) 模型：在金字塔上进行下一尺度预测。

docs/en.md 中描述的 VAR 机制的最小 numpy 实现。包含三个部分：

1. 在微型 8×8 "图像"（一个小型图案库：纯色、渐变、环形、棋盘、十字）上的多尺度残差 VQ 分词器。
   尺度 k 的 token 编码了尺度 1..k-1 遗留的残差。解码器是上采样尺度 embedding 的和。
2. 尺度条件的下一尺度预测器（在小词表上的 logistic / softmax 迷你 LM）。
   "transformer" 用每尺度条件直方图近似；本课教授的几何结构是尺度顺序条件和尺度内并行预测，
   而不是深层注意力。
3. 运行 K 次 transformer 传播（每尺度一次）的生成循环，并从条件中并行采样当前尺度的每个位置。
   解码后的尺度 embedding 之和重构图像。

重点是练习尺度顺序训练数据、尺度内并行采样和残差-VQ 重构。
   真实 VAR 将直方图替换为 transformer，将图案库替换为图像数据集；围绕它们的 harness 保持不变。

仅使用 stdlib + numpy。

运行:
    python main.py
"""

from __future__ import annotations

import numpy as np


IMG = 8
SCALES = (1, 2, 4, 8)
CODEBOOK = 16


def make_patterns(rng: np.random.Generator, n: int) -> np.ndarray:
    """返回从微型库中抽取的 n 个灰度 8×8 图案。"""
    out = np.zeros((n, IMG, IMG), dtype=np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32)
    for i in range(n):
        kind = int(rng.integers(0, 5))
        if kind == 0:
            out[i] = rng.uniform(0.1, 0.9)
        elif kind == 1:
            out[i] = (xx + yy) / (2 * (IMG - 1))
        elif kind == 2:
            cx, cy = IMG / 2 - 0.5, IMG / 2 - 0.5
            r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            out[i] = np.clip(1.0 - r / (IMG / 2), 0.0, 1.0)
        elif kind == 3:
            out[i] = ((xx.astype(int) + yy.astype(int)) % 2).astype(np.float32)
        else:
            mid = IMG // 2
            cross = ((xx == mid) | (yy == mid)).astype(np.float32)
            out[i] = cross * 0.9 + 0.05
    return out


def fit_codebook(samples: np.ndarray, k: int, iters: int = 30,
                 seed: int = 0) -> np.ndarray:
    """对标量样本进行 k-means；返回长度为 k 的码本。"""
    rng = np.random.default_rng(seed)
    flat = samples.reshape(-1)
    if flat.size < k:
        raise ValueError(f"need >= {k} samples for codebook init, got {flat.size}")
    idx = rng.choice(flat.size, size=k, replace=False)
    centers = flat[idx].astype(np.float32)
    for _ in range(iters):
        dists = (flat[:, None] - centers[None, :]) ** 2
        assign = dists.argmin(axis=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers[j] = flat[mask].mean()
    return np.sort(centers)


def encode(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """将每个值吸附到最近的码；返回整数 token。"""
    dists = (values[..., None] - codebook[None, None, :]) ** 2
    return dists.argmin(axis=-1).astype(np.int32)


def downsample(img: np.ndarray, target: int) -> np.ndarray:
    """将 H×W 图像平均池化降采样到 target × target。"""
    h, w = img.shape
    if target == h:
        return img.copy()
    factor = h // target
    return img.reshape(target, factor, target, factor).mean(axis=(1, 3))


def upsample(grid: np.ndarray, target: int) -> np.ndarray:
    """最近邻上采样将 H×W 网格放大到 target × target。"""
    h, w = grid.shape
    if target == h:
        return grid.copy()
    factor = target // h
    return grid.repeat(factor, axis=0).repeat(factor, axis=1)


def tokenize_multiscale(img: np.ndarray, codebooks: list[np.ndarray]
                        ) -> list[np.ndarray]:
    """残差 VQ：每个尺度对之前尺度遗漏的内容进行分词。"""
    residual = img.copy()
    tokens: list[np.ndarray] = []
    for scale, book in zip(SCALES, codebooks):
        coarse = downsample(residual, scale)
        tok = encode(coarse, book)
        recon = book[tok]
        residual = residual - upsample(recon, IMG)
        tokens.append(tok)
    return tokens


def detokenize_multiscale(tokens: list[np.ndarray],
                          codebooks: list[np.ndarray]) -> np.ndarray:
    """解码器：累加上采样尺度 embedding。"""
    out = np.zeros((IMG, IMG), dtype=np.float32)
    for tok, book, scale in zip(tokens, codebooks, SCALES):
        out = out + upsample(book[tok], IMG)
    return out


def train_codebooks(images: np.ndarray) -> list[np.ndarray]:
    """在小图像集的残差上拟合每尺度码本。"""
    residuals = images.copy()
    books: list[np.ndarray] = []
    for scale in SCALES:
        pooled = np.stack([downsample(r, scale) for r in residuals])
        book = fit_codebook(pooled, CODEBOOK)
        books.append(book)
        recon = np.stack([upsample(book[encode(p[None], book)[0]], IMG)
                          for p in pooled])
        residuals = residuals - recon
    return books


def context_key(prev_tokens: list[np.ndarray]) -> tuple:
    """所有之前尺度 token 的可哈希摘要。"""
    return tuple(int(t.mean() * 1000) for t in prev_tokens) if prev_tokens else ()


def fit_predictor(token_streams: list[list[np.ndarray]]
                  ) -> list[dict[tuple, np.ndarray]]:
    """每尺度一个条件直方图，以之前尺度的摘要为键。

    这充当 transformer 的替身：训练时，统计在尺度 1..k-1 的粗化摘要条件下，
    哪些 token 出现在尺度 k。
    """
    predictors: list[dict[tuple, np.ndarray]] = [
        {} for _ in SCALES
    ]
    for stream in token_streams:
        for k in range(len(SCALES)):
            ctx = context_key(stream[:k])
            table = predictors[k].setdefault(ctx, np.ones(CODEBOOK,
                                                          dtype=np.float64))
            for tok in stream[k].reshape(-1):
                table[int(tok)] += 1.0
    for table in predictors:
        for key, counts in table.items():
            table[key] = counts / counts.sum()
    return predictors


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(probs), p=probs))


def generate(predictors: list[dict[tuple, np.ndarray]],
             codebooks: list[np.ndarray],
             rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
    """一个 VAR 样本：K 次传播，尺度内并行，尺度间因果。"""
    drawn: list[np.ndarray] = []
    for k, scale in enumerate(SCALES):
        ctx = context_key(drawn[:k])
        table = predictors[k]
        probs = table.get(ctx)
        if probs is None:
            probs = np.ones(CODEBOOK) / CODEBOOK
        size = scale * scale
        flat = np.array([sample_categorical(probs, rng) for _ in range(size)],
                        dtype=np.int32)
        drawn.append(flat.reshape(scale, scale))
    image = detokenize_multiscale(drawn, codebooks)
    return image, drawn


def reconstruction_mse(images: np.ndarray,
                       codebooks: list[np.ndarray]) -> float:
    errs = []
    for img in images:
        toks = tokenize_multiscale(img, codebooks)
        recon = detokenize_multiscale(toks, codebooks)
        errs.append(float(np.mean((recon - img) ** 2)))
    return float(np.mean(errs))


def main() -> None:
    rng = np.random.default_rng(0)
    train_imgs = make_patterns(rng, 64)
    val_imgs = make_patterns(rng, 16)

    codebooks = train_codebooks(train_imgs)
    train_token_streams = [tokenize_multiscale(img, codebooks) for img in train_imgs]
    predictors = fit_predictor(train_token_streams)

    print(f"图像尺寸: {IMG}x{IMG}")
    print(f"尺度: {SCALES}")
    print(f"每尺度码本大小: {CODEBOOK}")
    print(f"训练集重建 MSE: {reconstruction_mse(train_imgs, codebooks):.5f}")
    print(f"验证集重建 MSE:   {reconstruction_mse(val_imgs, codebooks):.5f}")

    print()
    print("生成: 4 次 transformer 传播，尺度内所有位置并行")
    for trial in range(3):
        img, toks = generate(predictors, codebooks, rng)
        shapes = [t.shape for t in toks]
        print(f"  试验 {trial}: scales={shapes}  range=[{img.min():.2f}, {img.max():.2f}]")

    print()
    print("尺度顺序注意力检查: 每个尺度 k 只能看到尺度 1..k-1")
    for k, scale in enumerate(SCALES):
        n_pos = scale * scale
        prior_seen = sum(s * s for s in SCALES[:k])
        print(f"  尺度 {k} (尺寸 {scale}x{scale}, {n_pos} 个 token):"
              f" 关注 {prior_seen} 个先前 token")


if __name__ == "__main__":
    main()
