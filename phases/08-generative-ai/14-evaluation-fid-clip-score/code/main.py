import math
import random


def mean_vec(vectors):
    d = len(vectors[0])
    n = len(vectors)
    return [sum(v[i] for v in vectors) / n for i in range(d)]


def covariance(vectors, mu):
    d = len(mu)
    n = len(vectors)
    cov = [[0.0] * d for _ in range(d)]
    for v in vectors:
        for i in range(d):
            for j in range(d):
                cov[i][j] += (v[i] - mu[i]) * (v[j] - mu[j])
    return [[cov[i][j] / max(n - 1, 1) for j in range(d)] for i in range(d)]


def trace(M):
    return sum(M[i][i] for i in range(len(M)))


def matmul(A, B):
    n = len(A)
    p = len(B[0])
    m = len(B)
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                out[i][j] += A[i][k] * B[k][j]
    return out


def jacobi_sqrt(M, iters=30):
    """Matrix square root by Denman-Beavers iteration (stable for PSD M)."""
    n = len(M)
    Y = [row[:] for row in M]
    Z = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iters):
        Y_inv = inverse(Y)
        Z_inv = inverse(Z)
        Y = [[(Y[i][j] + Z_inv[i][j]) / 2 for j in range(n)] for i in range(n)]
        Z = [[(Z[i][j] + Y_inv[i][j]) / 2 for j in range(n)] for i in range(n)]
    return Y


def inverse(M):
    n = len(M)
    A = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(A[r][col]) > abs(A[pivot][col]):
                pivot = r
        A[col], A[pivot] = A[pivot], A[col]
        piv = A[col][col]
        if abs(piv) < 1e-12:
            piv = 1e-12
        for j in range(2 * n):
            A[col][j] /= piv
        for r in range(n):
            if r == col: continue
            factor = A[r][col]
            for j in range(2 * n):
                A[r][j] -= factor * A[col][j]
    return [row[n:] for row in A]


def fid(real_features, gen_features):
    mu_r = mean_vec(real_features)
    mu_g = mean_vec(gen_features)
    cov_r = covariance(real_features, mu_r)
    cov_g = covariance(gen_features, mu_g)
    mean_sq = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    prod = matmul(cov_r, cov_g)
    sqrt_prod = jacobi_sqrt(prod)
    return mean_sq + trace(cov_r) + trace(cov_g) - 2 * trace(sqrt_prod)


def clip_like(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / max(na * nb, 1e-8)


def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    delta = k * (actual_a - expected_a)
    return r_a + delta, r_b - delta


def make_features(center, n, d, rng, scale=0.4):
    return [[center + rng.gauss(0, scale) for _ in range(d)] for _ in range(n)]


def main():
    rng = random.Random(29)
    d = 4

    print("=== 小 N 下的 FID 偏差 ===")
    for n in [50, 200, 1000]:
        real = make_features(0.0, n, d, rng)
        gen = make_features(0.0, n, d, rng)  # same distribution
        score = fid(real, gen)
        print(f"  N={n:5d}: FID (identical distributions) = {score:.4f}  (越低 = 越相似)")

    print("  -> 相同分布的 FID 应为 0，但在小 N 下存在向上偏差")
    print()

    print("=== FID 区分不同分布 ===")
    real = make_features(0.0, 500, d, rng)
    for shift in [0.0, 0.2, 0.5, 1.0]:
        gen = make_features(shift, 500, d, rng)
        score = fid(real, gen)
        print(f"  shift={shift:.1f}: FID = {score:.3f}")

    print()
    print("=== 类 CLIP 余弦相似度 ===")
    prompt = [1.0, 0.5, -0.2, 0.3]
    for image_center in [1.0, 0.5, 0.0, -0.5]:
        image = [image_center + rng.gauss(0, 0.1) for _ in range(d)]
        score = clip_like(image, prompt)
        print(f"  image center {image_center:+.1f}: CLIP-like score = {score:+.3f}")

    print()
    print("=== 合成 A/B 偏好的 Elo ===")
    r_a, r_b = 1000, 1000
    for i in range(200):
        # 假设模型 A 70% 的时间获胜
        winner = "a" if rng.random() < 0.7 else "b"
        r_a, r_b = elo_update(r_a, r_b, winner)
    print(f"  200 对之后 (A 获胜 70%): r_A = {r_a:.0f}, r_B = {r_b:.0f}")

    print()
    print("要点: FID 是距离; CLIP 是遵循度分数; Elo 聚合偏好。")
    print("          生产评估使用全部三种加上定性失效审计。")


if __name__ == "__main__":
    main()
