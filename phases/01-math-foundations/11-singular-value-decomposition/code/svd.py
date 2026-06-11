import numpy as np


def power_iteration(M, num_iters=200, tol=1e-10):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        norm = np.linalg.norm(Mv)
        if norm < tol:
            return 0.0, v
        v_new = Mv / norm
        if np.abs(np.dot(v_new, v)) > 1 - tol:
            v = v_new
            break
        v = v_new

    eigenvalue = v @ M @ v
    return eigenvalue, v


def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for i in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=300)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(max(eigenvalue, 0))
        u = A_residual @ v / sigma

        u_norm = np.linalg.norm(u)
        if u_norm > 1e-10:
            u = u / u_norm

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V


def truncated_svd(A, k):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]


def reconstruct(U, S, Vt):
    return U @ np.diag(S) @ Vt


def compression_ratio(m, n, k):
    original = m * n
    compressed = k * (m + n + 1)
    return compressed / original


def pseudoinverse_via_svd(A, tol=1e-10):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.array([1.0 / s if s > tol else 0.0 for s in S])
    return Vt.T @ np.diag(S_inv) @ U.T


def demo_svd_basics():
    print("=" * 70)
    print("从零实现 SVD 对比 NumPy")
    print("=" * 70)

    np.random.seed(42)
    A = np.random.randn(6, 4)

    print(f"\n矩阵 A 形状: {A.shape}")
    print(f"矩阵 A:\n{np.round(A, 4)}")

    U_ours, S_ours, V_ours = svd_from_scratch(A)
    U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

    print(f"\n我们的 singular values:   {np.round(S_ours, 4)}")
    print(f"NumPy singular values: {np.round(S_np, 4)}")

    A_ours = U_ours @ np.diag(S_ours) @ V_ours.T
    A_np = U_np @ np.diag(S_np) @ Vt_np

    err_ours = np.linalg.norm(A - A_ours)
    err_np = np.linalg.norm(A - A_np)
    print(f"\n重构误差 (ours):  {err_ours:.10f}")
    print(f"重构误差 (NumPy): {err_np:.10f}")

    print("\n验证 A @ v_i = sigma_i * u_i:")
    for i in range(min(4, len(S_np))):
        v_i = Vt_np[i]
        u_i = U_np[:, i]
        lhs = A @ v_i
        rhs = S_np[i] * u_i
        match = np.allclose(lhs, rhs, atol=1e-10) or np.allclose(lhs, -rhs, atol=1e-10)
        print(f"  i={i}: sigma={S_np[i]:.4f}, match={match}")

    print()


def demo_geometry():
    print("=" * 70)
    print("SVD 几何含义：旋转、缩放、旋转")
    print("=" * 70)

    A = np.array([[3.0, 1.0],
                  [1.0, 3.0]])

    U, S, Vt = np.linalg.svd(A)

    print(f"\n矩阵 A:\n{A}")
    print(f"\nU (左旋转):\n{np.round(U, 4)}")
    print(f"Sigma (缩放): {np.round(S, 4)}")
    print(f"V^T (右旋转):\n{np.round(Vt, 4)}")

    print("\n验证 U 正交 (U^T U = I):")
    print(f"  {np.round(U.T @ U, 6)}")

    print("验证 V 正交 (V^T V = I):")
    print(f"  {np.round(Vt @ Vt.T, 6)}")

    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])

    print("\n单位圆点经过每个 SVD 阶段:")
    print(f"  {'Point':>8s}  {'V^T(p)':>12s}  {'Sig*V^T(p)':>14s}  {'U*Sig*V^T(p)':>16s}")
    for i in range(len(theta)):
        p = circle[i]
        step1 = Vt @ p
        step2 = S * step1
        step3 = U @ step2
        direct = A @ p
        print(f"  ({p[0]:5.2f},{p[1]:5.2f})  "
              f"({step1[0]:5.2f},{step1[1]:5.2f})  "
              f"({step2[0]:6.2f},{step2[1]:6.2f})  "
              f"({step3[0]:6.2f},{step3[1]:6.2f})  "
              f"check=({direct[0]:6.2f},{direct[1]:6.2f})")

    print()


def demo_low_rank_approximation():
    print("=" * 70)
    print("低秩近似 (ECKART-YOUNG)")
    print("=" * 70)

    np.random.seed(42)
    m, n, true_rank = 100, 80, 5

    U_true = np.linalg.qr(np.random.randn(m, true_rank))[0]
    V_true = np.linalg.qr(np.random.randn(n, true_rank))[0]
    S_true = np.array([50, 30, 15, 8, 3], dtype=float)
    A = U_true @ np.diag(S_true) @ V_true.T

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"\n矩阵形状: {A.shape}, 真实秩: {true_rank}")
    print(f"前 10 个 singular values: {np.round(S[:10], 4)}")
    print(f"  (第 6-10 个应接近 0，因为真实秩为 5)")

    print(f"\n{'k':>3s}  {'误差':>10s}  {'相对误差':>10s}  {'压缩比':>8s}")
    print("-" * 40)
    A_norm = np.linalg.norm(A, 'fro')
    for k in range(1, 8):
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        err = np.linalg.norm(A - A_k, 'fro')
        rel = err / A_norm
        ratio = compression_ratio(m, n, k)
        print(f"{k:3d}  {err:10.4f}  {rel:10.6f}  {ratio:7.1%}")

    print()


def demo_image_compression():
    print("=" * 70)
    print("使用 SVD 进行图像压缩")
    print("=" * 70)

    np.random.seed(42)
    rows, cols = 256, 256

    x = np.linspace(-3, 3, cols)
    y = np.linspace(-3, 3, rows)
    X, Y = np.meshgrid(x, y)
    image = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2 * X + Y)
    image = (image - image.min()) / (image.max() - image.min()) * 255

    print(f"\n合成图像: {rows}x{cols} = {rows * cols:,} 个值")

    U, S, Vt = np.linalg.svd(image, full_matrices=False)

    print(f"\nSingular value 谱:")
    print(f"  sigma_1   = {S[0]:.2f}")
    print(f"  sigma_5   = {S[4]:.2f}")
    print(f"  sigma_10  = {S[9]:.2f}")
    print(f"  sigma_50  = {S[49]:.2f}")
    print(f"  sigma_100 = {S[99]:.2f}")
    print(f"  sigma_256 = {S[255]:.6f}")

    total_energy = np.sum(S ** 2)
    print(f"\n压缩结果:")
    print(f"{'k':>5s}  {'存储量':>10s}  {'压缩比':>8s}  {'能量':>10s}  {'RMSE':>8s}")
    print("-" * 50)

    for k in [1, 2, 5, 10, 20, 50, 100, 200]:
        compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        storage = k * (rows + cols + 1)
        ratio = storage / (rows * cols)
        energy = np.sum(S[:k] ** 2) / total_energy
        rmse = np.sqrt(np.mean((image - compressed) ** 2))
        print(f"{k:5d}  {storage:10,d}  {ratio:7.1%}  {energy:9.4%}  {rmse:8.4f}")

    print()


def demo_recommendation_system():
    print("=" * 70)
    print("SVD 用于推荐系统")
    print("=" * 70)

    np.random.seed(42)

    n_users = 10
    n_movies = 8
    n_factors = 3

    user_prefs = np.random.randn(n_users, n_factors)
    movie_attrs = np.random.randn(n_movies, n_factors)

    true_ratings = user_prefs @ movie_attrs.T
    true_ratings = (true_ratings - true_ratings.min()) / (true_ratings.max() - true_ratings.min()) * 4 + 1
    true_ratings = np.round(true_ratings, 1)

    mask = np.random.random((n_users, n_movies)) > 0.4
    observed = true_ratings.copy()
    observed[~mask] = np.nan

    print(f"\n评分矩阵 ({n_users} 用户 x {n_movies} 电影):")
    print("  观测评分 (? = 缺失):")
    for i in range(n_users):
        row = "  "
        for j in range(n_movies):
            if mask[i, j]:
                row += f"{observed[i, j]:5.1f}"
            else:
                row += "    ?"
        print(row)

    filled = observed.copy()
    for i in range(n_users):
        row_mean = np.nanmean(filled[i])
        filled[i, np.isnan(filled[i])] = row_mean

    U, S, Vt = np.linalg.svd(filled, full_matrices=False)

    k = n_factors
    predicted = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    print(f"\nRank-{k} SVD 对缺失条目的预测:")
    errors = []
    for i in range(n_users):
        for j in range(n_movies):
            if not mask[i, j]:
                err = abs(predicted[i, j] - true_ratings[i, j])
                errors.append(err)
                print(f"  用户 {i}, 电影 {j}: "
                      f"预测={predicted[i, j]:.2f}, "
                      f"真实={true_ratings[i, j]:.1f}, "
                      f"误差={err:.2f}")

    print(f"\n缺失评分的平均绝对误差: {np.mean(errors):.3f}")

    print(f"\nLatent factors (前 {k} 个 singular values): {np.round(S[:k], 2)}")
    print(f"剩余 singular values: {np.round(S[k:], 2)}")
    energy_captured = np.sum(S[:k] ** 2) / np.sum(S ** 2)
    print(f"Rank-{k} 捕获的能量: {energy_captured:.1%}")

    print()


def demo_lsa():
    print("=" * 70)
    print("潜在语义分析 (LSA)")
    print("=" * 70)

    terms = ["cat", "dog", "fish", "kitten", "puppy",
             "ocean", "sea", "water", "bark", "meow",
             "swim", "pet", "fur", "fin", "paw"]

    docs = [
        "The cat and kitten have soft fur and paws. The cat likes to meow.",
        "The dog and puppy like to bark. Dogs have fur and paws.",
        "Fish swim in the ocean and sea. Fish have fins and swim in water.",
        "The pet cat meows while the pet dog barks.",
        "Ocean water is where fish swim. The sea has many fish.",
        "The kitten and puppy are small pets with fur and paws.",
    ]

    doc_labels = ["cat_doc", "dog_doc", "fish_doc", "pet_doc", "ocean_doc", "mixed_doc"]

    n_terms = len(terms)
    n_docs = len(docs)
    td_matrix = np.zeros((n_terms, n_docs))

    for j, doc in enumerate(docs):
        doc_lower = doc.lower()
        for i, term in enumerate(terms):
            td_matrix[i, j] = doc_lower.count(term)

    print(f"\nTerm-Document 矩阵 ({n_terms} 词 x {n_docs} 文档):")
    header = "          " + "".join(f"{dl:>10s}" for dl in doc_labels)
    print(header)
    for i, term in enumerate(terms):
        row = f"{term:>10s}" + "".join(f"{td_matrix[i, j]:10.0f}" for j in range(n_docs))
        print(row)

    U, S, Vt = np.linalg.svd(td_matrix, full_matrices=False)

    print(f"\nSingular values: {np.round(S, 3)}")

    k = 3
    print(f"\n{k}D 潜在空间中的文档 (V_k^T 的行按 Sigma_k 缩放):")
    doc_coords = np.diag(S[:k]) @ Vt[:k, :]
    for j in range(n_docs):
        coords = doc_coords[:, j]
        print(f"  {doc_labels[j]:>10s}: [{coords[0]:7.3f}, {coords[1]:7.3f}, {coords[2]:7.3f}]")

    print(f"\n{k}D 潜在空间中的词 (U_k 的行按 Sigma_k 缩放):")
    term_coords = U[:, :k] @ np.diag(S[:k])
    for i in range(n_terms):
        coords = term_coords[i]
        print(f"  {terms[i]:>10s}: [{coords[0]:7.3f}, {coords[1]:7.3f}, {coords[2]:7.3f}]")

    print(f"\n文档相似度 (潜在空间中的余弦相似度):")
    doc_vecs = Vt[:k, :].T
    header = "          " + "".join(f"{dl:>10s}" for dl in doc_labels)
    print(header)
    for i in range(n_docs):
        row = f"{doc_labels[i]:>10s}"
        for j in range(n_docs):
            cos_sim = np.dot(doc_vecs[i], doc_vecs[j]) / (
                np.linalg.norm(doc_vecs[i]) * np.linalg.norm(doc_vecs[j]) + 1e-10
            )
            row += f"{cos_sim:10.3f}"
        print(row)

    print()


def demo_noise_reduction():
    print("=" * 70)
    print("SVD 用于降噪")
    print("=" * 70)

    np.random.seed(42)
    m, n = 100, 80

    t1 = np.linspace(0, 4 * np.pi, m)
    t2 = np.linspace(0, 2 * np.pi, n)
    clean = (5 * np.outer(np.sin(t1), np.cos(t2))
             + 3 * np.outer(np.cos(2 * t1), np.sin(t2))
             + 2 * np.outer(np.ones(m), np.sin(3 * t2)))

    print(f"\n干净信号: 秩 {np.linalg.matrix_rank(clean)}, 形状 {clean.shape}")

    noise_levels = [0.1, 0.5, 1.0, 2.0]
    clean_norm = np.linalg.norm(clean, 'fro')

    for noise_std in noise_levels:
        noise = noise_std * np.random.randn(m, n)
        noisy = clean + noise

        U, S, Vt = np.linalg.svd(noisy, full_matrices=False)

        noisy_err = np.linalg.norm(noisy - clean, 'fro') / clean_norm

        print(f"\n  噪声水平 sigma={noise_std}:")
        print(f"    含噪声相对误差: {noisy_err:.4f}")
        print(f"    前 10 个 singular values: {np.round(S[:10], 2)}")

        best_k = 1
        best_err = float('inf')
        for k in range(1, min(m, n)):
            denoised = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            err = np.linalg.norm(denoised - clean, 'fro') / clean_norm
            if err < best_err:
                best_err = err
                best_k = k

        denoised = U[:, :best_k] @ np.diag(S[:best_k]) @ Vt[:best_k, :]
        improvement = 1 - best_err / noisy_err

        print(f"    最佳截断秩: k={best_k}")
        print(f"    降噪后相对误差: {best_err:.4f}")
        print(f"    改善程度: {improvement:.1%}")

    print()


def demo_pseudoinverse():
    print("=" * 70)
    print("通过 SVD 计算伪逆")
    print("=" * 70)

    print("\n--- 超定系统 (最小二乘) ---")
    A = np.array([[1, 1],
                  [2, 1],
                  [3, 1]], dtype=float)
    b = np.array([3.0, 5.0, 6.0])

    print(f"A:\n{A}")
    print(f"b: {b}")
    print("(3 个方程, 2 个未知数, 无精确解)")

    A_pinv = pseudoinverse_via_svd(A)
    x_svd = A_pinv @ b
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    x_normal = np.linalg.solve(A.T @ A, A.T @ b)

    print(f"\nSVD 伪逆解:     {np.round(x_svd, 6)}")
    print(f"np.linalg.lstsq 解:       {np.round(x_lstsq, 6)}")
    print(f"正规方程解:       {np.round(x_normal, 6)}")

    residual = A @ x_svd - b
    print(f"残差 (A x - b): {np.round(residual, 6)}")
    print(f"残差范数: {np.linalg.norm(residual):.6f}")

    print("\n--- 欠定系统 (最小范数) ---")
    A2 = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=float)
    b2 = np.array([14.0, 32.0])

    print(f"A:\n{A2}")
    print(f"b: {b2}")
    print("(2 个方程, 3 个未知数, 无穷多解)")

    A2_pinv = pseudoinverse_via_svd(A2)
    x_min_norm = A2_pinv @ b2
    x_lstsq2 = np.linalg.lstsq(A2, b2, rcond=None)[0]

    print(f"\nSVD 最小范数解:  {np.round(x_min_norm, 6)}")
    print(f"np.linalg.lstsq 解:   {np.round(x_lstsq2, 6)}")
    print(f"验证 A x = b: {np.round(A2 @ x_min_norm, 6)}")
    print(f"解范数: {np.linalg.norm(x_min_norm):.6f}")

    print("\n--- 奇异矩阵 ---")
    A3 = np.array([[1, 2],
                   [2, 4]], dtype=float)
    b3 = np.array([3.0, 6.0])

    print(f"A:\n{A3}")
    print(f"b: {b3}")
    print("(奇异矩阵, 秩 1)")

    U, S, Vt = np.linalg.svd(A3, full_matrices=False)
    print(f"Singular values: {np.round(S, 6)}")

    A3_pinv = pseudoinverse_via_svd(A3)
    x_pinv = A3_pinv @ b3
    print(f"伪逆解: {np.round(x_pinv, 6)}")
    print(f"验证 A x = b: {np.round(A3 @ x_pinv, 6)}")
    print(f"解范数: {np.linalg.norm(x_pinv):.6f}")

    print()


def demo_condition_number():
    print("=" * 70)
    print("条件数与数值稳定性")
    print("=" * 70)

    matrices = [
        ("良态", np.array([[2.0, 1.0], [1.0, 2.0]])),
        ("中等", np.array([[10.0, 7.0], [7.0, 5.0]])),
        ("病态", np.array([[1.0, 1.0], [1.0, 1.0001]])),
        ("接近奇异", np.array([[1.0, 2.0], [0.5, 1.00001]])),
    ]

    print(f"\n{'名称':>20s}  {'sigma_max':>10s}  {'sigma_min':>10s}  {'条件数':>12s}")
    print("-" * 58)

    for name, A in matrices:
        U, S, Vt = np.linalg.svd(A)
        cond = S[0] / S[-1] if S[-1] > 1e-15 else float('inf')
        print(f"{name:>20s}  {S[0]:10.4f}  {S[-1]:10.6f}  {cond:12.1f}")

    print("\n为什么重要:")
    print("  条件数 K 意味着：输入中大小为 eps 的扰动")
    print("  可能导致输出中大小为 K * eps 的扰动。")
    print("  K = 10^6 意味着你损失 6 位精度。")
    print()

    print("对比 SVD 与 eigendecomposition 的稳定性:")
    A = np.array([[1.0, 1.0], [1.0, 1.0001]])
    U, S, Vt = np.linalg.svd(A)
    AtA = A.T @ A
    eig_vals = np.linalg.eigvalsh(AtA)

    print(f"  A 的 singular values:     {S}")
    print(f"  A 的条件数:    {S[0] / S[-1]:.1f}")
    print(f"  A^T A 的 eigenvalues:     {eig_vals}")
    print(f"  A^T A 的条件数: {eig_vals[-1] / eig_vals[0]:.1f}")
    print(f"  (平方了! 直接 SVD 避免了这个问题。)")

    print()


def demo_pca_is_svd():
    print("=" * 70)
    print("PCA 就是对中心化数据做 SVD")
    print("=" * 70)

    np.random.seed(42)
    n_samples = 200
    n_features = 5

    mean = np.array([10, 20, 30, 40, 50], dtype=float)
    cov = np.array([
        [5.0, 2.0, 1.0, 0.5, 0.1],
        [2.0, 4.0, 1.5, 0.3, 0.2],
        [1.0, 1.5, 3.0, 0.8, 0.4],
        [0.5, 0.3, 0.8, 2.0, 0.6],
        [0.1, 0.2, 0.4, 0.6, 1.0],
    ])
    X = np.random.multivariate_normal(mean, cov, n_samples)

    X_centered = X - X.mean(axis=0)

    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    svd_variance = S ** 2 / (n_samples - 1)

    print(f"\n数据: {n_samples} 样本, {n_features} 特征")
    print(f"\n通过 covariance matrix 的 eigendecomposition 实现 PCA:")
    print(f"  Eigenvalues:  {np.round(eig_vals, 4)}")
    print(f"  PC1 方向: {np.round(eig_vecs[:, 0], 4)}")

    print(f"\n通过中心化数据的 SVD 实现 PCA:")
    print(f"  S^2/(n-1):    {np.round(svd_variance, 4)}")
    print(f"  V1 方向:  {np.round(Vt[0], 4)}")

    variance_match = np.allclose(eig_vals, svd_variance, atol=1e-8)
    direction_match = all(
        np.allclose(np.abs(eig_vecs[:, i]), np.abs(Vt[i]), atol=1e-8)
        for i in range(n_features)
    )
    print(f"\n  方差匹配: {variance_match}")
    print(f"  方向匹配 (符号除外): {direction_match}")

    explained = svd_variance / np.sum(svd_variance)
    cumulative = np.cumsum(explained)
    print(f"\n  解释方差比: {np.round(explained, 4)}")
    print(f"  累积:               {np.round(cumulative, 4)}")

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_features)
        pca.fit(X)
        print(f"\n  sklearn PCA 方差比: {np.round(pca.explained_variance_ratio_, 4)}")
        print(f"  与我们的 SVD 匹配: {np.allclose(explained, pca.explained_variance_ratio_, atol=1e-6)}")
    except ImportError:
        pass

    print()


def demo_matrix_properties():
    print("=" * 70)
    print("SVD 揭示的矩阵性质")
    print("=" * 70)

    np.random.seed(42)

    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=float)

    U, S, Vt = np.linalg.svd(A)

    print(f"\n矩阵 A:\n{A}")
    print(f"Singular values: {np.round(S, 6)}")

    print(f"\n秩 (非零 singular values): {np.sum(S > 1e-10)}")
    print(f"  (3x3 矩阵但秩只有 2: 行线性相关)")

    print(f"\nFrobenius 范数: {np.linalg.norm(A, 'fro'):.6f}")
    print(f"  sqrt(sum(sigma_i^2)): {np.sqrt(np.sum(S ** 2)):.6f}")

    print(f"\n谱范数 (最大 singular value): {S[0]:.6f}")
    print(f"  np.linalg.norm(A, 2): {np.linalg.norm(A, 2):.6f}")

    print(f"\n核范数 (singular values 之和): {np.sum(S):.6f}")

    B = np.array([[3, 1], [1, 3]], dtype=float)
    U_b, S_b, Vt_b = np.linalg.svd(B)
    print(f"\n方阵 B:\n{B}")
    print(f"Singular values: {S_b}")
    print(f"det(B) = {np.linalg.det(B):.4f}")
    print(f"Singular values 的乘积: {np.prod(S_b):.4f}")
    print(f"  (|det| = 方阵 singular values 的乘积)")

    print()


if __name__ == "__main__":
    demo_svd_basics()
    demo_geometry()
    demo_low_rank_approximation()
    demo_image_compression()
    demo_recommendation_system()
    demo_lsa()
    demo_noise_reduction()
    demo_pseudoinverse()
    demo_condition_number()
    demo_pca_is_svd()
    demo_matrix_properties()
