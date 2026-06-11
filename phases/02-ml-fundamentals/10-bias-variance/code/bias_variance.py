import numpy as np
import warnings
warnings.filterwarnings("ignore")


def true_function(x):
    """真实函数：f(x) = sin(1.5x) + 0.5x"""
    return np.sin(1.5 * x) + 0.5 * x


def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    """
    从真实函数生成带高斯噪声的合成数据。

    参数：
        n_samples: 样本数量
        noise_std: 噪声标准差（控制不可约误差 irreducible error 的大小）
        x_range:   输入 x 的均匀采样范围
        seed:      随机种子，保证可复现
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y


def fit_polynomial(x_train, y_train, degree, lam=0.0):
    """
    用最小二乘法（可选 L2 / 岭回归 Ridge 正则化）拟合多项式。

    参数：
        x_train: 训练输入
        y_train: 训练输出
        degree:  多项式次数（控制模型复杂度 model complexity）
        lam:     L2 正则化强度（lambda）。lam > 0 时即为岭回归 (Ridge regression)
    """
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    if lam > 0:
        penalty = lam * np.eye(X.shape[1])
        penalty[0, 0] = 0          # 不惩罚偏置项
        w = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    else:
        w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w


def predict_polynomial(x, w):
    """用拟合得到的多项式权重 w 对新输入 x 做预测。"""
    degree = len(w) - 1
    X = np.column_stack([x ** d for d in range(degree + 1)])
    return X @ w


def bias_variance_decomposition(
    degrees,
    n_bootstrap=200,
    n_train=30,
    noise_std=0.5,
    n_test=100,
    lam=0.0,
):
    """
    通过自助采样 (bootstrap) 估计偏差 (bias)、方差 (variance) 和总误差。

    对每种多项式次数（模型复杂度）：
      1. 重复 n_bootstrap 次：抽取训练集 -> 拟合 -> 在固定测试网格上预测
      2. 计算平均预测、偏差²、方差、总误差

    返回字典：{degree: {"bias_sq": ..., "variance": ..., "total_error": ..., "noise": ...}}
    """
    rng = np.random.RandomState(42)
    x_test = np.linspace(-2.5, 2.5, n_test)
    y_true = true_function(x_test)

    results = {}

    for degree in degrees:
        predictions = np.zeros((n_bootstrap, n_test))

        for b in range(n_bootstrap):
            x_train, y_train = generate_data(
                n_samples=n_train, noise_std=noise_std, seed=rng.randint(0, 100000)
            )
            w = fit_polynomial(x_train, y_train, degree, lam=lam)
            predictions[b] = predict_polynomial(x_test, w)

        mean_pred = predictions.mean(axis=0)
        bias_sq = np.mean((mean_pred - y_true) ** 2)
        variance = np.mean(predictions.var(axis=0))
        total_error = np.mean(np.mean((predictions - y_true) ** 2, axis=1))

        results[degree] = {
            "bias_sq": bias_sq,
            "variance": variance,
            "total_error": total_error,
            "noise": noise_std ** 2,
        }

    return results


def print_decomposition(results):
    """打印偏差-方差分解表格。"""
    print(f"{'Degree':>6}  {'Bias^2':>10}  {'Variance':>10}  {'Noise':>10}  {'Total':>10}  {'B+V+N':>10}")
    print("-" * 70)
    for degree, r in sorted(results.items()):
        bvn = r["bias_sq"] + r["variance"] + r["noise"]
        print(
            f"{degree:>6d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['noise']:>10.4f}  {r['total_error']:>10.4f}  {bvn:>10.4f}"
        )


def find_optimal(results):
    """返回总误差最小的多项式次数（最优模型复杂度）。"""
    best_degree = min(results, key=lambda d: results[d]["total_error"])
    return best_degree


def demo_basic_decomposition():
    """演示 1：基础偏差-方差分解。"""
    print("=" * 70)
    print("偏差-方差分解 (BIAS-VARIANCE DECOMPOSITION)")
    print("真实函数: sin(1.5x) + 0.5x")
    print("噪声标准差: 0.5, 训练样本: 30, 自助轮数: 200")
    print("=" * 70)
    print()

    degrees = [1, 2, 3, 5, 7, 10, 15]
    results = bias_variance_decomposition(degrees)
    print_decomposition(results)

    best = find_optimal(results)
    print(f"\n最优次数 (Optimal degree): {best}")
    print(f"  偏差² (Bias^2):   {results[best]['bias_sq']:.4f}")
    print(f"  方差 (Variance): {results[best]['variance']:.4f}")
    print(f"  总误差 (Total):    {results[best]['total_error']:.4f}")


def demo_complexity_tradeoff():
    """演示 2：模型复杂度 (model complexity) 扫描，展示 U 型曲线。"""
    print()
    print("=" * 70)
    print("模型复杂度权衡 (MODEL COMPLEXITY TRADEOFF)")
    print("扫描多项式次数从 1 到 15")
    print("=" * 70)
    print()

    degrees = list(range(1, 16))
    results = bias_variance_decomposition(degrees)

    print(f"{'Degree':>6}  {'Bias^2':>10}  {'Variance':>10}  {'Total':>10}  {'Dominant':>12}")
    print("-" * 60)
    for degree in degrees:
        r = results[degree]
        dominant = "BIAS" if r["bias_sq"] > r["variance"] else "VARIANCE"
        print(
            f"{degree:>6d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['total_error']:>10.4f}  {dominant:>12}"
        )

    crossover = None
    for d in degrees[:-1]:
        if results[d]["bias_sq"] > results[d]["variance"]:
            if results[d + 1]["bias_sq"] <= results[d + 1]["variance"]:
                crossover = d + 1
                break

    if crossover:
        print(f"\n偏差-方差交叉点在次数 {crossover}")
        print("低于此：偏差占主导（欠拟合 underfitting）")
        print("高于此：方差占主导（过拟合 overfitting）")


def demo_regularization_effect():
    """演示 3：L2 / 岭回归 (Ridge) 正则化对偏差和方差的影响。"""
    print()
    print("=" * 70)
    print("正则化效应 (REGULARIZATION EFFECT) — L2 / 岭回归 (Ridge)")
    print("固定次数=10，扫描 lambda")
    print("=" * 70)
    print()

    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"{'Lambda':>10}  {'Bias^2':>10}  {'Variance':>10}  {'Total':>10}")
    print("-" * 50)

    for lam in lambdas:
        results = bias_variance_decomposition([10], lam=lam)
        r = results[10]
        print(f"{lam:>10.3f}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  {r['total_error']:>10.4f}")

    print()
    print("随着 lambda 增大：")
    print("  - 方差减小（模型更受约束）")
    print("  - 偏差增大（模型被迫更简单）")
    print("  - 最优 lambda 平衡这两种效应")


def demo_data_size_effect():
    """演示 4：训练集大小对偏差和方差的影响。"""
    print()
    print("=" * 70)
    print("训练集大小效应 (TRAINING SET SIZE EFFECT)")
    print("固定次数=5，改变 n_train")
    print("=" * 70)
    print()

    sizes = [10, 20, 50, 100, 200, 500]

    print(f"{'N_train':>8}  {'Bias^2':>10}  {'Variance':>10}  {'Total':>10}")
    print("-" * 50)

    for n in sizes:
        results = bias_variance_decomposition([5], n_train=n)
        r = results[5]
        print(f"{n:>8d}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  {r['total_error']:>10.4f}")

    print()
    print("更多数据降低方差，但不影响偏差。")
    print("如果你的问题是高偏差 (high bias)，更多数据无济于事。")


def demo_diagnosis():
    """演示 5：欠拟合 (underfitting) vs 过拟合 (overfitting) 诊断。"""
    print()
    print("=" * 70)
    print("欠拟合 vs 过拟合诊断 (UNDERFITTING vs OVERFITTING DIAGNOSIS)")
    print("=" * 70)
    print()

    rng = np.random.RandomState(42)
    x_train, y_train = generate_data(n_samples=30, seed=42)
    x_test, y_test = generate_data(n_samples=100, seed=99)

    cases = [
        (1, "线性 (次数 1)"),
        (4, "多项式 (次数 4)"),
        (15, "多项式 (次数 15)"),
    ]

    for degree, name in cases:
        w = fit_polynomial(x_train, y_train, degree)
        train_pred = predict_polynomial(x_train, w)
        test_pred = predict_polynomial(x_test, w)

        train_mse = np.mean((train_pred - y_train) ** 2)
        test_mse = np.mean((test_pred - y_test) ** 2)
        gap = test_mse - train_mse

        if train_mse > 0.5 and test_mse > 0.5 and gap < train_mse * 0.5:
            diagnosis = "高偏差 (HIGH BIAS) — 欠拟合 (underfitting)"
        elif gap > train_mse * 2:
            diagnosis = "高方差 (HIGH VARIANCE) — 过拟合 (overfitting)"
        else:
            diagnosis = "拟合合理 (REASONABLE FIT)"

        print(f"{name}:")
        print(f"  训练 MSE: {train_mse:.4f}")
        print(f"  测试 MSE:  {test_mse:.4f}")
        print(f"  差距:       {gap:.4f}")
        print(f"  诊断: {diagnosis}")
        print()


def demo_learning_curves():
    """演示 6：学习曲线 (learning curves) — 训练/测试误差随训练集大小变化。"""
    print()
    print("=" * 70)
    print("学习曲线 (LEARNING CURVES)")
    print("训练误差 vs 测试误差，随训练集大小增长")
    print("=" * 70)
    print()

    rng = np.random.RandomState(42)
    x_test = np.linspace(-2.5, 2.5, 200)
    y_test = true_function(x_test)

    sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]

    for degree, label in [(1, "次数 1 (高偏差 high bias)"), (5, "次数 5 (平衡)"), (12, "次数 12 (高方差 high variance)")]:
        print(f"  {label}:")
        print(f"  {'N_train':>8}  {'Train MSE':>10}  {'Test MSE':>10}  {'Gap':>10}")
        print(f"  {'-' * 48}")

        for n in sizes:
            train_errors = []
            test_errors = []
            for seed in range(50):
                x_train, y_train = generate_data(n_samples=n, seed=rng.randint(0, 100000))
                try:
                    w = fit_polynomial(x_train, y_train, degree)
                    train_pred = predict_polynomial(x_train, w)
                    test_pred = predict_polynomial(x_test, w)
                    train_mse = np.mean((train_pred - y_train) ** 2)
                    test_mse = np.mean((test_pred - y_test) ** 2)
                    train_errors.append(train_mse)
                    test_errors.append(test_mse)
                except (np.linalg.LinAlgError, ValueError):
                    continue

            if train_errors:
                avg_train = np.mean(train_errors)
                avg_test = np.mean(test_errors)
                gap = avg_test - avg_train
                print(f"  {n:>8d}  {avg_train:>10.4f}  {avg_test:>10.4f}  {gap:>10.4f}")

        print()

    print("高偏差（次数 1）：两条曲线收敛到高误差。差距保持小。")
    print("高方差（次数 12）：训练误差保持低，测试误差保持高。")
    print("更多数据降低方差，但无法修复偏差。")


def demo_regularization_sweep():
    """演示 7：正则化扫描 — 岭回归 (Ridge) alpha 对偏差/方差的影响。"""
    print()
    print("=" * 70)
    print("正则化扫描 (REGULARIZATION SWEEP) — 岭回归 (Ridge) alpha vs 偏差/方差")
    print("固定次数=15，扫描 alpha 从 0.001 到 100")
    print("=" * 70)
    print()

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    print(f"  {'Alpha':>10}  {'Bias^2':>10}  {'Variance':>10}  {'Total':>10}  {'Dominant':>12}")
    print(f"  {'-' * 60}")

    best_alpha = None
    best_total = float("inf")

    for alpha in alphas:
        results = bias_variance_decomposition([15], lam=alpha, n_bootstrap=200)
        r = results[15]
        dominant = "BIAS" if r["bias_sq"] > r["variance"] else "VARIANCE"
        print(
            f"  {alpha:>10.3f}  {r['bias_sq']:>10.4f}  {r['variance']:>10.4f}  "
            f"{r['total_error']:>10.4f}  {dominant:>12}"
        )
        if r["total_error"] < best_total:
            best_total = r["total_error"]
            best_alpha = alpha

    print()
    print(f"最优 alpha: {best_alpha}")
    print(f"  最优处总误差: {best_total:.4f}")
    print()
    print("小 alpha：方差占主导（模型无约束，拟合噪声）")
    print("大 alpha：偏差占主导（模型过度约束，错过信号）")
    print("最优 alpha 平衡两者，位于 U 型曲线底部。")


if __name__ == "__main__":
    demo_basic_decomposition()
    demo_complexity_tradeoff()
    demo_regularization_effect()
    demo_data_size_effect()
    demo_diagnosis()
    demo_learning_curves()
    demo_regularization_sweep()
    print("所有偏差-方差演示完成。")
