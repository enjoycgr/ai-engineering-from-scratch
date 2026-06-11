"""Scaling laws（扩展定律）— Chinchilla loss 方程、计算最优 (N, D)、过度训练成本。

纯标准库。通过网格搜索数值验证 D/N ≈ 20 规则。
"""

import math


A = 406.4
B_CONST = 410.7
ALPHA = 0.34
BETA = 0.28
E_CONST = 1.69


def chinchilla_loss(N, D, A=A, B=B_CONST, alpha=ALPHA, beta=BETA, E=E_CONST):
    return A / N ** alpha + B / D ** beta + E


def compute_optimal(C_flops, n_grid=200):
    """通过在 log N 上网格搜索，找到在约束 6ND = C 下最小化 loss 的 (N, D)。"""
    # 6ND = C => D = C / (6N)
    log_N_min = math.log10(1e5)
    log_N_max = math.log10(1e13)
    best = (None, None, float("inf"))
    for i in range(n_grid):
        log_N = log_N_min + (log_N_max - log_N_min) * i / (n_grid - 1)
        N = 10 ** log_N
        D = C_flops / (6 * N)
        if D < 1e6:
            continue
        loss = chinchilla_loss(N, D)
        if loss < best[2]:
            best = (N, D, loss)
    return best


def pretty(n):
    """人类可读格式。"""
    for unit, k in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if n >= k:
            return f"{n / k:.1f}{unit}"
    return f"{n:.0f}"


def main():
    print("=== 计算最优 (N, D) 跨计算预算 ===")
    print(f"{'compute':>12}  {'N*':>10}  {'D*':>10}  {'D/N':>7}  {'loss':>7}")
    for C in [1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24, 1e25]:
        N, D, L = compute_optimal(C)
        print(f"  {C:>10.0e}   {pretty(N):>9}   {pretty(D):>9}   {D / N:>6.1f}   {L:>6.3f}")
    print()
    print("Hoffmann 2022 发表 D/N ≈ 20 作为头条结果。使用上述拟合常数")
    print("(alpha=0.34, beta=0.28) 时，最优 D/N 随 C 增长。")
    print("真实扩展定律拟合将最优值放在 Chinchilla 研究的计算范围内")
    print("(~1e22 到 1e23 FLOPs) 约 20 附近；外推会漂移。")
    print()

    print("=== 过度训练成本 (Llama 风格) ===")
    # 取一个计算预算，使用最优 N 的 1/10 和最优 D 的 10 倍。
    C = 1e24
    N_opt, D_opt, L_opt = compute_optimal(C)
    N_under = N_opt / 10
    D_over = D_opt * 10
    L_over = chinchilla_loss(N_under, D_over)
    print(f"计算预算:                 {C:.0e} FLOPs")
    print(f"Chinchilla 最优:             N={pretty(N_opt)}  D={pretty(D_opt)}  loss={L_opt:.3f}")
    print(f"过度训练 (N/10, D×10):     N={pretty(N_under)}  D={pretty(D_over)}  loss={L_over:.3f}")
    print(f"loss 惩罚 (过度训练):      {L_over - L_opt:+.3f}")
    print(f"推理 FLOP 节省 (~N):    {N_opt / N_under:.0f}x 推理更便宜")
    print()

    print("=== 真实模型 vs 预测 loss ===")
    models = [
        ("GPT-3 175B",          175e9,  300e9),
        ("Chinchilla 70B",       70e9, 1400e9),
        ("Llama 2 70B",          70e9, 2000e9),
        ("Llama 3 8B",            8e9, 15_000e9),
        ("Llama 3 70B",          70e9, 15_000e9),
        ("DeepSeek-V3 (active)", 37e9, 14_800e9),
        ("Qwen 2.5 72B",         72e9,  18_000e9),
    ]
    print(f"{'model':<24}  {'N':>8}  {'D':>8}  {'D/N':>7}  {'loss':>7}")
    for name, N, D in models:
        L = chinchilla_loss(N, D)
        print(f"  {name:<22}  {pretty(N):>7}  {pretty(D):>7}  {D / N:>6.1f}  {L:>6.3f}")
    print()
    print("许多 2026 年的模型远超 Chinchilla (D/N ≈ 20)。")
    print("原因：推理成本随 N 缩放；过度训练节省推理")
    print("代价是额外的预训练 FLOPs。")


if __name__ == "__main__":
    main()
