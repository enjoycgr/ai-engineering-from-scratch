"""极简 AlphaEvolve 风格进化循环 —— stdlib Python。

玩具符号回归 (symbolic regression)。"LLM" 对候选表达式提出小型变异
（改变常数、改变运算符、添加项）。"评估器" (evaluator) 在训练集和
留出测试集 (held-out test) 上为表达式打分。

MAP-elites 网格保持多样化候选：按（表达式深度、常数量级桶）作为单元格键。
没有留出集 (held-out split) 时循环会严重过拟合；有了留出集后最佳候选
才能泛化。
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


DEFAULT_SEED = 1


# 循环试图重新发现的目标函数。
def target(x: float) -> float:
    return 2.0 * x * x + 3.0 * x - 1.0


Expr = tuple  # recursive: ("num", v) | ("x",) | ("add", a, b) | ("mul", a, b)


def evaluate_expr(e: Expr, x: float) -> float:
    tag = e[0]
    if tag == "num":
        return float(e[1])
    if tag == "x":
        return x
    if tag == "add":
        return evaluate_expr(e[1], x) + evaluate_expr(e[2], x)
    if tag == "mul":
        return evaluate_expr(e[1], x) * evaluate_expr(e[2], x)
    raise ValueError(tag)


def depth(e: Expr) -> int:
    tag = e[0]
    if tag in ("num", "x"):
        return 1
    return 1 + max(depth(e[1]), depth(e[2]))


def max_const(e: Expr) -> float:
    tag = e[0]
    if tag == "num":
        return abs(e[1])
    if tag == "x":
        return 0.0
    return max(max_const(e[1]), max_const(e[2]))


def mutate(e: Expr) -> Expr:
    """LLM 针对性编辑的替身。"""
    choice = random.random()
    if choice < 0.25:
        return random_leaf()
    if choice < 0.5:
        return ("add", e, random_leaf())
    if choice < 0.75:
        return ("mul", e, random_leaf())
    # perturb a constant somewhere
    return perturb(e)


def perturb(e: Expr) -> Expr:
    tag = e[0]
    if tag == "num":
        return ("num", e[1] + random.choice([-1.0, -0.5, 0.5, 1.0]))
    if tag == "x":
        return e
    return (tag, perturb(e[1]), e[2]) if random.random() < 0.5 else (tag, e[1], perturb(e[2]))


def random_leaf() -> Expr:
    if random.random() < 0.5:
        return ("x",)
    return ("num", float(random.choice([-2, -1, 0, 1, 2, 3])))


def render(e: Expr) -> str:
    tag = e[0]
    if tag == "num":
        return f"{e[1]:g}"
    if tag == "x":
        return "x"
    op = "+" if tag == "add" else "*"
    return f"({render(e[1])} {op} {render(e[2])})"


def mse(e: Expr, xs: list[float]) -> float:
    total = 0.0
    for x in xs:
        try:
            y = evaluate_expr(e, x)
        except (OverflowError, ValueError):
            return float("inf")
        total += (y - target(x)) ** 2
    return total / max(1, len(xs))


@dataclass
class Candidate:
    expr: Expr
    train_score: float
    test_score: float
    generation: int


def cell_key(e: Expr) -> tuple[int, int]:
    d = min(depth(e), 6)
    c = min(int(max_const(e) / 2), 4)
    return (d, c)


def seed_candidate(test_xs: list[float], train_xs: list[float], gen: int) -> Candidate:
    e = random_leaf()
    return Candidate(e, mse(e, train_xs), mse(e, test_xs), gen)


def run_loop(
    generations: int,
    pop: int,
    use_holdout: bool,
    seed: int | None = None,
) -> tuple[Candidate, list[float], list[float]]:
    if seed is not None:
        random.seed(seed)
    train_xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    test_xs = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]

    def signal_of(c: Candidate) -> float:
        return 0.5 * (c.train_score + c.test_score) if use_holdout else c.train_score

    archive: dict[tuple[int, int], Candidate] = {}
    for _ in range(pop):
        c = seed_candidate(test_xs, train_xs, 0)
        key = cell_key(c.expr)
        incumbent = archive.get(key)
        if incumbent is None or signal_of(c) < signal_of(incumbent):
            archive[key] = c

    best_trace: list[float] = []
    test_trace: list[float] = []
    for g in range(1, generations + 1):
        parent = random.choice(list(archive.values()))
        child_expr = mutate(parent.expr)
        tr = mse(child_expr, train_xs)
        te = mse(child_expr, test_xs)
        child = Candidate(child_expr, tr, te, g)
        key = cell_key(child_expr)
        incumbent = archive.get(key)
        if incumbent is None or signal_of(child) < signal_of(incumbent):
            archive[key] = child

        best = min(archive.values(), key=lambda c: c.train_score)
        best_trace.append(best.train_score)
        test_trace.append(best.test_score)

    # 最终选择必须使用与搜索相同的信号：当 use_holdout=False 时
    # 在这里使用留出测试会悄无声息地将留出集泄露回 Run B，
    # 并掩盖本课展示的过拟合现象。
    best = min(archive.values(), key=signal_of)
    return best, best_trace, test_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-holdout",
        action="store_true",
        help="跳过留出测试评估器 (仅 Run B；强制展示 reward-hacking)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ALPHAEVOLVE-STYLE LOOP (Phase 15, Lesson 3)")
    print("=" * 70)
    print("target: 2x^2 + 3x - 1")

    if not args.no_holdout:
        print("\nRun A: 评估信号中包含留出测试")
        best, train_trace, _ = run_loop(
            generations=1500, pop=20, use_holdout=True, seed=DEFAULT_SEED
        )
        print(f"  best expr : {render(best.expr)}")
        print(f"  train MSE : {best.train_score:.4f}")
        print(f"  test  MSE : {best.test_score:.4f}")
        print(f"  generation: {best.generation}")
        print("  progress  : gen 100 train={:.3f} gen 500 train={:.3f} gen 1500 train={:.3f}".format(
            train_trace[99], train_trace[499], train_trace[-1]))

    print("\nRun B: no held-out test (train-only evaluator -> reward hacking risk)")
    best, _train_trace, _test_trace = run_loop(
        generations=1500, pop=20, use_holdout=False, seed=DEFAULT_SEED
    )
    print(f"  best expr : {render(best.expr)}")
    print(f"  train MSE : {best.train_score:.4f}")
    print(f"  test  MSE : {best.test_score:.4f}")
    print(f"  generation: {best.generation}")
    gap = best.test_score - best.train_score
    print(f"  train-to-test gap: {gap:+.4f}  (large gap = overfit/reward hacking proxy)")

    print()
    print("=" * 70)
    print("HEADLINE: 评估器就是架构本身")
    print("-" * 70)
    print("  Run A 收敛到低训练 MSE 且低测试 MSE。")
    print("  Run B 收敛到低训练 MSE；测试 MSE 保持松散或更差。")
    print("  留出评估器 (held-out evaluator) 是发现 (discovery) 与")
    print("  reward hacking 之间的区别。AlphaEvolve 的胜利在于")
    print("  存在此类评估器的领域。挑选这些领域才是难点。")


if __name__ == "__main__":
    main()
