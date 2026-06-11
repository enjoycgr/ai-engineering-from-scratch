"""能力 vs 对齐竞赛模拟器 —— stdlib Python。

每个 RSI 循环有两个复合过程。能力增长率 r_c，对齐增长率 r_a，
各自有可配置噪声。模拟器追踪差距 M(t) = C(t) - A(t) 以及
差距会跨越安全阈值的循环。
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass


DEFAULT_SEED = 11


@dataclass
class Config:
    r_c: float
    r_a: float
    noise_c: float
    noise_a: float
    threshold: float = 1.5


def run(cycles: int, cfg: Config) -> list[tuple[int, float, float, float]]:
    c = 1.0
    a = 1.0
    out = [(0, c, a, c - a)]
    for cyc in range(1, cycles + 1):
        nc = cfg.r_c + random.gauss(0, cfg.noise_c)
        na = cfg.r_a + random.gauss(0, cfg.noise_a)
        c *= max(0.9, nc)
        a *= max(0.9, na)
        out.append((cyc, c, a, c - a))
    return out


def crossing_cycle(trajectory, threshold: float) -> int:
    for cyc, _c, _a, gap in trajectory:
        if gap >= threshold:
            return cyc
    return -1


def print_trajectory(label: str, cfg: Config, cycles: int = 40) -> None:
    traj = run(cycles, cfg)
    print(f"\n{label}")
    print(f"  r_c={cfg.r_c:.2f} r_a={cfg.r_a:.2f} "
          f"noise_c={cfg.noise_c:.3f} noise_a={cfg.noise_a:.3f}")
    print(f"  threshold (C - A): {cfg.threshold:.2f}")
    print(f"  {'cycle':>6}  {'C(t)':>8}  {'A(t)':>8}  {'C-A':>8}  flag")
    # 打印大约九个始终包含循环 0 和总循环数的快照，
    # so changing `cycles` (e.g. for an exercise) doesn't silently drop rows.
    step = max(1, cycles // 8)
    for cyc, c, a, gap in traj:
        if cyc == 0 or cyc == cycles or cyc % step == 0:
            flag = "PAUSE" if gap >= cfg.threshold else "ok"
            print(f"  {cyc:>6}  {c:>8.2f}  {a:>8.2f}  {gap:>+8.2f}  {flag}")
    cross = crossing_cycle(traj, cfg.threshold)
    if cross >= 0:
        print(f"  -> threshold crossed at cycle {cross}")
    else:
        print("  -> threshold not crossed in simulated window")


def monte_carlo(cfg: Config, cycles: int, trials: int) -> None:
    crossings = []
    for _ in range(trials):
        traj = run(cycles, cfg)
        cross = crossing_cycle(traj, cfg.threshold)
        if cross >= 0:
            crossings.append(cross)
    print(f"\n  monte-carlo over {trials} trials, {cycles} cycles each")
    print(f"  crossed: {len(crossings)} ({len(crossings)/trials:.0%})")
    if crossings:
        avg = sum(crossings) / len(crossings)
        p50 = statistics.median(crossings)
        print(f"  mean crossing cycle: {avg:.1f}")
        print(f"  median crossing cycle: {p50}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="pause-gap threshold C - A (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed (default: %(default)s)")
    args = parser.parse_args()

    random.seed(args.seed)
    th = args.threshold
    print("=" * 70)
    print("CAPABILITY vs ALIGNMENT RACE (Phase 15, Lesson 7)")
    print("=" * 70)

    # 场景 A：能力适度超越对齐
    print_trajectory(
        "场景 A —— 能力超越对齐",
        Config(r_c=1.15, r_a=1.08, noise_c=0.02, noise_a=0.03, threshold=th),
    )

    # 场景 B：对齐保持同步
    print_trajectory(
        "场景 B —— 匹配速率（仅噪声漂移）",
        Config(r_c=1.10, r_a=1.10, noise_c=0.02, noise_a=0.03, threshold=th),
    )

    # 场景 C：对齐率更高，但能力有激增
    print_trajectory(
        "场景 C —— 对齐平均率更高但能力有激增",
        Config(r_c=1.10, r_a=1.13, noise_c=0.06, noise_a=0.01, threshold=th),
    )

    print("\nMonte-Carlo on Scenario A")
    monte_carlo(
        Config(r_c=1.15, r_a=1.08, noise_c=0.02, noise_a=0.03, threshold=th),
        cycles=30, trials=500,
    )
    print("\nMonte-Carlo on Scenario C")
    monte_carlo(
        Config(r_c=1.10, r_a=1.13, noise_c=0.06, noise_a=0.01, threshold=th),
        cycles=30, trials=500,
    )

    print()
    print("=" * 70)
    print("HEADLINE: 微小速率差异复合成安全阈值跨越")
    print("-" * 70)
    print("  场景 A 在不到 10 个循环内跨越绝对 1.5 差距 (C - A)。")
    print("  场景 B 保持有界 —— 相同平均速率，仅噪声漂移。")
    print("  场景 C：如果能力有大激增，更高的对齐平均值")
    print("  并不能救你。噪声和漂移一样重要。")
    print("  RSI 风格管道需要内置差距暂停阈值。")


if __name__ == "__main__":
    main()
