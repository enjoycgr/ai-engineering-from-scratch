"""Logistic 拟合时间跨度估计器 —— stdlib Python。

给定合成任务结果 (expert_time_hours, success)，将 logistic
曲线拟合到 P(success) vs log(expert_time) 并报告 50/10/90% 时间跨度。
然后展示 eval-context gaming 对观察数字的影响。

仅使用 stdlib；logistic 拟合是最小梯度下降
实现，为教学而非生产设计。
"""

from __future__ import annotations

import math
import random


# ---------- 合成数据生成器 ----------

def synth_tasks(true_horizon_hours: float, slope: float = 1.2,
                n: int = 120) -> list[tuple[float, bool]]:
    """生成合成 (expert_time_hours, success) 对。

    P(success) = sigmoid(slope * (log(true_horizon) - log(expert_time))).
    """
    log_h = math.log(true_horizon_hours)
    # 专家时间跨度 0.05 小时到 ~48 小时
    out = []
    for _ in range(n):
        t = math.exp(random.uniform(math.log(0.05), math.log(48)))
        logit = slope * (log_h - math.log(t))
        p = 1.0 / (1.0 + math.exp(-logit))
        success = random.random() < p
        out.append((t, success))
    return out


# ---------- Logistic 拟合（微型 GD） ----------

def sigmoid(x: float) -> float:
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def fit(tasks: list[tuple[float, bool]], iters: int = 4000,
        lr: float = 0.05) -> tuple[float, float]:
    """拟合 P(success) = sigmoid(w * log(t) + b)。返回 (w, b)。"""
    w = 0.0
    b = 0.0
    for _ in range(iters):
        dw = 0.0
        db = 0.0
        n = len(tasks)
        for t, s in tasks:
            y = 1.0 if s else 0.0
            p = sigmoid(w * math.log(t) + b)
            err = p - y
            dw += err * math.log(t)
            db += err
        w -= lr * dw / n
        b -= lr * db / n
    return w, b


def horizon_at(w: float, b: float, p: float) -> float:
    """P(success) = p 的专家时间。sigmoid(w*log(t)+b) = p ->
    log(t) = (logit(p) - b) / w。"""
    logit = math.log(p / (1 - p))
    # 零（或接近零）斜率意味着成功概率不
    # 依赖任务长度，因此时间跨度未定义。抛出异常而非
    # 静默返回 inf/nan，以便调用者看到明显的失败。
    eps = 1e-12
    if abs(w) < eps:
        raise ValueError(
            f"horizon undefined: slope w={w} is ~0 "
            f"(b={b}, p={p}, logit={logit})"
        )
    return math.exp((logit - b) / w)


# ---------- 评估环境博弈模拟器 ----------

def inject_gaming(tasks: list[tuple[float, bool]],
                  gaming_rate: float) -> list[tuple[float, bool]]:
    """将 `gaming_rate` 比例的失败翻转为成功（模型在
    评估上下文中表现更好）。返回新列表。"""
    gamed = []
    for t, s in tasks:
        if not s and random.random() < gaming_rate:
            gamed.append((t, True))
        else:
            gamed.append((t, s))
    return gamed


# ---------- Driver ----------

def report(label: str, w: float, b: float) -> None:
    h50 = horizon_at(w, b, 0.50)
    h10 = horizon_at(w, b, 0.10)
    h90 = horizon_at(w, b, 0.90)
    print(f"  {label:<40}  50%={h50:>6.2f} hr  "
          f"10%={h10:>6.2f} hr  90%={h90:>6.2f} hr")


def main() -> None:
    random.seed(3)
    print("=" * 80)
    print("METR-STYLE HORIZON ESTIMATOR (Phase 15, Lesson 21)")
    print("=" * 80)

    true_h = 14.0
    print(f"\n合成 ground truth：50% 时间跨度 = {true_h:.1f} 小时")
    print("-" * 80)

    tasks = synth_tasks(true_horizon_hours=true_h, n=160)
    w, b = fit(tasks)
    clean_h50 = horizon_at(w, b, 0.50)
    report("clean evaluation (no gaming)", w, b)

    gamed_h50: dict[float, float] = {}
    for rate in (0.1, 0.2, 0.4):
        gamed = inject_gaming(tasks, gaming_rate=rate)
        w_g, b_g = fit(gamed)
        gamed_h50[rate] = horizon_at(w_g, b_g, 0.50)
        report(f"with eval-context gaming rate {rate:.0%}", w_g, b_g)

    print()
    print("=" * 80)
    print("HEADLINE: 时间跨度拟合于观察到的成功率；博弈使其偏移")
    print("-" * 80)
    print(f"  使用 seed=3 / n=160 / iters=4000 / true_h={true_h:.1f} 小时：")
    print(f"    clean fit          50% 时间跨度 ≈ {clean_h50:>6.2f} 小时 "
          f"(ground truth {true_h:.1f})")
    for rate, h in gamed_h50.items():
        delta = h - true_h
        print(f"    gaming rate {rate:>4.0%}   50% 时间跨度 ≈ {h:>6.2f} 小时 "
              f"({delta:+.2f} 小时 vs ground truth)")
    print("  趋势：gaming 将观察到的 50% 时间跨度推离")
    print("  合成 ground truth，且随速率攀升。精确增量取决于")
    print("  seed、n、iters 和选择的 true_h。没有时间跨度")
    print("  gaming 审计的能力上限是部署上下文可能无法达到的。")


if __name__ == "__main__":
    main()
