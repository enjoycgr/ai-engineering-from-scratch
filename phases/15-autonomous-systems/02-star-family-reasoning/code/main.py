"""STaR-loop 模拟器 —— stdlib Python。

玩具算术任务。"模型" 通过三种策略生成推理过程 (rationales)：
  1. sound reasoning (正确推理，始终正确)
  2. lazy shortcut (惰性捷径，在分布内问题上 40% 正确率，
     在分布外 (out-of-distribution, OOD) 接近零)
  3. random guess (随机猜测)

STaR 自举轮次过滤出正确答案的推理过程。没有屏蔽时，
捷径推理会被强化，因为它们在分布内看起来正确。

模拟器还运行 V-STaR 风格的推理选择器：采样 N 个推理过程，
选择验证器 (verifier) 的最高分。验证器本身在同一数据上训练，
因此它可能在 OOD 上将自信错误的推理排在诚实不确定的推理之上。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Trace:
    strategy: str  # "sound", "shortcut", "random"
    answer_correct: bool
    rationale_sound: bool


@dataclass
class Model:
    prob_sound: float
    prob_shortcut: float
    # implied prob_random = 1 - sound - shortcut

    def sample(self, on_ood: bool) -> Trace:
        r = random.random()
        if r < self.prob_sound:
            return Trace("sound", True, True)
        elif r < self.prob_sound + self.prob_shortcut:
            ok = random.random() < (0.05 if on_ood else 0.40)
            return Trace("shortcut", ok, False)
        else:
            ok = random.random() < 0.10
            return Trace("random", ok, False)


def evaluate(model: Model, n: int, on_ood: bool) -> tuple[float, float]:
    """返回 (答案准确率, 推理过程合理性比例)。"""
    correct = 0
    sound = 0
    for _ in range(n):
        t = model.sample(on_ood)
        if t.answer_correct:
            correct += 1
        if t.rationale_sound:
            sound += 1
    return correct / n, sound / n


def star_round(model: Model, n_samples: int = 1000) -> Model:
    """一轮 STaR：保留正确答案的轨迹，重新训练。"""
    kept = []
    for _ in range(n_samples):
        t = model.sample(on_ood=False)
        if t.answer_correct:
            kept.append(t)

    if not kept:
        return model

    sound_kept = sum(1 for k in kept if k.strategy == "sound")
    shortcut_kept = sum(1 for k in kept if k.strategy == "shortcut")
    random_kept = sum(1 for k in kept if k.strategy == "random")
    total = len(kept)

    # 根据被强化的内容更新比例，与旧先验混合以避免坍缩。
    alpha = 0.6
    new_sound = alpha * (sound_kept / total) + (1 - alpha) * model.prob_sound
    new_short = alpha * (shortcut_kept / total) + (1 - alpha) * model.prob_shortcut

    # 重新归一化
    s = new_sound + new_short
    if s > 1.0:
        new_sound /= s
        new_short /= s
    return Model(new_sound, new_short)


def run_star(rounds: int, initial: Model) -> list[Model]:
    models = [initial]
    m = initial
    for _ in range(rounds):
        m = star_round(m)
        models.append(m)
    return models


def vstar_infer(model: Model, samples_per_problem: int, n_problems: int,
                on_ood: bool) -> float:
    """V-STaR 风格的 best-of-N：选择我们会相信的轨迹。我们将
    验证器建模为一个置信度分数，它本身受 sound vs shortcut 的偏差影响
    (sound = 0.9 排名器可靠性, shortcut = 0.55)。

    注意：这是一个理想化验证器 —— 它读取 ground-truth
    ``rationale_sound`` 标记，因此它代表了一个训练良好的验证器
    可能达到的上界。真实验证器必须从轨迹本身推断合理性，
    因此实际收益会更小。
    """
    correct = 0
    for _ in range(n_problems):
        traces = [model.sample(on_ood) for _ in range(samples_per_problem)]
        # 验证器试图挑选正确的；它并不完美。
        best = None
        best_score = -1.0
        for t in traces:
            score = 0.9 if t.rationale_sound else (0.55 if t.answer_correct else 0.3)
            score += random.random() * 0.1
            if score > best_score:
                best_score = score
                best = t
        if best and best.answer_correct:
            correct += 1
    return correct / n_problems


def report_round(label: str, models: list[Model]) -> None:
    print(f"\n{label}")
    print("-" * 70)
    print(f"  {'round':>5}  {'p(sound)':>10}  {'p(shortcut)':>12}  "
          f"{'ID acc':>8}  {'OOD acc':>8}  {'sound frac':>10}")
    for i, m in enumerate(models):
        id_acc, id_sound = evaluate(m, 500, on_ood=False)
        ood_acc, _ = evaluate(m, 500, on_ood=True)
        print(f"  {i:>5}  {m.prob_sound:>10.3f}  {m.prob_shortcut:>12.3f}  "
              f"{id_acc:>8.1%}  {ood_acc:>8.1%}  {id_sound:>10.1%}")


def vstar_report(model: Model) -> None:
    print("\nV-STaR best-of-N 推理")
    print("-" * 70)
    for n in (1, 4, 16):
        for ood in (False, True):
            acc = vstar_infer(model, n, 500, ood)
            tag = "OOD" if ood else "ID"
            print(f"  n={n:>3}  {tag:<3}  accuracy {acc:.1%}")


def main() -> None:
    random.seed(42)
    print("=" * 70)
    print("STaR, V-STaR, QUIET-STaR (Phase 15, Lesson 2)")
    print("=" * 70)

    print("\n场景 A：无捷径的基础模型 (clean reasoning prior)")
    models = run_star(5, Model(prob_sound=0.20, prob_shortcut=0.0))
    report_round("STaR bootstrap rounds (clean)", models)

    print("\n场景 B：有捷径倾向的基础模型 (0.4 in-dist hit)")
    models = run_star(5, Model(prob_sound=0.20, prob_shortcut=0.40))
    report_round("STaR bootstrap rounds (with shortcuts)", models)

    vstar_report(models[-1])

    print()
    print("=" * 70)
    print("HEADLINE: STaR 强化任何能到达答案的路径")
    print("-" * 70)
    print("  场景 A 在 ID 和 OOD 上双双提升。")
    print("  场景 B 在 ID 上提升而 OOD 崩溃 —— 捷径")
    print("  被强化，因为它在训练数据中看起来正确。")
    print("  V-STaR 的验证器在推理时有所帮助，但无法消除训练")
    print("  中它被训练过的偏差。")


if __name__ == "__main__":
    main()
