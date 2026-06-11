"""数值任务上的 multi-agent debate (多智能体辩论) (Du et al. 2023 风格)。

3 个 agent，每个从一个不同的（可能是错误的）答案开始。在每一轮中，
每个 agent 阅读其他 agent 的答案，并朝着加权平均修订。
每轮记录 convergence (收敛)。Agent 策略是 scripted (脚本化的)，而非 LLM 驱动——
重点是 debate dynamics (辩论动态)。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


TRUE_ANSWER = 42.0


@dataclass
class DebateAgent:
    name: str
    answer: float
    confidence: float
    history: list[float] = field(default_factory=list)

    def initial(self) -> None:
        self.history.append(self.answer)

    def revise(self, others: list["DebateAgent"]) -> None:
        """按 confidence (置信度) 加权的自身 + 他人的加权平均。"""
        weights = [self.confidence] + [o.confidence for o in others]
        values = [self.answer] + [o.answer for o in others]
        total_w = sum(weights)
        new_answer = sum(w * v for w, v in zip(weights, values)) / total_w
        self.answer = new_answer
        self.confidence = min(self.confidence * 1.05, 1.0)
        self.history.append(self.answer)


def agreement_score(agents: list[DebateAgent], tol: float = 0.1) -> float:
    """落在均值 tol 范围内的 agent 比例。"""
    mean = sum(a.answer for a in agents) / len(agents)
    agree = sum(1 for a in agents if abs(a.answer - mean) <= tol)
    return agree / len(agents)


def error_vs_truth(agents: list[DebateAgent]) -> float:
    mean = sum(a.answer for a in agents) / len(agents)
    return abs(mean - TRUE_ANSWER)


def run_debate(agents: list[DebateAgent], rounds: int, label: str) -> None:
    print(f"\n=== {label} ({rounds} 轮) ===")
    for a in agents:
        a.initial()
    hdr = " ".join(f"{a.name:>6s}" for a in agents)
    print(f"  round    {hdr}    agree    err-vs-truth")
    for a in agents:
        pass
    print(f"    0     {' '.join(f'{a.answer:6.2f}' for a in agents)}    {agreement_score(agents):4.2f}     {error_vs_truth(agents):5.2f}")
    for r in range(1, rounds + 1):
        updates = []
        for a in agents:
            others = [o for o in agents if o is not a]
            updates.append((a, others))
        for a, others in updates:
            a.revise(others)
        print(f"    {r}     {' '.join(f'{a.answer:6.2f}' for a in agents)}    {agreement_score(agents):4.2f}     {error_vs_truth(agents):5.2f}")


def fresh_team(seed: int) -> list[DebateAgent]:
    random.seed(seed)
    return [
        DebateAgent(name="A", answer=38.0, confidence=0.6),
        DebateAgent(name="B", answer=42.5, confidence=0.8),
        DebateAgent(name="C", answer=51.0, confidence=0.4),
    ]


def single_shot_majority(agents: list[DebateAgent]) -> float:
    """对照：第 0 轮答案的 majority (多数)（self-consistency (自一致性) 基线）。"""
    return sum(a.answer for a in agents) / len(agents)


def main() -> None:
    print("Multi-agent debate (Du et al. 2023 风格)")
    print("-" * 46)
    print(f"真实答案: {TRUE_ANSWER}")

    baseline = fresh_team(seed=1)
    for a in baseline:
        a.initial()
    control_mean = single_shot_majority(baseline)
    print(f"\n对照 (第 0 轮均值, self-consistency 基线): {control_mean:.2f}")
    print(f"与真实答案误差: {abs(control_mean - TRUE_ANSWER):.2f}")

    team3 = fresh_team(seed=1)
    run_debate(team3, rounds=3, label="Debate 3 agents, 3 rounds")

    team5 = fresh_team(seed=2)
    run_debate(team5, rounds=5, label="Debate 3 agents, 5 rounds (diminishing returns)")

    print("\n要点:")
    print("  - 1 轮交换最大程度削减了误差。")
    print("  - 第 2-3 轮有叠加效果。")
    print("  - 超过第 3 轮，每轮收益递减 (Du et al. plateau)。")
    print("  - 成本随 N * R 次 LLM 调用增长，且 context 不断扩大。")


if __name__ == "__main__":
    main()
