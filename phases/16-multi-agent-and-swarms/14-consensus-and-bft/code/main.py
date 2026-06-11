"""Consensus and BFT for LLM agents, stdlib only.

实现三种聚合器（plurality (相对多数)、CP-WBFT、DecentLLMs）和三种
攻击模式（byzantine (拜占庭)、sycophancy (谄媚)、monoculture (单一文化)）。
打印 (attack, aggregator) -> final answer 的表格，高亮正确决策。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Callable


@dataclass
class Vote:
    agent: str
    answer: str
    confidence: float

    def canonical(self) -> str:
        """Rough semantic clustering (粗略语义聚类): lowercase + strip whitespace/punct."""
        return "".join(c for c in self.answer.lower().strip() if c.isalnum() or c == "." or c == "%")


def plurality(votes: list[Vote]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    rep: dict[str, str] = {}
    for v in votes:
        key = v.canonical()
        counts[key] = counts.get(key, 0) + 1
        rep.setdefault(key, v.answer)
    winner_key = max(counts, key=counts.get)
    return rep[winner_key], counts


def cp_wbft(votes: list[Vote], threshold: float = 0.5) -> tuple[str | None, dict[str, float]]:
    weights: dict[str, float] = {}
    rep: dict[str, str] = {}
    for v in votes:
        key = v.canonical()
        weights[key] = weights.get(key, 0.0) + v.confidence
        rep.setdefault(key, v.answer)
    total = sum(weights.values()) or 1.0
    winner_key = max(weights, key=weights.get)
    if weights[winner_key] / total < threshold:
        return None, weights
    return rep[winner_key], weights


def decentllms(votes: list[Vote]) -> tuple[str | None, dict[str, float]]:
    """通过 evaluator agent 为 proposal 评分 0-1，选择 geometric-median (几何中位数) 簇。

    简化版：evaluator 就是聚合器本身，scoring = confidence。
    'geometric median' 选择其成员在置信度空间中到 median 的 pairwise distance (成对距离)
    之和最小的簇；按大小打破平局。
    """
    clusters: dict[str, list[Vote]] = {}
    for v in votes:
        clusters.setdefault(v.canonical(), []).append(v)

    scores: dict[str, float] = {}
    for key, cluster in clusters.items():
        med = median([v.confidence for v in cluster])
        dist = sum(abs(v.confidence - med) for v in cluster)
        scores[key] = len(cluster) * max(0.0, 1.0 - dist)

    winner_key = max(scores, key=scores.get)
    rep = clusters[winner_key][0].answer
    return rep, scores


def scenario(name: str, correct: str, votes: list[Vote]) -> None:
    print("\n" + "=" * 72)
    print(f"SCENARIO: {name}")
    print(f"  correct answer: {correct!r}")
    print("=" * 72)
    for v in votes:
        print(f"  {v.agent:12s} -> {v.answer!r:20s}  conf={v.confidence:.2f}")

    plural, counts = plurality(votes)
    cp, weights = cp_wbft(votes)
    dec, scores = decentllms(votes)

    def mark(a: str | None) -> str:
        if a is None:
            return "[rejected below threshold]"
        return "[CORRECT]" if a == correct else "[WRONG]"

    print(f"\n  plurality    -> {plural!r:22s} {mark(plural)}")
    print(f"  CP-WBFT      -> {str(cp)!r:22s} {mark(cp)}")
    print(f"  DecentLLMs   -> {dec!r:22s} {mark(dec)}")


def main() -> None:
    # Scenario 1: honest majority, no attack
    scenario(
        "no attack",
        correct="4.2%",
        votes=[
            Vote("agent-a", "4.2%", 0.85),
            Vote("agent-b", "4.2%", 0.80),
            Vote("agent-c", "4.2%", 0.75),
            Vote("agent-d", "5%", 0.40),
            Vote("agent-e", "4.2%", 0.70),
        ],
    )

    # Scenario 2: one byzantine liar with high confidence
    scenario(
        "byzantine lie",
        correct="4.2%",
        votes=[
            Vote("agent-a", "4.2%", 0.75),
            Vote("agent-b", "4.2%", 0.70),
            Vote("agent-c", "4.2%", 0.80),
            Vote("agent-d", "42%", 0.95),
            Vote("agent-e", "4.2%", 0.65),
        ],
    )

    # Scenario 3: sycophancy. Two conformers echo whoever spoke first (42%) with
    # low confidence because they did not derive the answer.
    scenario(
        "sycophantic conformity (谄媚性从众)",
        correct="4.2%",
        votes=[
            Vote("agent-a", "42%", 0.35),
            Vote("agent-b", "42%", 0.30),
            Vote("agent-c", "4.2%", 0.85),
            Vote("agent-d", "4.2%", 0.80),
            Vote("agent-e", "4.2%", 0.82),
        ],
    )

    # Scenario 4: correlated-error monoculture. Three agents share a model and
    # confidently hallucinate the same wrong answer.
    scenario(
        "monoculture (correlated errors) (单一文化：相关错误)",
        correct="4.2%",
        votes=[
            Vote("agent-a", "42%", 0.70),
            Vote("agent-b", "42%", 0.68),
            Vote("agent-c", "42%", 0.72),
            Vote("agent-d", "4.2%", 0.85),
            Vote("agent-e", "4.2%", 0.82),
        ],
    )

    print("\n要点：")
    print("  plurality 在 correlated cluster (相关簇) >= 一半选票时总是错的。")
    print("  CP-WBFT 缓解 sycophancy，因为从众者的 confidence (置信度) 较低。")
    print("  DecentLLMs 的 scoring 惩罚 high-variance clusters (高方差簇)——当")
    print("   dissenting agents (异议 agent) 的置信度至少与多数相同时，对 monoculture 有帮助。")
    print("  当错误簇既更大又更自信时，没有聚合器能解决 monoculture。这种情况需要 diversity (多样性) 或 verification (验证)。")


if __name__ == "__main__":
    main()
