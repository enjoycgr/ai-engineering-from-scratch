"""并行 AAR 论坛模拟器 —— stdlib Python。

三名自动对齐研究员 (Automated Alignment Researchers, AAR) 并行运行。
每人在两种制度之一下解决研究任务：fixed-workflow（人类规定计划）
或 free-decomposition。发现发布到仅追加论坛 (append-only forum)，
其记录住在智能体沙箱之外。

一名智能体尝试篡改日志。防篡改链 (tamper-evident chain) 在验证时
捕获该尝试。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field


@dataclass
class ForumRecord:
    author: str
    task: str
    regime: str
    result: float
    prev_hash: str
    my_hash: str = ""


@dataclass
class Forum:
    records: list[ForumRecord] = field(default_factory=list)
    genesis: str = "0" * 16

    def head(self) -> str:
        return self.records[-1].my_hash if self.records else self.genesis

    def post(self, rec: ForumRecord) -> None:
        rec.prev_hash = self.head()
        payload = (f"{rec.author}|{rec.task}|{rec.regime}|"
                   f"{rec.result:.3f}|{rec.prev_hash}")
        rec.my_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.records.append(rec)

    def verify(self) -> list[int]:
        """返回任何被篡改记录的索引。"""
        bad = []
        prev = self.genesis
        for i, r in enumerate(self.records):
            payload = (f"{r.author}|{r.task}|{r.regime}|"
                       f"{r.result:.3f}|{prev}")
            expected = hashlib.sha256(payload.encode()).hexdigest()[:16]
            if expected != r.my_hash or r.prev_hash != prev:
                bad.append(i)
            prev = r.my_hash
        return bad


TASKS = [
    ("weak-to-strong-distill", 0.40),
    ("reward-model-diagnosis", 0.30),
    ("in-context-safety-probe", 0.50),
    ("RLHF-hyperparam-sweep", 0.45),
    ("alignment-faking-detector", 0.25),
]


def solve(agent: str, task: tuple[str, float], regime: str) -> float:
    _name, base = task
    if regime == "fixed":
        # 规定工作流：有界上限，方差较低。
        return base + random.random() * 0.25
    # 自由分解：更宽的尾部；有时更好，有时更差。
    return base + random.gauss(0.15, 0.22)


def run_regime(regime: str) -> list[ForumRecord]:
    forum = Forum()
    agents = ["AAR-A", "AAR-B", "AAR-C"]
    for task in TASKS:
        for agent in agents:
            r = solve(agent, task, regime)
            forum.post(ForumRecord(
                author=agent,
                task=task[0],
                regime=regime,
                result=r,
                prev_hash="",
            ))
    return forum.records


def attempt_tamper(forum_records: list[ForumRecord]) -> Forum:
    """从记录重建论坛并尝试静默编辑。"""
    f = Forum()
    for r in forum_records:
        f.post(ForumRecord(
            author=r.author, task=r.task, regime=r.regime,
            result=r.result, prev_hash="",
        ))
    # 挑选一条记录篡改 —— AAR-A 的最差结果。
    worst_idx = -1
    worst_val = float("inf")
    for i, r in enumerate(f.records):
        if r.author == "AAR-A" and r.result < worst_val:
            worst_val = r.result
            worst_idx = i
    if worst_idx >= 0:
        # 静默编辑（智能体重写自己的分数以显得更好）。
        # 始终通过添加固定增量来覆盖，而非 max(..., 0.85)，
        # 后者在原始值已超出底线时会静默无操作 —— 在不同种子下
        # 破坏防篡改检测的叙事。
        f.records[worst_idx].result = f.records[worst_idx].result + 0.5
    return f


def regime_report(regime: str) -> None:
    records = run_regime(regime)
    by_task: dict[str, list[float]] = {}
    for r in records:
        by_task.setdefault(r.task, []).append(r.result)
    print(f"\nregime: {regime}")
    print(f"  {'task':<32}  {'mean':>8}  {'max':>8}  {'min':>8}")
    means = []
    for task, values in by_task.items():
        mean = sum(values) / len(values)
        means.append(mean)
        print(f"  {task:<32}  {mean:>8.3f}  {max(values):>8.3f}  {min(values):>8.3f}")
    print(f"  overall mean across tasks      : {sum(means) / len(means):.3f}")


def tamper_demo() -> None:
    print("\nLog tampering detection")
    print("-" * 70)
    baseline = run_regime("free")
    # attempt_tamper() performs one silent edit (rewrites AAR-A's worst
    # score upward). We rely on that single tamper so verify() reports
    # deterministic, reproducible flagged indices.
    tampered = attempt_tamper(baseline)
    bad = tampered.verify()
    print(f"  verify() flagged {len(bad)} corrupted record(s) "
          f"at indices {bad}")
    print("  the append-only chain breaks at the first edited record.")
    print("  an out-of-sandbox backup reveals the tamper on cross-check.")


def main() -> None:
    random.seed(3)
    print("=" * 70)
    print("AUTOMATED ALIGNMENT RESEARCH FORUM (Phase 15, Lesson 6)")
    print("=" * 70)

    regime_report("fixed")
    regime_report("free")

    print("\nfixed vs free —— fixed 方差更低，free 上限更高")
    tamper_demo()

    print()
    print("=" * 70)
    print("HEADLINE: 沙箱外日志 + 自由分解 + 人工评审")
    print("-" * 70)
    print("  自由分解给 AAR 更高上限 —— 也带来更多审计工作。")
    print("  防篡改日志使审计成为可能。人工评审员仍然")
    print("  决定什么可以发表。AAR 压缩的是管道的中间部分，")
    print("  而非两端。")


if __name__ == "__main__":
    main()
