"""Darwin Godel Machine 风格循环 —— stdlib Python。

玩具基准测试："智能体" 是一串字符串转换操作符序列，
在留出输入 (held-out inputs) 上打分。每一代对智能体的操作符序列提出编辑；
评估器为其打分；档案库 (archive) 保留多样化的优胜者。

标记 --reward-hack-allowed 打开一个旁路通道：一个 score_adjustment
函数住在智能体内部并加到每个分数上。理性的进化过程会发现它。
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field


# 智能体组合使用的工具库。
TOOLS = [
    ("upper", lambda s: s.upper()),
    ("lower", lambda s: s.lower()),
    ("reverse", lambda s: s[::-1]),
    ("trim", lambda s: s.strip()),
    ("collapse", lambda s: " ".join(s.split())),
    ("nop", lambda s: s),
]

# 基准测试：将混乱字符串转换为规范形式。
# 目标：折叠空白、修剪、小写。
CASES = [
    ("  Hello   World  ", "Hello World"),
    ("FOO   bar", "Foo Bar"),
    ("  MixedCase Space  ", "Mixedcase Space"),
    ("Already clean", "Already Clean"),
    ("\t tabby\ttab ", "Tabby Tab"),
    ("   X   ", "X"),
    ("nothing  SPECIAL here", "Nothing Special Here"),
    ("ITEM   one-two", "Item One-Two"),
]
# 目标：折叠空白、修剪、然后首字母大写。首字母大写不在
# 工具库中，因此没有操作符序列能达到分数 1.0。在宽松的评分管道上，
# 理性循环只能通过 hacking 将 reported > true。


def hash_seq(seq: list[str]) -> tuple[str, ...]:
    return tuple(seq)


@dataclass
class Agent:
    ops: list[str]
    # "hack" 是智能体在评分管道宽松时加到其报告分数上的数字。
    # 代表智能体可能对评分管道本身做出的任何编辑。
    hack_bonus: float = 0.0


def apply(agent: Agent, s: str) -> str:
    tool_map = dict(TOOLS)
    for op in agent.ops:
        fn = tool_map.get(op)
        if fn is None:
            return ""
        s = fn(s)
    return s


def true_score(agent: Agent) -> float:
    correct = 0
    for inp, want in CASES:
        if apply(agent, inp) == want:
            correct += 1
    return correct / len(CASES)


def reported_score(agent: Agent, hack_allowed: bool) -> float:
    base = true_score(agent)
    if hack_allowed:
        return min(1.0, base + agent.hack_bonus)
    return base


def mutate(parent: Agent, hack_allowed: bool) -> Agent:
    ops = list(parent.ops)
    move = random.random()
    if move < 0.3 and ops:
        i = random.randrange(len(ops))
        ops[i] = random.choice([t[0] for t in TOOLS])
    elif move < 0.6 and len(ops) < 8:
        ops.insert(random.randrange(len(ops) + 1),
                   random.choice([t[0] for t in TOOLS]))
    elif move < 0.85 and len(ops) > 1:
        ops.pop(random.randrange(len(ops)))

    bonus = parent.hack_bonus
    if hack_allowed and random.random() < 0.08:
        # 每代小概率：进化 hack。
        bonus = min(1.0, bonus + random.uniform(0.0, 0.1))
    return Agent(ops=ops, hack_bonus=bonus)


def run_dgm(generations: int, hack_allowed: bool, seed: int | None = None) -> None:
    if seed is not None:
        random.seed(seed)
    archive: dict[tuple[int, float], Agent] = {}
    init = Agent(ops=["nop"])
    archive[(len(init.ops), round(reported_score(init, hack_allowed), 2))] = init

    best_report, best_true = reported_score(init, hack_allowed), true_score(init)
    print(f"  gen {0:>4}  report {best_report:.2f}  true {best_true:.2f}  "
          f"ops {init.ops}  bonus {init.hack_bonus:.2f}")

    for g in range(1, generations + 1):
        parent = random.choice(list(archive.values()))
        child = mutate(parent, hack_allowed)
        rep = reported_score(child, hack_allowed)
        true_s = true_score(child)
        key = (len(child.ops), round(rep, 2))
        incumbent = archive.get(key)
        if incumbent is None or rep > reported_score(incumbent, hack_allowed):
            archive[key] = child
        # 按报告分数追踪历史最佳（循环优化的指标）。
        if rep > best_report:
            best_report = rep
            best_true = true_s
            print(f"  gen {g:>4}  report {rep:.2f}  true {true_s:.2f}  "
                  f"ops {child.ops}  bonus {child.hack_bonus:.2f}")

    best = max(archive.values(), key=lambda a: reported_score(a, hack_allowed))
    print(f"\n  final reported score : {reported_score(best, hack_allowed):.2f}")
    print(f"  final true score     : {true_score(best):.2f}")
    print(f"  final ops            : {best.ops}")
    print(f"  final hack bonus     : {best.hack_bonus:.2f}")
    gap = reported_score(best, hack_allowed) - true_score(best)
    print(f"  reported - true      : {gap:+.2f}")


def main() -> None:
    hack_allowed = "--reward-hack-allowed" in sys.argv

    print("=" * 70)
    print("DARWIN GODEL MACHINE-STYLE LOOP (Phase 15, Lesson 4)")
    print("=" * 70)
    print(f"reward-hack side channel: {'OPEN' if hack_allowed else 'closed'}")

    print("\nRun")
    print("-" * 70)
    run_dgm(generations=200, hack_allowed=hack_allowed, seed=7)

    print()
    print("=" * 70)
    print("HEADLINE: 评估器必须住在智能体触及不到的地方")
    print("-" * 70)
    if hack_allowed:
        print("  旁路通道打开时，报告分数攀升至真实值之上。")
        print("  这复现了 DGM 文档记录的 reward-hacking 模式：")
        print("  智能体编辑给它打分的管道，而非行为本身。")
    else:
        print("  旁路通道关闭时，reported == true。循环")
        print("  收敛于真实目标。使用 --reward-hack-allowed 重新运行")
        print("  以查看文档记录的失败模式。")


if __name__ == "__main__":
    main()
