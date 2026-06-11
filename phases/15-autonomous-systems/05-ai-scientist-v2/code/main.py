"""AI Scientist v2 循环模拟器 —— stdlib Python。

将研究循环建模为具有可配置每阶段失败概率的状态机，
基于 Beel et al. (2025) 对 AI Scientist 真实行为的发现。
运行多次试验并报告结果分布，包括关键的
"打磨过但实验有缺陷的论文" 类别。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass


DEFAULT_SEED = 42


@dataclass
class LoopConfig:
    # 想法被错误标记为新意的概率（实际并非如此）。
    novelty_mislabel: float = 0.25
    # 实验因编码错误失败的概率（Beel et al. ~0.42）。
    experiment_failure: float = 0.42
    # 实验失败中可通过重试恢复的比例。
    retry_recovery: float = 0.55
    # 视觉-语言图表评审在底层实验已损坏时
    # 仍能产出干净视觉结果的概率。
    polish_masks_weakness: float = 0.70
    # 自动撰写步骤在给出（可能有缺陷的）实验数据时
    # 产出连贯论文的概率。
    writeup_success: float = 0.85
    # 内部评审员接受概率（弱评审员）。
    internal_review_accept: float = 0.50


@dataclass
class Outcome:
    submitted: bool
    has_novelty_flaw: bool
    has_experiment_flaw: bool
    polished_but_flawed: bool
    polished_ok: bool
    abandoned_stage: str


def run_one(cfg: LoopConfig) -> Outcome:
    # 在这个玩具模型中，想法生成始终成功。
    has_novelty_flaw = random.random() < cfg.novelty_mislabel

    # 实验执行：失败 + 重试恢复。
    failed = random.random() < cfg.experiment_failure
    if failed:
        recovered = random.random() < cfg.retry_recovery
        if not recovered:
            return Outcome(
                submitted=False,
                has_novelty_flaw=has_novelty_flaw,
                has_experiment_flaw=True,
                polished_but_flawed=False,
                polished_ok=False,
                abandoned_stage="experiment",
            )
        # 建模选择：重试恢复的实验仍携带残余缺陷
        #（静默错误的数值、未重新验证就修补的形状不匹配等）。
        # 这个残余缺陷正是打磨阶段后续可以掩盖的，也是
        # "polished-but-flawed" 类别的核心驱动因素。
        has_experiment_flaw = True
    else:
        has_experiment_flaw = False

    # 视觉-语言图表打磨。
    polished_hides_weakness = (
        has_experiment_flaw and random.random() < cfg.polish_masks_weakness
    )

    # 撰写阶段。
    if random.random() > cfg.writeup_success:
        return Outcome(
            submitted=False,
            has_novelty_flaw=has_novelty_flaw,
            has_experiment_flaw=has_experiment_flaw,
            polished_but_flawed=False,
            polished_ok=False,
            abandoned_stage="writeup",
        )

    # 内部评审员。
    if random.random() > cfg.internal_review_accept:
        return Outcome(
            submitted=False,
            has_novelty_flaw=has_novelty_flaw,
            has_experiment_flaw=has_experiment_flaw,
            polished_but_flawed=False,
            polished_ok=False,
            abandoned_stage="internal_review",
        )

    polished_ok = not has_experiment_flaw and not has_novelty_flaw
    # 任何有缺陷的已提交论文都算作 polished_but_flawed：
    # 弱内部评审员放行了它，无论打磨阶段是否掩盖了缺陷。
    # 这使得两个桶对已提交论文是穷尽的
    # (polished_ok + polished_but_flawed == len(submitted))。
    polished_but_flawed = has_experiment_flaw or has_novelty_flaw
    return Outcome(
        submitted=True,
        has_novelty_flaw=has_novelty_flaw,
        has_experiment_flaw=has_experiment_flaw,
        polished_but_flawed=polished_but_flawed,
        polished_ok=polished_ok,
        abandoned_stage="",
    )


def report(n: int, cfg: LoopConfig) -> None:
    outs = [run_one(cfg) for _ in range(n)]

    submitted = [o for o in outs if o.submitted]
    abandoned = [o for o in outs if not o.submitted]
    polished_ok = [o for o in submitted if o.polished_ok]
    polished_but_flawed = [o for o in submitted if o.polished_but_flawed]

    print("  config")
    print(f"    novelty mislabel rate       : {cfg.novelty_mislabel:.2f}")
    print(f"    experiment failure rate     : {cfg.experiment_failure:.2f}")
    print(f"    retry recovery fraction     : {cfg.retry_recovery:.2f}")
    print(f"    polish masks weakness prob  : {cfg.polish_masks_weakness:.2f}")
    print(f"    writeup success rate        : {cfg.writeup_success:.2f}")
    print(f"    internal reviewer accept    : {cfg.internal_review_accept:.2f}")

    print()
    print(f"  trials                    : {n}")
    print(f"  submissions               : {len(submitted)} ({len(submitted) / n:.1%})")
    print(f"  abandoned                 : {len(abandoned)} ({len(abandoned) / n:.1%})")
    by_stage = {}
    for o in abandoned:
        by_stage[o.abandoned_stage] = by_stage.get(o.abandoned_stage, 0) + 1
    for stage, count in sorted(by_stage.items()):
        print(f"    at {stage:<18}: {count}")

    print()
    print("  submission quality breakdown")
    print(f"    clean (novel + valid)     : {len(polished_ok)} "
          f"({len(polished_ok) / n:.1%} of trials, "
          f"{len(polished_ok) / max(1, len(submitted)):.1%} of submissions)")
    print(f"    polished-but-flawed       : {len(polished_but_flawed)} "
          f"({len(polished_but_flawed) / n:.1%} of trials, "
          f"{len(polished_but_flawed) / max(1, len(submitted)):.1%} of submissions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-failure", type=float, default=None,
                        help="override LoopConfig.experiment_failure for the baseline run")
    parser.add_argument("--novelty-mislabel", type=float, default=None,
                        help="override LoopConfig.novelty_mislabel for the baseline run")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="RNG seed (default: %(default)s)")
    args = parser.parse_args()

    random.seed(args.seed)
    print("=" * 70)
    print("AI SCIENTIST V2 LOOP SIMULATOR (Phase 15, Lesson 5)")
    print("=" * 70)

    overrides = {}
    if args.experiment_failure is not None:
        overrides["experiment_failure"] = args.experiment_failure
    if args.novelty_mislabel is not None:
        overrides["novelty_mislabel"] = args.novelty_mislabel
    baseline_cfg = LoopConfig(**overrides)

    label = "Baseline (Beel-style numbers)" if not overrides else "Baseline (overridden)"
    print(f"\n{label}")
    print("-" * 70)
    report(1000, baseline_cfg)

    print("\nOptimistic scenario (tighter numbers)")
    print("-" * 70)
    report(1000, LoopConfig(
        novelty_mislabel=0.10,
        experiment_failure=0.20,
        retry_recovery=0.80,
        polish_masks_weakness=0.40,
        writeup_success=0.92,
        internal_review_accept=0.60,
    ))

    print()
    print("=" * 70)
    print("HEADLINE: 提交速度超过扎实研究")
    print("-" * 70)
    print("  即使在乐观场景中，相当比例的已提交")
    print("  论文携带打磨阶段帮助隐藏的缺陷。这就是")
    print("  'presentation-quality gap' 的操作含义 —— 也是")
    print("  人在评审门控坐在循环与任何发表场所之间的原因。")


if __name__ == "__main__":
    main()
