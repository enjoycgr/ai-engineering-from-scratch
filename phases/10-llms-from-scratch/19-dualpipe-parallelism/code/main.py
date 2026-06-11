"""Pipeline schedule simulator — 1F1B vs Zero Bubble vs DualPipe vs DualPipeV.

教学工具。针对给定 (P, micro_batches) 计算每种调度的流水线气泡。
输出：
  - 固定 (P, micro_batches) 下每种调度的气泡比例
  - micro_batches 增长时气泡的缩放

非生产模拟器。前向/后向块成本已单位归一化。
通信成本建模为重叠窗口，非完整 kernel 模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ScheduleStats:
    name: str
    stable_bubble_frac: float
    scales_with_micro_batches: bool
    param_copies: int
    comm_overlap: str


def bubble_1f1b(P: int, M: int) -> float:
    """1F1B：warmup 阶段有 (P-1) 个无后向重叠的前向槽。
    Cooldown 镜像。稳定阶段每个 rank 每个 micro-batch 零气泡，
    但 warmup/cooldown 气泡是每个 rank (P-1) 前向 + (P-1) 后向块，
    总计 2 * M + 2 * (P - 1) 块。
    """
    total = 2 * M + 2 * (P - 1)
    bubble = 2 * (P - 1)
    return bubble / total


def bubble_zero_bubble(P: int, M: int) -> float:
    """Zero Bubble (Qi 2023) 将后向分成 B + W。W 部分可以填充
    1F1B 气泡。近似残余气泡是 warmup (P - 1) / 2 块加上
    相同的 cooldown，总计 3 * M + 2 * (P - 1) 子块。
    """
    total = 3 * M + 2 * (P - 1)
    bubble = (P - 1)
    return bubble / total


def bubble_dualpipe(P: int, M: int) -> float:
    """DualPipe 从流水线两端注入 micro-batch。稳定阶段
    气泡为零。Warmup/cooldown 有固定气泡，独立于 M。
    """
    total = 3 * M + (P - 1)
    bubble = (P - 1) // 2
    return bubble / total


def bubble_dualpipev(P: int, M: int) -> float:
    """DualPipeV 在单份参数拷贝上使用 V 形调度。
    其气泡略大于 DualPipe 的，好处是内存减半。
    近似为 1.2 倍 DualPipe 气泡。"""
    return bubble_dualpipe(P, M) * 1.2


def summarize(P: int, M: int) -> List[tuple[str, float, int, str]]:
    return [
        ("1F1B",       bubble_1f1b(P, M),        1, "minimal"),
        ("Zero Bubble", bubble_zero_bubble(P, M), 1, "partial"),
        ("DualPipe",   bubble_dualpipe(P, M),    2, "full"),
        ("DualPipeV",  bubble_dualpipev(P, M),   1, "partial"),
    ]


def gpu_hours_recovered(P: int, M: int, total_gpu_hours: float) -> dict:
    b1 = bubble_1f1b(P, M)
    bd = bubble_dualpipe(P, M)
    recovered = (b1 - bd) * total_gpu_hours
    return {
        "1F1B_bubble_frac": b1,
        "DualPipe_bubble_frac": bd,
        "recovered_gpu_hours": recovered,
    }


def main() -> None:
    print("=" * 70)
    print("DUALPIPE PARALLELISM SIMULATOR (Phase 10, Lesson 19)")
    print("=" * 70)
    print()

    print("-" * 70)
    print("Step 1: bubble fraction at P=8, micro_batches=16")
    print("-" * 70)
    print(f"  {'schedule':<14} {'bubble':>10} {'param copies':>14} {'comm overlap':>14}")
    for name, b, pc, co in summarize(P=8, M=16):
        print(f"  {name:<14} {b:>9.1%}  {pc:>14}  {co:>14}")
    print()

    print("-" * 70)
    print("Step 2: bubble fraction scaling vs micro_batches (P=8)")
    print("-" * 70)
    header = "  " + "M".rjust(6)
    for name in ("1F1B", "ZeroBubble", "DualPipe", "DualPipeV"):
        header += name.rjust(12)
    print(header)
    for M in (4, 8, 16, 32, 64, 128):
        row = f"  {M:>6}"
        for _, b, _, _ in summarize(P=8, M=M):
            row += f"{b:>12.1%}"
        print(row)
    print()

    print("-" * 70)
    print("Step 3: bubble fraction scaling vs pipeline depth (M=64 fixed)")
    print("-" * 70)
    header = "  " + "P".rjust(6)
    for name in ("1F1B", "ZeroBubble", "DualPipe", "DualPipeV"):
        header += name.rjust(12)
    print(header)
    for P in (4, 8, 16, 32, 64):
        row = f"  {P:>6}"
        for _, b, _, _ in summarize(P=P, M=64):
            row += f"{b:>12.1%}"
        print(row)
    print()

    print("-" * 70)
    print("Step 4: recovered GPU-hours (DeepSeek-V3-shape run)")
    print("-" * 70)
    print("  DeepSeek-V3: 2048 H800 GPUs, ~2.8M GPU-hours total.")
    print("  Assume P=16 pipeline depth, M=128 micro-batches per step.")
    r = gpu_hours_recovered(P=16, M=128, total_gpu_hours=2_800_000)
    print(f"  1F1B bubble     : {r['1F1B_bubble_frac']:.1%}")
    print(f"  DualPipe bubble : {r['DualPipe_bubble_frac']:.1%}")
    print(f"  recovered       : {r['recovered_gpu_hours']:,.0f} GPU-hours")
    print(f"  (that is roughly the cost of a full 70B dense pre-training run)")
    print()

    print("takeaway: DualPipe 的气泡不随 M 增长。在 MoE 规模下，")
    print("          2 倍参数复制的成本可以收回，因为 Expert")
    print("          Parallelism 已经将主导权重分散得很薄。")
    print("          DualPipeV 以略小的气泡代价去掉了 2 倍复制。")


if __name__ == "__main__":
    main()
