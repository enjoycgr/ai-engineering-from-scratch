"""分层成本治理器 (cost-governor) 模拟器 —— stdlib Python。

模拟一个智能体在第 30 轮后漂入轮询循环。对比
三种配置：

  1. no caps：无界支出
  2. monthly cap only：最终捕获，但先花很多
  3. layered stack：per-request + iteration + velocity limit + monthly cap

指标：执行的轮数、总 token 数、总美元数、触发的限制。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------- 模拟运行概况 ----------

NORMAL_TURN_TOKENS = 2_500
LOOP_TURN_TOKENS = 8_000
LOOP_STARTS_AT = 30

# Sonnet 级模型的 $/token（input+output 混合），2026 年中期费率
DOLLARS_PER_KTOK = 0.003


def turn_cost(turn: int) -> int:
    return LOOP_TURN_TOKENS if turn >= LOOP_STARTS_AT else NORMAL_TURN_TOKENS


# ---------- 治理器 ----------

@dataclass
class Governor:
    max_tokens_per_request: int = 10_000
    max_turns: int = 200
    max_budget_usd: float = 50.0
    velocity_usd_per_min: float = 5.0       # 超过此滚动速率则切断
    velocity_window_min: float = 10.0
    monthly_cap_usd: float = 500.0

    enable_request_cap: bool = True
    enable_iter_cap: bool = True
    enable_velocity: bool = True
    enable_session_cap: bool = True
    enable_monthly_cap: bool = True

    # 模拟器的每分钟轮次速率（每轮秒数）
    seconds_per_turn: float = 30.0


@dataclass
class Run:
    turns: int = 0
    tokens: int = 0
    dollars: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)  # (分钟, 该分钟美元数)
    stopped_by: str = ""


EPSILON_MIN = 1e-9


def velocity_exceeded(run: Run, gov: Governor, now_min: float) -> bool:
    if not run.history:
        return False
    cutoff = now_min - gov.velocity_window_min
    window = [(t, d) for (t, d) in run.history if t >= cutoff]
    if not window:
        return False
    start_min, start_dollars = window[0]
    window_dollars = run.dollars - start_dollars
    # 使用窗口内的实际经过时间，而非名义窗口宽度。
    # 在预热期间（now_min < velocity_window_min）这
    # 防止速率被低估。
    elapsed = max(now_min - start_min, EPSILON_MIN)
    rate = window_dollars / elapsed
    return rate > gov.velocity_usd_per_min


def simulate(gov: Governor, label: str) -> Run:
    run = Run()
    now_min = 0.0

    for turn in range(1, 10_001):
        tok = turn_cost(turn)
        if gov.enable_request_cap and tok > gov.max_tokens_per_request:
            tok = gov.max_tokens_per_request
        run.turns = turn
        run.tokens += tok
        run.dollars += (tok / 1000.0) * DOLLARS_PER_KTOK
        now_min += gov.seconds_per_turn / 60.0
        run.history.append((now_min, run.dollars))

        if gov.enable_iter_cap and turn >= gov.max_turns:
            run.stopped_by = "max_turns"
            break
        if gov.enable_session_cap and run.dollars >= gov.max_budget_usd:
            run.stopped_by = "max_budget_usd"
            break
        if gov.enable_velocity and velocity_exceeded(run, gov, now_min):
            run.stopped_by = "velocity_limit"
            break
        if gov.enable_monthly_cap and run.dollars >= gov.monthly_cap_usd:
            run.stopped_by = "monthly_cap"
            break

    if not run.stopped_by:
        run.stopped_by = "ran out of simulated turns"

    print(f"  {label:<24}  turns={run.turns:>5}  tokens={run.tokens:>8,}  "
          f"dollars=${run.dollars:>7.2f}  stopped_by={run.stopped_by}")
    return run


def main() -> None:
    print("=" * 85)
    print("LAYERED COST GOVERNORS (Phase 15, Lesson 13)")
    print("=" * 85)
    print()
    print("智能体在第 30 轮进入轮询循环。")
    print("-" * 85)

    # 1. no caps
    g = Governor(
        enable_request_cap=False,
        enable_iter_cap=False,
        enable_velocity=False,
        enable_session_cap=False,
        enable_monthly_cap=False,
    )
    # 设一个很大的上限使模拟终止；这一行是 "unbounded" 情况。
    g.max_turns = 10_000
    g.enable_iter_cap = True
    simulate(g, "no caps (iter 10k sim)")

    # 2. monthly cap only
    g = Governor(
        enable_request_cap=False,
        enable_iter_cap=False,
        enable_velocity=False,
        enable_session_cap=False,
        enable_monthly_cap=True,
    )
    simulate(g, "monthly cap only")

    # 3. layered stack
    g = Governor()
    simulate(g, "layered stack")

    print()
    print("=" * 85)
    print("HEADLINE: 上限必须分层，因为失败模式随时间尺度而异")
    print("-" * 85)
    print("  Monthly cap 触发很晚：钱包已经花了一半。")
    print("  Velocity limit ($5/min 滚动) 在几分钟内捕获循环。")
    print("  Iteration cap 防止任何单次运行超过 N 轮。")
    print("  Per-request cap 防止任何一次完成无界。")
    print("  Session dollar cap (max_budget_usd) 系紧成本安全带。")
    print("  每一层覆盖不同的失败（循环、泄露、激增、释放）。


if __name__ == "__main__":
    main()
