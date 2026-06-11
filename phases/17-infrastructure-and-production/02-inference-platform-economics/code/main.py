"""推理平台经济学比较器 —— 纯 Python 标准库。

对六个供应商（Fireworks、Together、Baseten、Modal、Replicate、Anyscale）
在相同合成工作负载下进行建模。统一按 token、按分钟和按预测定价，
以便直接对比。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Vendor:
    name: str
    model: str
    per_mtok_output: float | None   # $/M 输出 token（不适用则为 None）
    per_minute: float | None        # 专属 GPU $/分钟（无服务器则为 None）
    per_prediction: float | None    # $/预测（按 token 则为 None）
    tokens_per_minute: int          # GPU 饱和时的有效 token
    cold_start_sec: float
    notes: str
    min_reserved_minutes_per_day: int = 0  # 按分钟供应商的预留分钟下限（热池 / 最低承诺）


VENDORS = [
    Vendor("Fireworks",    "Llama 70B",          0.90,  None,    None,  900_000, 1.5, "FireAttention, batch tier 50% off"),
    Vendor("Together",     "Llama 70B",          0.88,  None,    None,  850_000, 2.0, "200+ models, 50-70% below Replicate"),
    Vendor("Baseten",      "Custom Llama 70B",   None,  0.55,    None,  900_000, 5.0, "Truss, SOC2 HIPAA, per-min billing", 1440),
    Vendor("Modal",        "Custom Llama 70B",   None,  0.48,    None,  800_000, 2.5, "Python-native, per-sec billing, 60min warm-pool floor", 60),
    Vendor("Replicate",    "Llama 70B",          None,  None,    0.006, 750_000, 4.0, "Pay-per-prediction, multimodal"),
    Vendor("Anyscale",     "Llama 70B RayTurbo", None,  0.60,    None,  850_000, 3.0, "Ray-native, distributed Python", 1440),
]


def cost_per_day(v: Vendor, tokens_per_day: int, predictions_per_day: int) -> float:
    """给定供应商定价模型的有效 $/天。

    按分钟供应商按饱和服务时间和预留分钟下限（热池最低值 / 预留）
    的最大值计费。这使按分钟模型在 `run_scenario` 和
    `utilization_breakeven` 中保持一致，而非一处假设完美缩到零、
    另一处假设预留 24 小时。
    """
    if v.per_mtok_output is not None:
        return (tokens_per_day / 1e6) * v.per_mtok_output
    if v.per_minute is not None:
        saturated_minutes = tokens_per_day / v.tokens_per_minute
        minutes = max(saturated_minutes, v.min_reserved_minutes_per_day)
        return minutes * v.per_minute
    if v.per_prediction is not None:
        return predictions_per_day * v.per_prediction
    return 0.0


def effective_rate(v: Vendor, tokens_per_day: int, predictions_per_day: int) -> float:
    """统一为 $/M token 以便跨供应商比较。"""
    c = cost_per_day(v, tokens_per_day, predictions_per_day)
    return (c / (tokens_per_day / 1e6)) if tokens_per_day else 0


def run_scenario(label: str, tokens_per_day: int, predictions_per_day: int) -> None:
    print(f"\n{label}")
    print(f"Workload: {tokens_per_day/1e6:.1f}M output tokens/day  |  {predictions_per_day} predictions/day")
    header = f"{'Vendor':12}  {'Model':22}  {'$/day':>8}  {'$/M tok':>10}  Notes"
    print(header)
    print("-" * len(header))
    for v in VENDORS:
        cost = cost_per_day(v, tokens_per_day, predictions_per_day)
        rate = effective_rate(v, tokens_per_day, predictions_per_day)
        print(f"{v.name:12}  {v.model:22}  ${cost:7.2f}  ${rate:9.2f}  {v.notes}")


def utilization_breakeven() -> None:
    print("\n" + "=" * 80)
    print("PER-TOKEN vs PER-MINUTE BREAK-EVEN — Fireworks (per-token) vs Baseten (per-min)")
    print("=" * 80)
    fw = VENDORS[0]
    bt = VENDORS[2]
    print(f"Fireworks: ${fw.per_mtok_output:.2f}/M output  |  Baseten: ${bt.per_minute:.2f}/min, {bt.tokens_per_minute/1e3:.0f}k tok/min\n")
    print(f"{'Util %':>8}  {'Fireworks $/day':>16}  {'Baseten $/day':>14}  Winner")
    for util_pct in (5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100):
        tokens_per_day = int(bt.tokens_per_minute * 60 * 24 * util_pct / 100)
        fw_cost = cost_per_day(fw, tokens_per_day, 0)
        bt_cost = cost_per_day(bt, tokens_per_day, 0)
        winner = "Baseten" if bt_cost < fw_cost else "Fireworks"
        print(f"{util_pct:>7}%  ${fw_cost:>15.2f}  ${bt_cost:>13.2f}  {winner}")


def cold_start_penalty() -> None:
    print("\n" + "=" * 80)
    print("COLD START PENALTY — bursty workload")
    print("=" * 80)
    print(f"{'Vendor':12}  {'Cold start':>11}  Impact at 100 cold invocations/day")
    for v in VENDORS:
        impact_sec = v.cold_start_sec * 100
        print(f"{v.name:12}  {v.cold_start_sec:>8.1f} s   +{impact_sec:.0f} seconds/day of extra latency")


def main() -> None:
    print("=" * 80)
    print("INFERENCE PLATFORM ECONOMICS — 2026 approximations")
    print("=" * 80)

    run_scenario("场景 A —— 初创规模 LLM 产品",
                 tokens_per_day=2_000_000, predictions_per_day=10_000)
    run_scenario("场景 B —— 高吞吐量生产",
                 tokens_per_day=100_000_000, predictions_per_day=500_000)

    utilization_breakeven()
    cold_start_penalty()

    print("\nRule of thumb: under reserved-minute billing, per-minute (Baseten, Modal) beats per-token")
    print("once GPU saturation stays above ~60-70% utilization; below that, per-token wins.")


if __name__ == "__main__":
    main()
