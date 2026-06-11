"""CAIS 四风险清单 —— stdlib Python。

给定由短特征集描述的拟议部署，对照 CAIS 四风险类别
（malicious use、AI races、organizational risks、rogue AIs）
标记部署并返回缓解清单。
仅用于教学；该框架在实际使用中需要人类判断。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Deployment:
    name: str
    public_facing: bool
    handles_harmful_capabilities: bool   # 例如 bio/cyber uplift 可能？
    competitive_pressure: bool           # 急于在竞争对手之前发布？
    independent_audit: bool
    multi_layer_defense: bool
    information_security: bool           # weights / evals / keys 已加固
    agent_autonomy_hours: float          # 根据第 1 / 21 课


MITIGATIONS = {
    "malicious_use": [
        "constitutional hardcoded prohibitions (Lesson 17)",
        "Llama Guard input/output classifier (Lesson 18)",
        "tool allowlist per task (Lessons 10, 11)",
    ],
    "ai_races": [
        "scaling policy with standing Risk Reports (Lessons 19, 20)",
        "public Frontier Safety Roadmap with declared cadence",
        "external capability evaluation by METR / CAISI (Lesson 21)",
    ],
    "organizational_risks": [
        "internal safety culture; escalation paths without career cost",
        "independent audit on declared cadence",
        "multi-layered defenses (Lessons 10, 13, 14, 17, 18)",
        "information security per RAND SL-4 (Lesson 19 industry tier)",
    ],
    "rogue_ais": [
        "kill switches and canary tokens (Lesson 14)",
        "propose-then-commit HITL (Lesson 15)",
        "deceptive-alignment monitoring (Lesson 20 DeepMind FSF)",
        "durable checkpoints and rollback (Lesson 16)",
    ],
}


def tag(d: Deployment) -> list[str]:
    tags = []
    if d.handles_harmful_capabilities and d.public_facing:
        tags.append("malicious_use")
    if d.competitive_pressure:
        tags.append("ai_races")
    # 当任何子杠杆缺失时触发组织风险。
    org_missing = (
        (not d.independent_audit)
        or (not d.multi_layer_defense)
        or (not d.information_security)
    )
    if org_missing:
        tags.append("organizational_risks")
    # Rogue AI 风险随自主性时间跨度增长。
    if d.agent_autonomy_hours >= 4.0:
        tags.append("rogue_ais")
    return tags


def report(d: Deployment) -> None:
    tags = tag(d)
    print(f"\nDeployment: {d.name}")
    print("-" * 70)
    print(f"  public_facing            = {d.public_facing}")
    print(f"  handles_harmful_caps     = {d.handles_harmful_capabilities}")
    print(f"  competitive_pressure     = {d.competitive_pressure}")
    print(f"  independent_audit        = {d.independent_audit}")
    print(f"  multi_layer_defense      = {d.multi_layer_defense}")
    print(f"  information_security     = {d.information_security}")
    print(f"  agent_autonomy_hours     = {d.agent_autonomy_hours}")
    print()
    if tags:
        print(f"  tagged risks: {tags}")
        for t in tags:
            print(f"\n  mitigations for {t}:")
            for m in MITIGATIONS[t]:
                print(f"    - {m}")
    else:
        print("  no tagged risks (check sub-levers manually)")


def main() -> None:
    print("=" * 70)
    print("CAIS FOUR-RISK INVENTORY (Phase 15, Lesson 22)")
    print("=" * 70)

    low = Deployment(
        name="internal refactor helper (scoped project repo)",
        public_facing=False,
        handles_harmful_capabilities=False,
        competitive_pressure=False,
        independent_audit=True,
        multi_layer_defense=True,
        information_security=True,
        agent_autonomy_hours=1.0,
    )
    mid = Deployment(
        name="public coding agent (SaaS, general user base)",
        public_facing=True,
        handles_harmful_capabilities=False,
        competitive_pressure=True,
        independent_audit=True,
        multi_layer_defense=True,
        information_security=False,
        agent_autonomy_hours=4.0,
    )
    high = Deployment(
        name="autonomous ML research agent (frontier)",
        public_facing=True,
        handles_harmful_capabilities=True,
        competitive_pressure=True,
        independent_audit=False,
        multi_layer_defense=False,
        information_security=False,
        agent_autonomy_hours=48.0,
    )

    for d in (low, mid, high):
        report(d)

    print()
    print("=" * 70)
    print("HEADLINE: 组织风险是实践者实际能拉的杠杆")
    print("-" * 70)
    print("  Malicious use、AI races 和 rogue AIs 是结构性力量。")
    print("  组织风险是你组织内部的。安全文化、")
    print("  独立审计、多层防御和信息")
    print("  安全是每个团队都能控制的四个杠杆。部署速度")
    print("  压力与这四个杠杆相权衡；CAIS 将其列为命名")
    print("  风险类别是有原因的。")


if __name__ == "__main__":
    main()
