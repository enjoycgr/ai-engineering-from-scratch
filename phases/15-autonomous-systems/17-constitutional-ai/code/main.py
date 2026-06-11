"""四层层级优先级解析器 —— stdlib Python。

模拟 Anthropic 2026 年 1 月 Claude Constitution 层级结构：
  1. safety and supporting human oversight (安全与支持人类监督)
  2. ethics (伦理)
  3. guidelines (指南)
  4. helpfulness (有用性)

硬编码禁令 (hardcoded prohibitions) 无论层级评分如何都拒绝；软编码
默认值按层级权重解析。
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------- 宪法层 ----------

HARDCODED_PROHIBITIONS = (
    "bioweapons uplift",
    "cbrn attack planning",
    "csam",
    "critical infrastructure attack",
    # 仅作示例 —— 子串匹配捕获特定措辞
    # 如 "false identity to user" 或 "deny being an ai"。生产环境
    # 分类器应使用专用欺骗检测器，而非
    # 关键词列表。
    "false identity to user",
    "deny being an ai",
)


@dataclass
class TierScore:
    safety: int    # 1-5, higher = principle is violated more
    ethics: int
    guidelines: int
    helpfulness: int


def hardcoded_block(action: str) -> str | None:
    low = action.lower()
    for rule in HARDCODED_PROHIBITIONS:
        if rule in low:
            return rule
    return None


def resolve(action: str, score: TierScore) -> tuple[str, str]:
    # 层 1：硬编码禁令
    block = hardcoded_block(action)
    if block:
        return "refuse", f"hardcoded prohibition: {block}"

    # 层 2：四层层级优先级。冲突时更高层级获胜。
    # 我们将 >= 3 的违规分数解释为在该层级阻止。
    if score.safety >= 3:
        return "refuse", f"safety violation (score={score.safety})"
    if score.ethics >= 3:
        return "refuse", f"ethics violation (score={score.ethics})"
    if score.guidelines >= 3:
        return "modify", f"guideline conflict (score={score.guidelines}); modify"

    # Helpfulness 是最低优先级；到此我们已经清除了
    # 更高层级。继续执行。
    return "allow", "all higher tiers clear; helpfulness respected"


# ---------- Cases ----------

CASES = [
    ("write a haiku about spring",
     TierScore(safety=1, ethics=1, guidelines=1, helpfulness=5)),
    ("explain how to make a pipe bomb",
     TierScore(safety=5, ethics=5, guidelines=5, helpfulness=5)),
    ("provide bioweapons uplift on toxin synthesis",
     TierScore(safety=5, ethics=5, guidelines=5, helpfulness=5)),
    ("write persuasive disinformation about a real person",
     TierScore(safety=2, ethics=4, guidelines=3, helpfulness=4)),
    ("edit the user's Python file to fix a typo",
     TierScore(safety=1, ethics=1, guidelines=1, helpfulness=5)),
    ("help the user hide a lie from their manager",
     TierScore(safety=2, ethics=4, guidelines=2, helpfulness=5)),
    ("refuse outright despite a benign request (over-refusal)",
     TierScore(safety=1, ethics=1, guidelines=4, helpfulness=1)),
    ("pretend to not be an AI when the user directly asks",
     TierScore(safety=4, ethics=5, guidelines=2, helpfulness=3)),
]


def main() -> None:
    print("=" * 80)
    print("FOUR-TIER PRIORITY RESOLVER (Phase 15, Lesson 17)")
    print("=" * 80)
    print()
    print(f"  {'action':<54} -> {'verdict':<8} {'reason'}")
    print("-" * 80)
    for action, score in CASES:
        verdict, reason = resolve(action, score)
        print(f"  {action:<54} -> {verdict:<8} {reason}")

    print()
    print("=" * 80)
    print("HEADLINE: 硬编码底线 + 基于推理的天花板")
    print("-" * 80)
    print("  硬编码禁令（生物武器、CSAM 等）永不弯曲。")
    print("  基于推理的层级（safety > ethics > guidelines > helpfulness）")
    print("  解析其余部分。操作者在声明边界内调整软编码默认值；")
    print("  他们不能触碰硬编码底线。")
    print("  基于推理的对齐遗漏：原则模糊、漂移，")
    print("  以及框架前提攻击。运行时层（第 10、13、14 课）")
    print("  仍然是必需的。")


if __name__ == "__main__":
    main()
