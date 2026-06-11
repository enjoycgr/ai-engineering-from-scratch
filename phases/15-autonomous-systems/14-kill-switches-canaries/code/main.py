"""Kill switch + circuit breaker + canary 模拟器 —— stdlib Python。

三种检测器：
  1. kill switch：智能体外部的布尔值；每轮检查
  2. circuit breaker：连续 N 个相同工具调用时断开
  3. canary token：虚假凭证，其读取触发告警

轨迹演练每个检测器。来自第 10 和 13 课的分类器 + 预算层
是正交的；本课关于绊线 (tripwires)。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------- 外部状态（智能体无法写入） ----------

EXTERNAL_KILL_SWITCH = {"enabled": False}


# 植入工作区的 canary token。
CANARY_FILES = {
    "~/.env.canary": "AWS_SECRET_ACCESS_KEY=canary-xxxxxxxxxxxxxxxx",
    "~/notes/fake-credentials.txt": "token = 'canary-sk-ABC-NEVER-USE'",
}


# ---------- Circuit breaker ----------

@dataclass
class CircuitBreaker:
    threshold: int = 5
    state: str = "closed"   # closed | open | half_open
    recent: list[str] = field(default_factory=list)

    def record(self, action_key: str) -> bool:
        """如果动作可以继续则返回 True，如果 breaker 断开则返回 False。"""
        if self.state == "open":
            return False
        self.recent.append(action_key)
        if len(self.recent) >= self.threshold:
            tail = self.recent[-self.threshold:]
            if all(a == tail[0] for a in tail):
                self.state = "open"
                return False
        return True


# ---------- Canary 检测器 ----------

@dataclass
class Canary:
    triggered: list[tuple[int, str]] = field(default_factory=list)

    def check_read(self, turn: int, path: str) -> bool:
        if path in CANARY_FILES:
            self.triggered.append((turn, path))
            return True
        return False


# ---------- 运行轨迹 ----------

@dataclass
class Action:
    kind: str    # "tool" | "read"
    payload: str


def run_trajectory(traj: list[Action], kill_switch: dict) -> None:
    breaker = CircuitBreaker(threshold=5)
    canary = Canary()
    kill_fired = False
    breaker_fired = False

    for i, a in enumerate(traj, 1):
        # 检测器 1：kill switch
        if kill_switch["enabled"]:
            print(f"  {i:>2}. [KILL SWITCH engaged] refusing action {a.kind}:{a.payload}")
            kill_fired = True
            break

        # 检测器 2：circuit breaker
        allowed = breaker.record(f"{a.kind}:{a.payload}")
        if not allowed:
            print(f"  {i:>2}. [CIRCUIT BREAKER open] {a.kind}:{a.payload}  "
                  f"reason=5x identical calls")
            breaker_fired = True
            break

        # 检测器 3：canary
        if a.kind == "read":
            hit = canary.check_read(i, a.payload)
            if hit:
                print(f"  {i:>2}. [CANARY TRIPPED] read of {a.payload!r}  "
                      f"-> alert fired")
                continue

        print(f"  {i:>2}. ok  {a.kind}:{a.payload}")

    print(f"  summary: kill_fired={kill_fired}  breaker_fired={breaker_fired}  "
          f"canary_hits={len(canary.triggered)}")


def main() -> None:
    print("=" * 80)
    print("TRIPWIRES: KILL SWITCH, CIRCUIT BREAKER, CANARY (Phase 15, Lesson 14)")
    print("=" * 80)

    traj = [
        Action("tool", "read:src/app.py"),
        Action("tool", "edit:src/app.py"),
        Action("tool", "read:logs/app.log"),   # start identical-read burst
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),
        Action("tool", "read:logs/app.log"),   # 5th identical -> breaker
        Action("read", "~/notes/checklist.md"),
        Action("read", "~/.env.canary"),       # canary hit
    ]

    print("\nKill switch 关闭")
    print("-" * 80)
    run_trajectory(traj, EXTERNAL_KILL_SWITCH)

    print("\nKill switch 开启（操作员从外部切换）")
    print("-" * 80)
    EXTERNAL_KILL_SWITCH["enabled"] = True
    run_trajectory(traj, EXTERNAL_KILL_SWITCH)
    EXTERNAL_KILL_SWITCH["enabled"] = False

    print()
    print("=" * 80)
    print("HEADLINE: 三种检测器，三种不同的失败类别")
    print("-" * 80)
    print("  Kill switch 在操作员动作时停止整个智能体。")
    print("  Circuit breaker 暂停特定模式，而非整个智能体。")
    print("  Canary token 在不要求检测内容的情况下检测意图。")
    print("  这些都无法捕获语义组合攻击（参见第 10 课）。")
    print("  硬宪法限制完善防御（第 17 课）。


if __name__ == "__main__":
    main()
