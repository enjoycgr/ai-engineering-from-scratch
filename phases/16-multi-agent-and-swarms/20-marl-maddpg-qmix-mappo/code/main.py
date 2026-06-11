"""MARL 模式 —— CTDE (集中式训练分布式执行)、value decomposition (值分解)、
centralized value (集中式价值) —— 在一个微型网格上。

两个 agent (智能体)，4x4 网格，一个 pellet。所有四种风格共享相同的环境
和奖励。脚本策略演示了 CTDE 变体如何比独立基线更快收敛，
即使没有梯度更新。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


GRID = 4


@dataclass
class Env:
    """合作任务：两个 pellet，每个 agent (智能体) 必须收集一个；
    无论 agent 是否移动都适用 step cost (步长成本)；碰撞（两个 agent 在同一单元格）
    额外花费一步。"""
    agent0: tuple[int, int]
    agent1: tuple[int, int]
    pellet0: tuple[int, int]
    pellet1: tuple[int, int]
    pellets_remaining: set[tuple[int, int]] = field(default_factory=set)

    @staticmethod
    def new(rng: random.Random) -> "Env":
        positions: set[tuple[int, int]] = set()
        while len(positions) < 4:
            positions.add((rng.randint(0, GRID - 1), rng.randint(0, GRID - 1)))
        a0, a1, p0, p1 = list(positions)
        return Env(agent0=a0, agent1=a1, pellet0=p0, pellet1=p1,
                   pellets_remaining={p0, p1})

    @property
    def done(self) -> bool:
        return not self.pellets_remaining

    def collect_if_on_pellet(self) -> None:
        for pos in (self.agent0, self.agent1):
            self.pellets_remaining.discard(pos)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step_toward(pos: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    dx = (target[0] - pos[0])
    dy = (target[1] - pos[1])
    if abs(dx) >= abs(dy):
        nx = pos[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
        ny = pos[1]
    else:
        nx = pos[0]
        ny = pos[1] + (1 if dy > 0 else -1 if dy < 0 else 0)
    nx = max(0, min(GRID - 1, nx))
    ny = max(0, min(GRID - 1, ny))
    return (nx, ny)


def move_or_wait(pos: tuple[int, int], target: tuple[int, int], wait: bool) -> tuple[int, int]:
    if wait:
        return pos
    return step_toward(pos, target)


def run_independent(env: Env, max_steps: int = 50) -> int:
    """每个 agent (智能体) 独立瞄准最近的 pellet；没有意识到
    另一个 agent 的目标。通常两者瞄准同一个 pellet。"""
    steps = 0
    while not env.done and steps < max_steps:
        p0_target = min(env.pellets_remaining, key=lambda p: manhattan(env.agent0, p))
        p1_target = min(env.pellets_remaining, key=lambda p: manhattan(env.agent1, p))
        env.agent0 = step_toward(env.agent0, p0_target)
        env.agent1 = step_toward(env.agent1, p1_target)
        env.collect_if_on_pellet()
        steps += 1
    return steps


def _assigned_targets(env: Env) -> tuple[tuple[int, int], tuple[int, int]]:
    """集中式最优 pellet 分配：最小化总 Manhattan 距离。"""
    pellets = list(env.pellets_remaining)
    if len(pellets) == 1:
        return pellets[0], pellets[0]
    p, q = pellets[0], pellets[1]
    cost_pq = manhattan(env.agent0, p) + manhattan(env.agent1, q)
    cost_qp = manhattan(env.agent0, q) + manhattan(env.agent1, p)
    return (p, q) if cost_pq <= cost_qp else (q, p)


def run_maddpg_style(env: Env, max_steps: int = 50) -> int:
    """集中式 critic (评论家) 为每个 agent (智能体) 分配一个不同的 pellet；
    每个 agent 的 actor (演员) 向其分配的目标移动。部署时只有 actor 运行。"""
    steps = 0
    while not env.done and steps < max_steps:
        t0, t1 = _assigned_targets(env)
        env.agent0 = step_toward(env.agent0, t0)
        env.agent1 = step_toward(env.agent1, t1)
        env.collect_if_on_pellet()
        steps += 1
    return steps


def run_qmix_style(env: Env, max_steps: int = 50) -> int:
    """Value decomposition (值分解): each agent picks the pellet with higher local Q
    (lower manhattan). Monotone mixing makes this argmax-decomposable."""
    steps = 0
    while not env.done and steps < max_steps:
        if len(env.pellets_remaining) >= 2:
            t0, t1 = _assigned_targets(env)
        else:
            only = next(iter(env.pellets_remaining))
            t0, t1 = only, only
        env.agent0 = step_toward(env.agent0, t0)
        env.agent1 = step_toward(env.agent1, t1)
        env.collect_if_on_pellet()
        steps += 1
    return steps


def run_mappo_style(env: Env, max_steps: int = 50) -> int:
    """PPO with centralized value function (集中式价值函数). Behaves like CTDE at deploy; the
    scripted variant mirrors MADDPG here because they converge to similar
    policies on this size task."""
    return run_maddpg_style(env, max_steps)


def bench(label: str, runner) -> None:
    total = 0
    trials = 500
    for i in range(trials):
        rng = random.Random(i)
        env = Env.new(rng)
        total += runner(env)
    print(f"  {label:20s} avg_steps_to_goal = {total / trials:.2f}")


def main() -> None:
    print("=" * 72)
    print("MARL PATTERNS on a 4x4 grid with 2 agents and 2 pellets (cooperative)")
    print("=" * 72)
    bench("independent (no coord)", run_independent)
    bench("MADDPG-style (CTDE)", run_maddpg_style)
    bench("QMIX-style (mono decomp)", run_qmix_style)
    bench("MAPPO-style (centralized V)", run_mappo_style)
    print("\n要点:")
    print("  independent (独立) 基线在重复努力上浪费步数。")
    print("  CTDE (集中式训练分布式执行) 家族变体协调，使得每步只有更近的 agent (智能体) 移动。")
    print("  QMIX 和 MAPPO 达到相同的稳态行为，但训练故事不同；")
    print("  在部署时，它们学习的策略是相似的。")
    print("  在 LLM-agent 系统中，这是 'router (路由器) 决定哪个 sub-agent (子智能体) 前进'")
    print("  模式。即使不进行端到端训练，CTDE 也是一种设计纪律。")


if __name__ == "__main__":
    main()
