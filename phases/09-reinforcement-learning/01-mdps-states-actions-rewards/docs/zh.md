# MDP、状态、动作与奖励

> 马尔可夫决策过程 (Markov Decision Process, MDP) 包含五个要素：状态、动作、转移、奖励和折扣因子。强化学习 (reinforcement learning, RL) 中的一切——Q-learning、PPO、DPO、GRPO——都是在这个框架上进行优化。一次学会，后续免费。

**类型：** 学习
**语言：** Python
**先决条件：** Phase 1 · 06（概率与分布），Phase 2 · 01（机器学习分类法）
**时间：** ~45 分钟

## 问题

你正在编写一个国际象棋机器人。或者库存规划器。或者交易智能体。或者训练推理模型的 PPO 循环。四个不同的领域，一个令人惊讶的事实：它们都归结为同一个数学对象。

监督学习 (supervised learning) 提供 `(x, y)` 对并要求你拟合函数。强化学习 (reinforcement learning) 不提供标签——只有一连串的状态、你采取的动作和一个标量奖励。这步棋赢了比赛吗？补货决策省钱了吗？交易盈利了吗？大语言模型 (LLM) 刚刚生成的 token 是否带来了来自评判模型的更高奖励？

在形式化之前，你无法从这个数据流中学习。"我看到了什么"、"我做了什么"、"接下来发生了什么"、"那有多好"——每一个都必须变成可以推理的对象。这种形式化就是马尔可夫决策过程 (Markov Decision Process, MDP)。本阶段中的每个 RL 算法，包括最后的 RLHF 和 GRPO 循环，都是在这个框架上进行优化。

## 概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五个对象。**

- **状态 (States)** `S`。智能体 (agent) 决策所需的一切。在 GridWorld 中，是格子位置。在国际象棋中，是棋盘。在大语言模型中，是上下文窗口加上任何记忆。
- **动作 (Actions)** `A`。可选行为。上下左右移动。走一步棋。生成一个 token。
- **转移 (Transitions)** `P(s' | s, a)`。给定状态 `s` 和动作 `a`，下一状态的分布。在国际象棋中是确定性的，在库存管理中是随机的，在大语言模型解码中几乎是确定性的。
- **奖励 (Rewards)** `R(s, a, s')`。标量信号。赢 = +1，输 = -1。收入减去成本。GRPO 中的对数似然比项。
- **折扣因子 (Discount)** `γ ∈ [0, 1)`。未来奖励相对于现在的权重。`γ = 0.99` 对应约 100 步的视界；`γ = 0.9` 对应约 10 步。

**马尔可夫性质 (Markov property)** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来只取决于当前状态。如果不满足，说明状态表示不完整——不是方法的失败，而是状态的失败。

**策略 (Policy) 与回报 (Return)。** 策略 `π(a | s)` 将状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣和。价值 (value) `V^π(s) = E[G_t | s_t = s]` 是在策略 `π` 下从状态 `s` 开始的期望回报。Q 值 (Q-value) `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从特定动作开始的期望回报。每个 RL 算法都估计这两个值之一，然后相应改进 `π`。

**贝尔曼方程 (Bellman equations)。** 本阶段所有内容使用的定点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些将期望回报分解为"这一步的奖励"加上"到达位置的折扣价值"。递归。第 9 阶段的每个算法要么迭代此方程至收敛（动态规划），要么从中采样（蒙特卡洛），或自举一步（时序差分）。

## 构建

### 步骤 1：一个简单的确定性 MDP

一个 4×4 的 GridWorld。智能体从左上角开始，右下角为终止状态，每步奖励 -1，动作集为 `{上, 下, 左, 右}`。参见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码。这就是整个环境。确定性转移、固定的步进惩罚、吸收型终止状态。

### 步骤 2：展开策略

策略是从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略 1000 次。这个 4×4 棋盘的平均回报约为 -60 到 -80。最优回报是 -6（直线路径向右下）。缩小这个差距就是第 9 阶段的全部内容。

### 步骤 3：通过贝尔曼方程精确计算 `V^π`

对于小型 MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到价值停止变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这是迭代策略评估 (iterative policy evaluation)。它是 Sutton & Barto 中的第一个算法，也是后续每个 RL 方法的理论基础。

### 步骤 4：`γ` 是具有物理意义的超参数

有效视界约为 `1 / (1 - γ)`。`γ = 0.9` → 10 步。`γ = 0.99` → 100 步。`γ = 0.999` → 1000 步。

太低会导致智能体短视。太高会导致信用分配 (credit assignment) 变得嘈杂，因为许多早期步骤共享对远期奖励的责任。LLM RLHF 通常使用 `γ = 1`，因为片段短且有界。控制任务使用 `0.95–0.99`。长视界策略游戏使用 `0.999`。

## 陷阱

- **非马尔可夫状态。** 如果你需要最近三个观察值才能决策，那么"状态"不仅仅是当前观察值。修复方法：堆叠帧（Atari 上的 DQN 堆叠 4 帧）或使用循环状态（对观察值的 LSTM/GRU）。
- **稀疏奖励 (Sparse rewards)。** 只有获胜奖励在大状态空间中几乎不可能学习。塑造奖励（中间信号）或使用模仿学习启动（第 9 阶段 · 09）。
- **奖励黑客 (Reward hacking)。** 优化代理奖励通常会产生病态行为。OpenAI 的赛艇智能体转圈收集能量块，而不是完成比赛。始终根据目标结果定义奖励，而不是代理指标。
- **折扣因子设定错误。** 在无限视界任务上 `γ = 1` 会使每个价值变为无穷大。始终用有限视界或 `γ < 1` 来限制。
- **奖励尺度。** {+100, -100} 与 {+1, -1} 的奖励给出相同的最优策略，但梯度大小完全不同。在输入 PPO/DQN 之前归一化到 `[-1, 1]` 左右。

## 应用

2026 年的技术栈将每个 RL 管道在写代码前都简化为一个 MDP：

| 场景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（运动、操作） | 关节角度 + 速度 | 连续扭矩 | 任务特定的塑造奖励 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘 + 历史 | 合法走法 | 赢=+1 / 输=-1 | 1.0（有限） |
| 库存 / 定价 | 库存 + 需求 | 订购数量 | 收入 - 成本 | 0.95 |
| LLM RLHF | 上下文 token | 下一个 token | 末端奖励模型分数 | 1.0（片段约 200 个 token） |
| GRPO 推理 | 提示 + 部分响应 | 下一个 token | 末端验证器 0/1 | 1.0 |

在写任何训练循环之前先写出五元组。大多数"RL 不起作用"的错误报告都追溯到在纸面上就已经有问题的 MDP 形式化。

## 交付

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现 4×4 GridWorld 和随机策略展开。运行 10,000 个片段。报告回报的均值和标准差。与最优回报 (-6) 比较。
2. **中等。** 对均匀随机策略运行 `policy_evaluation`，取 `γ ∈ {0.5, 0.9, 0.99}`。将每个的 `V` 打印为 4×4 网格。解释为什么终止状态附近的状态价值随更大的 `γ` 增长更快。
3. **困难。** 将 GridWorld 变为随机的：每个动作以概率 `p = 0.1` 滑向相邻方向。重新评估均匀策略。`V[start]` 会变好还是变差？为什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MDP | "强化学习设置" | 满足马尔可夫性质的五元组 `(S, A, P, R, γ)`。 |
| 状态 (State) | "智能体看到的东西" | 在选定策略类下对未来动态的充分统计量。 |
| 策略 (Policy) | "智能体的行为" | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| 回报 (Return) | "总奖励" | 从当前步骤开始的折扣和 `Σ γ^t r_t`。 |
| 价值 (Value) | "状态有多好" | 在 `π` 下从 `s` 开始的期望回报。 |
| Q 值 (Q-value) | "动作有多好" | 在 `π` 下从 `s` 以第一个动作 `a` 开始的期望回报。 |
| 贝尔曼方程 (Bellman equation) | "动态规划递归" | 将价值 / Q 分解为单步奖励加折扣后继价值的定点方程。 |
| 折扣因子 `γ` | "未来 vs 现在" | 远期奖励的几何权重；有效视界 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书。第 3 章涵盖 MDP 和贝尔曼方程；第 1 章阐述了支撑后续每节课的奖励假设。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度 RL 角度简洁的 MDP 入门。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 关于 MDP 和精确解法的运筹学参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — MDP 作为动态规划特例的最清晰推导。
