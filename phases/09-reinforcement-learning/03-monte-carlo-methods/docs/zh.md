# 蒙特卡洛方法 —— 从完整片段中学习

> 动态规划 (dynamic programming, DP) 需要模型。蒙特卡洛 (Monte Carlo, MC) 除了片段什么都不需要。运行策略，观察回报，取平均。强化学习 (reinforcement learning, RL) 中最简单的想法——也是解锁后续一切的关键。

**类型：** 构建
**语言：** Python
**先决条件：** Phase 9 · 01 (MDP)，Phase 9 · 02 (Dynamic Programming)
**时间：** ~75 分钟

## 问题

动态规划很优雅，但它假设你可以查询每个状态和动作的 `P(s' | s, a)`。现实世界中几乎什么都不这样工作。机器人无法解析计算关节扭矩后相机像素值的分布。定价算法无法对每个可能的客户反应进行积分。大语言模型无法枚举一个 token 后的所有可能续写。

你需要一种只需要从环境中*采样*能力的方法。运行策略。获得轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`。用它来估计价值。这就是蒙特卡洛 (Monte Carlo)。

从 DP 到 MC 的转变在哲学上很重要：我们从*已知模型 + 精确备份*转向*采样片段 + 平均回报*。方差跃升，但适用性爆炸。这节课之后的每个 RL 算法——TD、Q-learning、REINFORCE、PPO、GRPO——本质上都是蒙特卡洛估计器，有时加上自举层。

## 概念

![蒙特卡洛：展开、计算回报、平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想，一句话：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中 `G^{(i)}(s)` 是在策略 `π` 下访问 `s` 后观察到的回报。

**首次访问 (First-visit) vs 每次访问 (Every-visit) MC。** 给定一个多次访问状态 `s` 的片段，首次访问 MC 只计算首次访问后的回报；每次访问 MC 计算所有访问。两者在极限下都是无偏的。首次访问分析更简单（独立同分布样本）。每次访问每片段使用更多数据，实践中通常收敛更快。

**增量均值 (Incremental mean)。** 不存储所有回报，更新运行平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新整理：`V_new = V_old + α · (target - V_old)`，其中 `α = 1/n`。将 `1/n` 替换为常数步长 `α ∈ (0, 1)`，你就得到了一个跟踪 `π` 变化的非平稳 MC 估计器。这一步就是从 MC 跳到 TD 再到每个现代 RL 算法的全部关键。

**探索现在成了问题。** DP 通过枚举触及每个状态。MC 只看到策略访问的状态。如果 `π` 是确定性的，状态空间的整个区域永远不会被采样，它们的价值估计永远保持为零。三种修复方法，按历史顺序：

1. **探索性启动 (Exploring starts)。** 每个片段从随机的 (s, a) 对开始。保证覆盖；实践中不现实（你无法将机器人"重置"到任意状态）。
2. **ε-贪婪 (ε-greedy)。** 对当前 Q 贪婪行动，但以概率 `ε` 选择随机动作。所有状态-动作对在渐近意义下都会被采样。
3. **离线策略 MC (Off-policy MC)。** 在行为策略 `μ` 下收集数据，通过重要性采样 (importance sampling) 学习目标策略 `π`。方差高，但它是通向 DQN 等回放缓冲区方法的桥梁。

**蒙特卡洛控制 (Monte Carlo Control)。** 评估 → 改进 → 评估，就像策略迭代，但评估基于采样：

1. 运行 `π`，获得一个片段。
2. 从观察到的回报更新 `Q(s, a)`。
3. 使 `π` 对 `Q` ε-贪婪。
4. 重复。

在温和条件下（每对无限次访问，`α` 满足 Robbins-Monro），以概率 1 收敛到 `Q*` 和 `π*`。

## 构建

### 步骤 1：展开 → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有模型，只有 `env.reset()` 和 `env.step(s, a)`。与 gym 环境相同的接口，但精简版。

### 步骤 2：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，`O(T)`。反向递推 `G_t = r_{t+1} + γ G_{t+1}` 避免重复求和。

### 步骤 3：首次访问 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成工作：首次访问时标记状态，增加计数，更新运行均值。

### 步骤 4：ε-贪婪 MC 控制（同策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步骤 5：与 DP 黄金标准比较

你的 MC `V^π` 估计应该在片段数 → ∞ 时与第 02 课的 DP 结果一致。实践中：4×4 GridWorld 上 50,000 个片段可以让你在 DP 答案的 `~0.1` 范围内。

## 陷阱

- **无限片段。** MC 要求片段*终止*。如果你的策略可以永远循环，设置 `max_steps` 上限并将上限视为隐式失败。GridWorld 上的随机策略经常超时——这很正常，只要确保正确计数。
- **方差。** MC 使用完整回报。在长片段上，方差巨大——末端一个不幸的奖励会以相同幅度改变 `V(s_0)`。TD 方法（第 04 课）通过自举来削减这个。
- **状态覆盖。** 在全新 Q 上的贪婪 MC 平局时只会尝试一个动作。你*必须*探索（ε-贪婪、探索性启动、UCB）。
- **非平稳策略。** 如果 `π` 变化（如在 MC 控制中），旧回报来自不同策略。常数-α MC 处理这个；样本平均 MC 不能。
- **离线策略重要性采样。** 权重 `π(a|s)/μ(a|s)` 在整个轨迹上乘积。随视界方差爆炸。用每决策加权 IS 限制或切换到 TD。

## 应用

2026 年蒙特卡洛方法的角色：

| 用例 | 为什么用 MC |
|----------|--------|
| 短视界游戏（二十一点、扑克） | 片段自然终止；回报清晰。 |
| 记录策略的离线评估 | 对存储的轨迹取平均折扣回报。 |
| 蒙特卡洛树搜索 (AlphaZero) | 从树叶开始的 MC 展开指导选择。 |
| 大语言模型 RL 评估 | 对给定策略的采样补全计算平均奖励。 |
| PPO 中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 使用 MC `G_t`。 |
| RL 教学 | 实际工作的最简单算法——剥离自举以看清核心。 |

现代深度 RL 算法（PPO、SAC）通过 `n` 步回报或 GAE 在纯 MC（完整回报）和纯 TD（单步自举）之间插值。两个端点都是同一估计器的实例。

## 交付

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现均匀随机策略的首次访问 MC 评估。运行 10,000 个片段。将 `V(0,0)` 作为片段数的函数绘制，与 DP 答案对比。
2. **中等。** 实现 ε-贪婪 MC 控制，`ε ∈ {0.01, 0.1, 0.3}`。比较 20,000 个片段后的平均回报。曲线长什么样？偏差-方差权衡在哪里？
3. **困难。** 实现*离线策略* MC 与重要性采样：在均匀随机策略 `μ` 下收集数据，估计确定性最优策略 `π` 的 `V^π`。比较普通 IS、每决策 IS 和加权 IS。哪个方差最低？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 蒙特卡洛 (Monte Carlo) | "随机采样" | 通过对分布的独立同分布样本取平均来估计期望。 |
| 回报 `G_t` | "未来奖励" | 从步骤 `t` 到片段结束的折扣奖励和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问 MC | "每个状态只计数一次" | 只有片段中的首次访问对价值估计有贡献。 |
| 每次访问 MC | "使用所有访问" | 每次访问都贡献；略有偏差但样本效率更高。 |
| ε-贪婪 | "探索噪声" | 以概率 `1-ε` 选择贪婪动作；以概率 `ε` 选择随机动作。 |
| 重要性采样 (Importance sampling) | "修正从错误分布采样" | 通过 `π(a\|s)/μ(a\|s)` 乘积重新加权回报，从 `μ` 数据估计 `V^π`。 |
| 同策略 (On-policy) | "从我自己的数据学习" | 目标策略 = 行为策略。普通 MC、PPO、SARSA。 |
| 离线策略 (Off-policy) | "从别人的数据学习" | 目标策略 ≠ 行为策略。重要性采样 MC、Q-learning、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 经典处理。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — 首次访问 vs 每次访问分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — 离线策略 MC 和方差控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 现代低方差 IS 估计器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD 自弈收敛到超人水平的首个大规模实证演示；本阶段后半部分每节课的概念先驱。
