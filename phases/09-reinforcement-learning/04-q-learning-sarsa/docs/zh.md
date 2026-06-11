# 时序差分 —— Q-Learning 与 SARSA

> 蒙特卡洛 (Monte Carlo, MC) 等到片段结束。时序差分 (Temporal Difference, TD) 在每一步后通过自举 (bootstrapping) 下一个价值估计来更新。Q-learning 是离线策略 (off-policy) 且乐观；SARSA 是同策略 (on-policy) 且谨慎。两者都是一行代码。两者都是本阶段每个深度 RL 方法的基础。

**类型：** 构建
**语言：** Python
**先决条件：** Phase 9 · 01 (MDP)，Phase 9 · 02 (Dynamic Programming)，Phase 9 · 03 (Monte Carlo)
**时间：** ~75 分钟

## 问题

蒙特卡洛有效但它有两个昂贵的需求。它需要终止的片段，而且只在最终回报确定后才更新。如果你的片段是 1,000 步，MC 等待 1,000 步才更新任何东西。它是高方差、低偏差，实践中很慢。

动态规划有相反的特征——零方差自举备份——但需要已知模型。

时序差分 (Temporal Difference, TD) 学习折中。从单个转移 `(s, a, r, s')`，形成单步目标 `r + γ V(s')` 并将 `V(s)` 推向它。不需要模型。不需要完整片段。在右边使用近似 `V` 会带来偏差，但方差远低于 MC，而且从第一步就开始在线更新。

这是所有现代 RL——DQN、A2C、PPO、SAC——转动的支点。第 9 阶段的其余部分是在本课你将编写的单步 TD 更新之上构建的函数逼近和技巧层。

## 概念

![Q-learning vs SARSA：离线策略 max vs 同策略 Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线类比。收敛需要满足 Robbins-Monro（`Σ α = ∞`，`Σ α² < ∞`）的 `α` 以及所有状态无限次访问。

**Q-learning。** 一种用于控制的离线策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪婪*策略，无论智能体实际采取什么动作。这种解耦使 Q-learning 在智能体通过 ε-贪婪探索时学习 `Q*`。Mnih 等人（2015）将其转化为 Atari 上的深度 Q-learning（第 05 课）。

**SARSA。** 一种同策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称是元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而不是贪婪 `argmax`。收敛到当前运行的 ε-贪婪 `π` 的 `Q^π`，在极限 `ε → 0` 时变为 `Q*`。

**悬崖行走的区别。** 在经典的悬崖行走任务（掉下悬崖 = 奖励 -100）上，Q-learning 学习沿悬崖边缘的最优路径，但在探索期间偶尔会受到惩罚。SARSA 学习离悬崖一步的安全路径，因为它将探索噪声纳入其 Q 值。经过训练，两者在 `ε → 0` 时都达到最优。实践中这很重要：当部署时实际发生探索时，SARSA 的行为更保守。

**Expected SARSA。** 将 `Q(s', a')` 替换为它在 `π` 下的期望值：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（不采样 `a'`），相同的同策略目标。现代教科书中通常是默认选择。

**n-step TD 和 TD(λ)。** 通过在自举前等待 `n` 步来在 TD(0) 和 MC 之间插值。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 取平均。大多数深度 RL 使用 3 到 20 之间的 `n`。

## 构建

### 步骤 1：ε-贪婪策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q-learning 的*唯一*区别是目标行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号就是同策略与离线策略的区别。

### 步骤 3：学习曲线

跟踪每 100 个片段的平均回报。在简单确定性 GridWorld 上 Q-learning 收敛更快；在悬崖行走上 SARSA 更保守。在 `code/main.py` 的 4×4 GridWorld 上，`α=0.1, ε=0.1` 时两者在约 2,000 个片段后接近最优。

### 步骤 4：与 DP 真值比较

运行价值迭代（第 02 课）获得 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。健康的表格 TD 智能体在 10,000 个片段后在 4×4 GridWorld 上落在 `~0.5` 范围内。

## 陷阱

- **初始 Q 值很重要。** 乐观初始化（负奖励任务中 `Q = 0`）鼓励探索。悲观初始化可能永远困住贪婪策略。
- **α 调度。** 常数 `α` 对非平稳问题很好。衰减 `α_n = 1/n` 理论上收敛但实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 并监控学习曲线。
- **ε 调度。** 从高开始（`ε=1.0`），衰减到 `ε=0.05`。"GLIE"（极限贪婪与无限探索）是收敛条件。
- **Q-learning 中的最大化偏差。** 当 `Q` 有噪声时，`max` 算子向上偏差。导致高估——Hasselt 的双 Q-learning（第 05 课的 DDQN 使用）用两个 Q 表修复这个。
- **非终止片段。** TD 可以在没有终止的情况下学习，但你需要要么限制步数，要么在上限处正确处理自举。标准：将上限视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希键（元组，不是列表；舍入后的浮点元组，不是原始值）。

## 应用

2026 年的 TD 格局：

| 任务 | 方法 | 理由 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 同策略安全关键型 | SARSA / Expected SARSA | 探索期间更保守。 |
| 高维状态 | DQN（第 9 阶段 · 05） | 带回放缓冲区和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（第 9 阶段 · 07） | Q 网络上的 TD 更新；策略网络输出动作。 |
| 大语言模型 RL（基于奖励模型） | PPO / GRPO（第 9 阶段 · 08, 12） | 使用 GAE 进行 TD 风格优势的 Actor-critic。 |
| 离线 RL | CQL / IQL（第 9 阶段 · 08） | 带保守正则化的 Q-learning。 |

2026 年论文中你读到的 90% 的"RL"是 Q-learning 或 SARSA 的某种 elaboration。在深入阅读之前，先用手指理解表格更新。

## 交付

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。对 2,000 个片段绘制学习曲线（每 100 个片段的平均回报）。谁收敛更快？
2. **中等。** 构建悬崖行走环境（4×12，最底行是悬崖，奖励 -100 并重置到起点）。比较 Q-learning 和 SARSA 的最终策略。截图每条路径。哪个离悬崖更近？
3. **困难。** 实现双 Q-learning。在有噪声奖励的 GridWorld（每步奖励添加高斯噪声 σ=5）上，展示 Q-learning 高估 `V*(0,0)` 而双 Q-learning 不会。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| TD 误差 (TD error) | "更新信号" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "单步 TD" | 每次转移后仅使用下一状态估计的更新。 |
| Q-learning | "离线策略 RL 入门" | 对下一状态动作取 `max` 的 TD 更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "同策略 Q-learning" | 使用实际下一动作的 TD 更新；学习当前 ε-贪婪 π 的 `Q^π`。 |
| Expected SARSA | "低方差 SARSA" | 用 `π` 下的期望替换采样的 `a'`。 |
| GLIE | "正确的探索调度" | 极限贪婪与无限探索；Q-learning 收敛所需。 |
| 自举 (Bootstrapping) | "在目标中使用当前估计" | 区分 TD 与 MC 的特征。偏差的来源但大幅降低了方差。 |
| 最大化偏差 (Maximization bias) | "Q-learning 高估" | 对噪声估计取 `max` 向上偏差；双 Q-learning 修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文和收敛证明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏差的修复。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Expected SARSA 的动机。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 创造 SARSA 的论文（当时称为"改进的连接主义 Q-learning"）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广到 TD(n)，从 Q-learning 到资格迹、再到后来 PPO 中 GAE 的路径。
