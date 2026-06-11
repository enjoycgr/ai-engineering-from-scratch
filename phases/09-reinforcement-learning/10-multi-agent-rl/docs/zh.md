# 多智能体强化学习 (Multi-Agent RL)

> 单智能体 RL 假设环境是平稳的。将两个学习智能体放入同一个世界，这个假设就失效了：每个智能体都是另一个环境的一部分，而且两者都在变化。多智能体 RL 是使学习在马尔可夫假设不再成立时收敛的技巧集合。

**类型：** 构建
**语言：** Python
**先决条件：** Phase 9 · 04 (Q-learning)，Phase 9 · 06 (REINFORCE)，Phase 9 · 07 (Actor-Critic)
**时间：** ~45 分钟

## 问题

机器人学习在房间中导航是单智能体 RL 问题。足球队不是。AlphaStar 对星际争霸对手不是。竞价智能体的市场不是。两辆车协商四向停车不是。许多对多现实问题都不是。

在每个多智能体设置中，从任何一个智能体的视角，其他智能体*是*环境的一部分。随着它们学习和改变行为，环境变得非平稳。马尔可夫性质——"下一状态仅取决于当前状态和我的动作"——被违反，因为下一状态还取决于*其他*智能体选择了什么，而它们的策略是移动目标。

这破坏了表格收敛证明（Q-learning 的保证假设平稳环境）。它也破坏了朴素的深度 RL：智能体在循环中追逐彼此，从不收敛到稳定策略。你需要多智能体特定技术：中心化训练 / 去中心化执行、反事实基线、联盟博弈、自我博弈。

2026 年应用：机器人群、交通路由、自动驾驶车队、市场模拟器、多智能体大语言模型系统（第 16 阶段），以及任何有超过一个智能玩家的游戏。

## 概念

![四种 MARL 机制：独立、中心化评论器、自我博弈、联盟](../assets/marl.svg)

**形式化：马尔可夫博弈 (Markov Game)。** MDP 的推广：状态 `S`，联合动作 `a = (a_1, …, a_n)`，转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在其自身策略 `π_i` 下最大化自身回报。如果奖励相同，它是**完全合作的**。如果是零和，它是**对抗的**。如果是混合的，它是**一般和**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角，`P(s' | s, a_i)` 依赖 `π_{-i}`，而它在变化。
- **信用分配。** 共享奖励时，哪个智能体造成了它？
- **探索协调。** 智能体必须探索互补策略，而非冗余地探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 指数增长。
- **部分可观察性。** 每个智能体只看到自身观察；全局状态被隐藏。

**四种主导机制：**

**1. 独立 Q-learning / 独立 PPO (IQL, IPPO)。** 每个智能体学习自己的 Q 或策略，将其他视为环境的一部分。简单，有时有效（尤其当经验回放充当平滑智能体建模技巧时）。理论收敛：无。实践中：对松散耦合任务不错，对紧密耦合任务糟糕。

**2. 中心化训练，去中心化执行 (Centralized Training, Decentralized Execution, CTDE)。** 最常见的现代范式。每个智能体有自己的*策略* `π_i`，基于局部观察 `o_i` 条件——部署时标准去中心化执行。在*训练*期间，中心化评论器 `Q(s, a_1, …, a_n)` 基于完整全局状态和联合动作条件。例子：
- **MADDPG**（Lowe 等人 2017）：每个智能体带中心化评论器的 DDPG。
- **COMA**（Foerster 等人 2017）：反事实基线——问"如果我采取动作 `a'` 而不是，我的奖励会是多少？"——隔离我的贡献。
- **MAPPO** / **带共享评论器的 IPPO**（Yu 等人 2022）：带中心化价值函数的 PPO。2026 年合作 MARL 的主导方法。
- **QMIX**（Rashid 等人 2018）：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` 带单调混合。

**3. 自我博弈 (Self-play)。** 同一智能体的两个副本互相对战。对手的策略*是*我过去快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。对零和游戏最有效；训练信号是对称的。

**4. 联盟博弈 (League play)。** 自我博弈在一般和 / 对抗环境中的扩展：保留过去和当前策略的群体，从联盟中采样对手，针对它们训练。添加利用者（专门击败当前最佳）和主利用者（专门击败利用者）。AlphaStar（星际争霸 II）。当游戏允许"石头剪刀布"策略循环时需要。

**通信。** 允许智能体相互发送学习到的消息 `m_i`。在合作设置中有效。Foerster 等人（2016）表明可微智能体间通信可以端到端训练。今天基于大语言模型的多智能体系统（第 16 阶段）本质上用自然语言通信。

## 构建

本课使用一个 6×6 GridWorld，有两个合作智能体。它们从对角开始，必须到达共享目标。共享奖励：任一智能体仍在移动时每步 `-1`，两者到达时 `+10`。参见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 两个智能体

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 步骤 2：独立 Q-learning

每个智能体运行自己的以联合状态为键的 Q 表。每步：两者选择 ε-贪婪动作，收集联合转移，各自用共享奖励更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在此任务上有效，因为奖励密集且对齐。在紧密耦合任务上失败（例如，一个智能体必须*等待*另一个）。

### 步骤 3：联合 Q 与分解价值更新

在联合动作上使用一个 Q：`Q(s, a_1, a_2)`。从共享奖励更新。执行时通过边缘化去中心化：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数级联合动作空间换取*正确的*全局视图。

### 步骤 4：简单自我博弈（对抗 2 智能体）

同一智能体，两个角色。训练智能体 A 对智能体 B；`K` 个片段后，将 A 的权重复制到 B。对称训练，一致进步。微缩版 AlphaZero 配方。

## 陷阱

- **非平稳回放。** 独立智能体的经验回放比单智能体更糟，因为旧转移由现在过时的对手生成。修复：重标记或按新近度加权。
- **信用分配模糊。** 长片段后的共享奖励；没有明确方式说哪个智能体贡献了。修复：反事实基线（COMA），或每个智能体的奖励塑造。
- **策略漂移 / 追逐。** 每个智能体的最佳响应随彼此的更新而变化。修复：中心化评论器、慢学习率，或一次冻结一个。
- **通过协调的奖励黑客。** 智能体找到设计者未预料到的协调漏洞。竞价智能体收敛到出价零。修复：仔细的奖励设计、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵奖励，或角色条件化。
- **联盟循环。** 纯自我博弈可能卡在主导循环中。修复：带多样化对手的联盟博弈。
- **样本爆炸。** `n` 智能体 × 状态空间 × 联合动作。用函数逼近近似；分解动作空间（每个智能体一个策略输出头）。

## 应用

2026 年 MARL 应用地图：

| 领域 | 方法 | 备注 |
|--------|--------|-------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享评论器 + 去中心化演员。 |
| 双人游戏（国际象棋、围棋、扑克） | 带 MCTS 的自我博弈 (AlphaZero) | 零和；对称训练。 |
| 复杂多人游戏（Dota、星际争霸） | 联盟博弈 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | 带注意力的 CTDE MAPPO / PPO | 部分观察；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | `n → ∞` 时的平均场 RL。 |
| 大语言模型多智能体系统（第 16 阶段） | 自然语言通信 + 角色条件化 | 智能体规划层的 RL 循环。 |

2026 年，MARL 最大的增长领域是基于大语言模型的：语言模型智能体群协商、辩论、构建软件。RL 表现为*轨迹级*输出的偏好优化，而非 token 级（第 16 阶段 · 03）。

## 交付

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2 智能体合作 GridWorld 上训练独立 Q-learning。平均回报 > 0 需要多少片段？绘制联合学习曲线。
2. **中等。** 添加"协调"任务：仅当两个智能体同时踏上目标时才到达目标。独立 Q 仍然收敛吗？什么破坏了？
3. **困难。** 为 MAPPO 风格训练实现中心化评论器，并与协调任务上的独立 PPO 比较收敛速度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 马尔可夫博弈 (Markov game) | "多智能体 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "中心化训练，去中心化执行" | 训练时联合评论器；每个智能体的策略仅使用局部观察。 |
| IPPO | "独立 PPO" | 每个智能体单独运行 PPO。简单基线；常被低估。 |
| MAPPO | "多智能体 PPO" | 基于全局状态的 PPO 带中心化价值函数。 |
| QMIX | "单调价值分解" | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许去中心化 argmax。 |
| COMA | "反事实多智能体" | 优势 = 我的 Q 减去对我的动作边缘化的期望 Q。 |
| 自我博弈 (Self-play) | "智能体对过去的自己" | 单个智能体，两个角色；零和游戏的标准。 |
| 联盟博弈 (League play) | "群体训练" | 缓存过去策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带中心化评论器的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性的价值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 对 MARL 出奇地强。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联盟博弈。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和游戏中的纯自我博弈。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包括教科书对多智能体设置和 CTDE 旨在解决的非平稳性问题的简短处理。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 及收敛结果的综述。
