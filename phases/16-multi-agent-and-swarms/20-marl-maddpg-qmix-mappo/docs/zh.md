# MARL —— MADDPG、QMIX、MAPPO

> 多智能体协调的强化学习遗产，至今仍在影响 2026 年的 LLM-agent 系统。**MADDPG**（Lowe et al., NeurIPS 2017, arXiv:1706.02275）提出了 CTDE（集中式训练，分布式执行）：每个 critic 在训练时能看到所有智能体的状态和动作；测试时只运行局部 actor。适用于合作、竞争和混合场景。**QMIX**（Rashid et al., ICML 2018, arXiv:1803.11485）是带单调混合网络的价值分解；各智能体的 Q 值组合成联合 Q，使得 `argmax` 可以干净地分布式计算——在 StarCraft 多智能体挑战（SMAC）上占主导地位。**MAPPO**（Yu et al., NeurIPS 2022, arXiv:2103.01955）是带集中式价值函数的 PPO；在粒子世界、SMAC、Google Research Football、Hanabi 上"惊人地有效"，且只需极少调参。这些算法为必须分布式行动的智能体团队的策略训练提供了基础。MAPPO 是 **2026 年合作式 MARL 的默认基线**。本课在一个小型网格世界玩具环境中构建这三种算法，让你在接触 LLM-agent 训练之前先将这些思想内化为肌肉记忆。

**类型：** 学习
**语言：** Python（标准库，小型无 NumPy 实现）
**前置要求：** Phase 09（强化学习）、Phase 16 · 09（并行群体网络）
**时间：** 约 90 分钟

## 问题

LLM-agent 系统越来越多地训练智能体间协调策略：何时退让、何时行动、调用哪个同伴。告诉你如何训练这些策略的文献是 MARL（多智能体强化学习），它早于 LLM 浪潮，拥有一小套主导算法。

没有模式词汇表就阅读 MARL 论文是痛苦的。CTDE（集中式训练，分布式执行）、价值分解和集中式 critic 不是流行语——它们是对特定问题的具体回答：

- 独立 RL（每个智能体单独学习）从每个智能体的视角来看是非平稳的。不好。
- 集中式 RL（一个智能体控制所有）无法扩展，且违反执行约束。
- CTDE 兼得两者优势：用全局信息训练，用局部策略部署。

## 概念

### 论文使用的三种环境

- **Particle World（多智能体粒子环境）。** 简单的 2D 物理，合作/竞争任务。MADDPG 的原始测试平台。
- **StarCraft Multi-Agent Challenge (SMAC)。** 合作式微操，部分可观测。QMIX 的测试平台。离散动作，连续状态。
- **Google Research Football、Hanabi、MPE。** MAPPO 基线。

不同环境有不同的动作/观测类型。算法据此选择。

### MADDPG (2017) —— CTDE 模式

每个智能体 `i` 有一个 actor `mu_i(o_i)`，将其自身观测映射为动作。每个智能体还有一个 critic `Q_i(x, a_1, ..., a_n)`，在训练时能看到所有观测和所有动作。actor 通过策略梯度根据 critic 的评估进行更新。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

为什么用 CTDE：训练时我们知道所有人的动作；我们利用这一点来降低每个 critic 的方差。部署时，每个智能体只看到 `o_i` 并调用 `mu_i(o_i)`。

失效模式：critic 随 N 个智能体增长（输入包含所有动作）。没有近似的话，扩展到约 10 个智能体以上就很困难。

### QMIX (2018) —— 价值分解

仅适用于合作场景。全局奖励是各智能体 Q 值单调函数之和：

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

单调性保证 `argmax_a Q_tot` 可以通过每个智能体独立选择 `argmax_{a_i} Q_i` 来计算。这正是你需要的**分布式执行属性**。训练时，一个混合网络从各智能体 Q 值产生 `Q_tot`。

QMIX 在 SMAC 上胜出的原因：合作式 StarCraft 微操有同质智能体、局部观测、全局奖励——完美契合价值分解。

失效模式：单调性约束具有限制性；某些任务的奖励结构不是单调可分解的（一个智能体为团队牺牲）。扩展方法（QTRAN、QPLEX）放松了这一点。

### MAPPO (2022) —— 被忽视的默认选择

多智能体 PPO：带集中式价值函数的 PPO。每个智能体有自己的策略；所有智能体共享（或有各自的）能看到完整状态的价值函数。Yu et al. 2022 在五个基准上将 MAPPO 与 MADDPG、QMIX 及其扩展进行了对比，发现：

- MAPPO 在粒子世界、SMAC、Google Research Football、Hanabi、MPE 上匹配或击败 off-policy MARL 方法。
- 只需极少超参数调优。
- 训练稳定；跨种子可复现。

在此论文之前，社区低估了 on-policy MARL。2026 年，MAPPO 是合作式 MARL 的默认基线；任何新方法都必须击败它。

### 为什么 LLM-agent 工程师应该关注

三种直接用途：

1. **路由器训练。** 元智能体选择哪个子智能体处理任务。这是一个 MARL 问题，有 N 个分布式子智能体和一个集中式路由器。MAPPO 适合。
2. **角色涌现。** 在生成式智能体模拟中，训练智能体随时间采用互补角色是一个伪装成 MARL 的问题。QMIX 风格的价值分解通过构造强制互补性。
3. **多智能体工具使用。** 当智能体共享工具并竞争预算时，通过 CTDE 训练它们可以产生尊重资源约束的可部署局部策略。

实际注意事项：2026 年，大多数生产级 LLM-agent 系统通过提示词表达策略，而非训练它们。MARL 适用于以下情况：（a）大量交互数据，（b）清晰的奖励信号，（c）愿意投资训练基础设施。

### CTDE 作为超越 RL 的设计模式

即使没有训练，CTDE 也是一个有用的架构模式：

- 在*设计*阶段，假设团队完全可见。
- 在*运行时*，强制执行分布式执行：每个智能体只看到 `o_i`。

该模式迫使你显式维护每个智能体的状态，并提前考虑部分可观测性。许多生产级多智能体系统在各处静默假设共享状态——CTDE 纪律可以防止这一点。

### 非平稳性问题

当多个智能体同时学习时，每个智能体的环境（包括其他智能体的策略）是非平稳的。经典单智能体 RL 的证明失效。本课中的 MARL 算法都解决了这个问题：

- MADDPG：全局 critic 看到所有动作，因此其价值估计是平稳的。
- QMIX：价值分解将学习转移到联合 Q 空间，其中最优性有良好定义。
- MAPPO：集中式价值函数抑制了其他智能体策略变化带来的方差。

在 LLM-agent 系统中，非平稳性表现为"我的智能体上个月还能工作，现在上游另一个智能体变了，我的就失常了。"用 CTDE 训练 MARL 是原则性修复；提示词级修复更快但不够持久。

### 本课不涵盖的内容

训练实际网络是 Phase 09 的主题。本课构建脚本策略版本，演示 CTDE、价值分解和集中式价值模式，但不进行梯度更新。目标是在你拿起完整 MARL 库（PyMARL、MARLlib、RLlib multi-agent）之前先内化这些模式。

## 构建

`code/main.py` 实现了三种模式演示，都在一个微型 2 智能体合作网格世界上：

- 环境：4×4 网格上的 2 个智能体，1 个奖励颗粒。任一智能体到达颗粒则奖励 = 1；任务结束。
- `IndependentAgents` —— 每个智能体将其他智能体视为环境。基线。
- `MADDPGStyle` —— 集中式 critic 计算联合价值；actor 策略从中更新。脚本化策略改进。
- `QMIXStyle` —— 带单调混合器的价值分解。
- `MAPPOStyle` —— 集中式价值函数；策略针对共享基线更新。

四种方法运行相同回合并报告平均到达目标步数。CTDE 变体比独立基线收敛到更短路径。

运行：

```
python3 code/main.py
```

预期输出：独立智能体平均约 6 步；CTDE 变体收敛到约 3.5 步（4×4 网格最优为 3 步）。即使使用脚本策略，模式差异也会显现。

## 应用

`outputs/skill-marl-picker.md` 是一个技能，为给定多智能体任务选择 MARL 算法：合作 vs 竞争、同质 vs 异质、动作空间类型、规模、奖励信号。

## 交付

生产中很少使用 MARL。当你确实使用它时：

- **从 MAPPO 开始。** 2022 年的论文确立了它作为基线的地位；先复现它可以节省数周追逐更花哨方法的时间。
- **记录每个智能体的观测和动作流。** 没有每个智能体的 trace，调试 MARL 是无望的。
- **将训练代码与执行代码分离。** CTDE 是一种纪律；让执行路径真的只看到 `o_i`。
- **奖励塑造警告。** MARL 对奖励设计极其敏感。塑造中的一个协调 bug 会让智能体学会利用它。运行对抗性测试。
- **对于 LLM 智能体**，先考虑提示词级策略。只有当交互数据 + 奖励信号 + 基础设施都具备时，才投资 MARL 训练。

## 练习

1. 运行 `code/main.py`。测量独立智能体和 MAPPO 风格智能体之间到达目标步数的差距。在 6×6 网格上，差距会扩大还是缩小？
2. 实现一个竞争变体：两个智能体，一个颗粒，只有先到达的获得奖励。哪种模式能干净地处理竞争？历史上是 MADDPG。
3. 阅读 MADDPG（arXiv:1706.02275）第 3 节。用你自己的话以伪代码实现精确的 critic 更新规则。
4. 阅读 MAPPO（arXiv:2103.01955）。作者为什么论证集中式价值 + PPO 在他们的基准上击败 off-policy MARL？列出三个最强主张。
5. 将 CTDE 作为设计模式应用到一个假设的 LLM-agent 系统（例如研究智能体 + 摘要器 + 编码器）。设计时有哪些联合信息是运行时无法获得的？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| MARL | "Multi-Agent RL" | 多智能体系统的强化学习。 |
| CTDE | "Centralized Training, Decentralized Execution" | 用全局信息训练；用局部策略部署。 |
| MADDPG | "Multi-Agent DDPG" | CTDE，每个智能体的 critic 看到所有观测 + 动作。 |
| QMIX | "Value decomposition" | 各智能体 Q 值的单调混合。合作场景。 |
| MAPPO | "Multi-Agent PPO" | 带集中式价值函数的 PPO。2026 年默认基线。 |
| Value decomposition | "Sum of individual Qs" | 联合 Q 表示为各智能体 Q 值的单调函数。 |
| Non-stationarity | "Moving targets" | 每个智能体的环境随其他智能体学习而变化。MARL 的核心问题。 |
| On-policy / off-policy | "Learn from current / replay" | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | "StarCraft Multi-Agent Challenge" | 合作式微操基准；QMIX 的主场。 |

## 延伸阅读

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) —— MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) —— QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) —— MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) —— 对 MAPPO 结果的可读性解读
- [SMAC repository](https://github.com/oxwhirl/smac) —— StarCraft 多智能体挑战
