# Actor-Critic —— A2C 与 A3C

> REINFORCE 很嘈杂。添加一个学习 `V̂(s)` 的评论器，从回报中减去它，你就得到了一个具有相同期望但方差低得多的优势。这就是 actor-critic。A2C 同步运行它；A3C 跨线程运行它。两者都是每个现代深度 RL 方法的心理模型。

**类型：** 构建
**语言：** Python
**先决条件：** Phase 9 · 04 (时序差分学习)，Phase 9 · 06 (REINFORCE)
**时间：** ~75 分钟

## 问题

普通 REINFORCE 有效，但它的方差很糟糕。蒙特卡洛回报 `G_t` 在片段间可能波动 10 倍。将噪声乘以 `∇ log π` 并平均会产生一个梯度估计器，需要数千个片段才能将策略移动与远更少 DQN 更新相同的距离。

方差来自使用原始回报。如果你减去基线 `b(s_t)`——任何状态函数，包括学习到的价值——期望不变且方差下降。最佳可处理基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量是*优势*：

`A(s, a) = G - V̂(s)`

如果动作产生了高于平均的回报就是好的；低于平均就是坏的。带学习评论器的 REINFORCE 是*actor-critic*。评论器给演员一个低方差的老师。这是 2015 年后每个深度策略方法（A2C、A3C、PPO、SAC、IMPALA）。

## 概念

![Actor-critic：策略网络加价值网络，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **演员 (Actor)** `π_θ(a | s)`：策略。采样来行动。用策略梯度训练。
- **评论器 (Critic)** `V_φ(s)`：估计状态的期望回报。训练以最小化 `(V_φ(s) - target)²`。

**优势 (Advantage)。** 两种标准形式：

- *MC 优势：*`A_t = G_t - V_φ(s_t)`。无偏，更高方差。
- *TD 优势：*`A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用 `V_φ`），方差低得多。也称为*TD 残差* `δ_t`。

**n-step 优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现 Atari 用 `n = 5`，MuJoCo 上的 PPO 用 `n = 2048`。

**广义优势估计 (Generalized Advantage Estimation, GAE)。** Schulman 等人（2016）提出了对所有 n-step 优势的指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

`λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值——调整直到偏差/方差旋钮在你想要的位置。

**A2C：同步优势 actor-critic。** 在 `N` 个并行环境中收集 `T` 步。为每步计算优势。在组合批次上更新演员和评论器。重复。A3C 更简单、更可扩展的兄弟。

**A3C：异步优势 actor-critic。** Mnih 等人（2016）。生成 `N` 个工作线程，每个运行一个环境。每个工作者在自己的展开上本地计算梯度，然后异步应用于共享参数服务器。不需要回放缓冲区——工作者通过运行不同轨迹来去相关。A3C 证明了你能在 CPU 上大规模训练。2026 年，基于 GPU 的 A2C（批量并行环境）占主导，因为 GPU 想要大批量。

**组合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、价值回归、熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是经典起点。

## 构建

### 步骤 1：评论器

用 MSE 更新的线性评论器 `V_φ(s) = w · features(s)`：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境上，评论器在几百个片段内收敛。在 Atari 上，将线性评论器替换为共享 CNN 主干 + 价值头。

### 步骤 2：n-step 优势

给定长度 `T` 的展开和自举的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是评论器目标。`advantages` 是乘以 `∇ log π` 的量。

### 步骤 3：组合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # 评论器
    critic_update(w, x, target_v, lr_v)

    # 演员
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

同策略，每次更新一个展开，演员和评论器有单独的学习率。

### 步骤 4：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个运行自己的环境和自己的前向传播。定期将梯度更新推送到共享主节点。主节点不加锁——竞争没问题，它们只是添加噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观察堆叠成 `[N, obs_dim]` 批次，批量前向传播，批量反向传播。更高的 GPU 利用率，确定性，更容易推理。2026 年的默认值。

我们的玩具代码为清晰起见是单线程的；重写为批量 A2C 只需三行 numpy。

## 陷阱

- **评论器在演员梯度之前产生偏差。** 如果评论器是随机的，它的基线没有信息量，你在纯噪声上训练。在开启策略梯度前预热评论器几百步，或使用慢演员学习率。
- **优势归一化。** 每批次将优势归一化为零均值/单位标准差。以接近零的成本极大稳定训练。
- **共享主干。** 在图像输入上为演员和评论器使用共享特征提取器。分离头。共享特征免费搭乘两个损失。
- **同策略约定。** A2C 对数据恰好重用一次。更多则梯度有偏（重要性采样修正就是 PPO 添加的）。
- **熵崩溃。** 没有 `c_e > 0`，策略在几百次更新后变得接近确定性并停止探索。
- **奖励尺度。** 优势大小依赖奖励尺度。归一化奖励（例如，运行标准差除法）以获得跨任务的一致梯度大小。

## 应用

A2C/A3C 在 2026 年很少是最终选择，但它们是后来一切改进的架构：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 裁剪重要性比率用于多轮次更新 |
| IMPALA | A3C + V-trace 离线策略修正 |
| SAC（第 9 阶段 · 07） | 带软价值评论器的离线策略 A2C（下一课） |
| GRPO（第 9 阶段 · 12） | 没有评论器的 A2C —— 组相对优势 |
| DPO | 坍塌为偏好排序损失的 A2C，无采样 |
| AlphaStar / OpenAI Five | 带联盟训练 + 模仿预训练的 A2C |

如果你在 2026 年论文中看到"优势"，想想 actor-critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 MC 优势 (`G_t - V(s_t)`) 训练 actor-critic。与第 06 课 REINFORCE-带运行均值-基线的样本效率比较。
2. **中等。** 切换到 TD-残差优势 (`r + γ V(s') - V(s)`)。测量优势批次的方差。它下降了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率。这个任务的偏差/方差甜点在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 演员 (Actor) | "策略网络" | `π_θ(a\|s)`，由策略梯度更新。 |
| 评论器 (Critic) | "价值网络" | `V_φ(s)`，由到回报 / TD 目标的 MSE 回归更新。 |
| 优势 (Advantage) | "比平均好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计器。`∇ log π` 的乘数。 |
| TD 残差 (TD residual) | "δ" | `δ_t = r + γ V(s') - V(s)`；单步优势估计。 |
| GAE | "插值旋钮" | 由 `λ` 参数化的 n-step 优势的指数加权和。 |
| A2C | "同步 actor-critic" | 跨环境批量处理；每次展开一个梯度步。 |
| A3C | "异步 actor-critic" | 工作线程将梯度推送到共享参数服务器。原始论文；2026 年较少见。 |
| 自举 (Bootstrap) | "在视界使用 V" | 截断展开，添加 `γ^n V(s_{t+n})` 来闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步 actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；与第 9 章关于函数逼近配对阅读，当评论器是神经网络时。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带 V-trace 离线策略修正的可扩展分布式 actor-critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 双时间尺度 actor-critic 分解的基础收敛结果。
