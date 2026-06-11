# 近端策略优化 (Proximal Policy Optimization, PPO)

> A2C 每次更新后丢弃每个展开。PPO 将策略梯度包裹在裁剪的重要性比率中，因此你可以在相同数据上进行 10+ 轮次而不会策略爆炸。Schulman 等人（2017）。2026 年仍然是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**先决条件：** Phase 9 · 06 (REINFORCE)，Phase 9 · 07 (Actor-Critic)
**时间：** ~75 分钟

## 问题

A2C（第 07 课）是同策略的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。取一次更新，`π_θ` 变化；你使用的数据现在变成离线策略。重用它你的梯度就有偏。

展开很昂贵。在 Atari 上，一个展开跨越 8 个环境 × 128 步 = 1024 个转移和十几秒的环境时间。一次梯度步后就丢弃是浪费。

信任区域策略优化 (Trust Region Policy Optimization, TRPO, Schulman 2015) 是第一个修复：约束每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论上干净，但每次更新需要共轭梯度求解。2026 年没人运行 TRPO。

PPO（Schulman 等人 2017）用简单的裁剪目标替换硬信任区域约束。多一行代码。每展开 10 轮次。没有共轭梯度。足够好的理论保证。九年后，它仍然是从 MuJoCo 到 RLHF 的所有事物的默认策略梯度算法。

## 概念

![PPO 裁剪替代目标：在 1 ± ε 处比率裁剪](../assets/ppo.svg)

**重要性比率 (Importance ratio)。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略 vs 收集数据的策略的似然比。`r_t = 1` 意味着没有变化。`r_t = 2` 意味着新策略采取 `a_t` 的可能性是旧策略的两倍。

**裁剪替代目标 (Clipped surrogate)。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图超过 `1 + ε`，裁剪压平梯度——不要把好动作推到比旧概率高 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图超过 `1 - ε`（意味着我们会使坏动作相对于其裁剪后的减少更可能），裁剪限制梯度——不要把坏动作推到 `-ε` 以下。

`min` 处理另一个方向：如果比率已向*有益*方向移动，你仍然获得梯度（对你有害的那一侧不裁剪）。

典型 `ε = 0.2`。将目标作为 `r_t` 的函数绘制：在"好侧"有平顶、在"坏侧"有平底的分段线性函数。

**完整 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中收集 `N × T` 个转移，每个环境 `T` 步。
2. 计算优势（GAE），将它们冻结为常数。
3. 冻结 `π_{θ_old}` 作为当前 `π_θ` 的快照。
4. 对于 `K` 轮次，对于每个小批量 `(s, a, A, V_target, log π_old(a|s))`：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃展开。返回步骤 1。

`K = 10` 和小批量 64 是标准超参数集。PPO 很稳健：精确数字在 ±50% 内很少重要。

**KL 惩罚变体。** 原始论文提出了使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，`β` 基于观察到的 KL 调整。裁剪版本成为主导；KL 变体在 RLHF 中存活（其中对参考策略的 KL 是你无论如何都想要的单独约束）。

## 构建

### 步骤 1：在展开时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在展开时拍摄一次。在更新轮次期间不变化。

### 步骤 2：计算 GAE 优势（第 07 课）

与 A2C 相同。在批次上归一化。

### 步骤 3：裁剪替代更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 -surrogate，添加价值损失，减去熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 已裁剪
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"裁剪 → 零梯度"模式是 PPO 的核心。如果新策略已在有益方向漂移太远，更新停止。

### 步骤 4：价值和熵

在演员上添加标准 MSE 到评论器目标和熵奖励，与 A2C 相同。

### 步骤 5：诊断

每次更新关注三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]` 内。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **裁剪比例** —— 比率位于 `[1-ε, 1+ε]` 外的样本比例。应为 `~0.1-0.3`。如果 `~0`，裁剪从不触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，你在过拟合展开 → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。评论器质量指标。应随评论器学习向 1 攀升。

## 陷阱

- **裁剪系数调错。** `ε = 0.2` 是事实标准。降到 `0.1` 使更新太胆小；`0.3+` 邀请不稳定性。
- **轮次太多。** `K > 20` 经常不稳定，因为策略远离 `π_old`。限制轮次，尤其对大型网络。
- **没有奖励归一化。** 大奖励尺度侵蚀裁剪范围。在计算优势前归一化奖励（运行标准差）。
- **忘记优势归一化。** 每批次零均值/单位标准差归一化是标准。跳过它在大多数基准上破坏 PPO。
- **学习率不衰减。** PPO 受益于线性 LR 衰减到零。常数 LR 通常更差。
- **重要性比率数学错误。** 始终用 `exp(log_new - log_old)` 保证数值稳定性，而不是 `new / old`。
- **梯度符号错误。** 最大化替代目标 = *最小化* `-L^{CLIP}`。翻转符号是最常见的 PPO bug。

## 应用

PPO 是 2026 年跨惊人数量领域的默认 RL 算法：

| 用例 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 带高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 带分类策略的 PPO，滚动 128 步展开 |
| 大语言模型 RLHF | 带对参考模型 KL 惩罚的 PPO，响应末端来自 RM 的奖励 |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理大语言模型 | GRPO（第 12 课）—— 无评论器的 PPO 变体 |
| 仅偏好数据 | DPO —— PPO+KL 的闭式坍塌，无在线采样 |

PPO *损失形状* —— 裁剪替代目标 + 价值 + 熵 —— 是 DPO、GRPO 和几乎每个 RLHF 管道的脚手架。

## 交付

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，`ε=0.2, K=4`。在匹配环境步数下与 A2C（每展开一轮次）比较样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数并跟踪每次更新的平均 KL。在这个任务上 KL 在什么 `K` 爆炸？
3. **困难。** 用自适应 KL 惩罚替换裁剪替代目标（如果 `KL > 2·target` 则 `β` 翻倍，如果 `KL < target/2` 则减半）。比较最终回报、稳定性和无裁剪性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 重要性比率 (Importance ratio) | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；偏离收集数据的策略。 |
| 裁剪替代目标 (Clipped surrogate) | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；有益侧超过裁剪后梯度变平。 |
| 信任域 (Trust region) | "TRPO / PPO 意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚 (KL penalty) | "软信任域" | PPO 替代方案：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| 裁剪比例 (Clip fraction) | "裁剪触发的频率" | 诊断 —— 应为 0.1-0.3；超出意味着调错。 |
| 多轮次训练 (Multi-epoch training) | "数据重用" | 每次展开 K 轮次；方差成本交换样本效率。 |
| 类同策略 (On-policy-ish) | "大部分同策略" | PPO 名义上是同策略但 K>1 轮次安全地使用略微离线策略的数据。 |
| PPO-KL | "另一个 PPO" | KL 惩罚变体；用于 RLHF 其中对参考的 KL 已经是约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 每个 PPO 超参数的消融。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中的 PPO 配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 带 PyTorch 的清晰现代阐述。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型上 PPO 的生产配方；与第 09 课（RLHF）一起阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 个代码级优化"论文；哪些 PPO 技巧是承重的，哪些是民间传说。
