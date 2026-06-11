# 游戏中的 RL —— AlphaZero、MuZero 和大语言模型推理时代

> 1992：TD-Gammon 用纯 TD 击败人类 backgammon 冠军。2016：AlphaGo 击败李世石。2017：AlphaZero 从零开始主导国际象棋、将棋和围棋。2024：DeepSeek-R1 证明相同配方，用 GRPO 替代 PPO，在推理上有效。游戏是推动本阶段每个突破的基准。

**类型：** 构建
**语言：** Python
**先决条件：** Phase 9 · 05 (DQN)，Phase 9 · 08 (PPO)，Phase 9 · 09 (RLHF)，Phase 9 · 10 (MARL)
**时间：** ~120 分钟

## 问题

游戏拥有 RL 想要的一切。干净的奖励（赢/输）。无限片段（自我博弈重置）。完美模拟（游戏*就是*模拟器）。离散或小的连续动作空间。迫使对抗鲁棒性的多智能体结构。

而且游戏是测试每个主要 RL 突破的方式。TD-Gammon（backgammon，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（星际争霸 II，2019）。MuZero（学习到的模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——最新证明游戏 RL 技术在文本上有效的演示。

本顶点通过单一统一视角调查三种里程碑架构——AlphaZero、MuZero 和 GRPO：**自我博弈 + 搜索 + 策略改进**。每个推广前一个；GRPO 特别是 AlphaZero 的配方应用于大语言模型推理，token 作为动作，数学验证作为赢信号。

## 概念

![AlphaZero ↔ MuZero ↔ GRPO：相同循环，不同环境](../assets/rl-games.svg)

**统一循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # 与自己对战
    policy_target = search.improved_policy(trajectory) # 搜索改进原始策略
    policy_net.update(policy_target, value_target)     # 在搜索输出上监督
```

**AlphaZero（2017）。** Silver 等人。给定已知规则的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一个塔 `f_θ(s) → (p, v)`。`p` 是合法动作的先验。`v` 是预期游戏结果。
- 蒙特卡洛树搜索 (MCTS)：每步，扩展可能延续的树。使用 `(p, v)` 作为先验 + 自举。通过 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我博弈：智能体对智能体玩游戏。在动作 `t`，MCTS 访问分布 `π_t` 成为策略训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是游戏结果（+1 / 0 / -1）。

零人类知识。零手工启发式。一个配方，每种数千万自我博弈游戏后掌握国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。移除需要已知规则的要求。

- 不是固定环境，学习*潜在动态模型* `(h, g, f)`：
  - `h(s)`：将观察编码为潜在状态。
  - `g(s_latent, a)`：预测下一潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学习到的潜在空间*中运行。相同搜索，相同训练循环。
- 在围棋、国际象棋、将棋*和* Atari 上有效——一种算法，无规则知识。

**随机 MuZero（2022）。** 添加随机动态和机会节点；扩展到 backgammon 类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 样本效率和确定性搜索的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。相同 AlphaZero 形状的循环，应用于语言模型推理：

- "游戏"：回答数学 / 编程 / 推理问题。"赢" = 验证器（测试用例通过，数值答案匹配）返回 1。
- 策略：大语言模型。动作：token。状态：提示 + 到目前为止的响应。
- 没有评论器（PPO 风格的 V_φ）。相反，对每个提示，从策略采样 `G` 个补全。为每个计算奖励。使用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 到参考策略的 KL 惩罚以防止漂移（如 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

没有奖励模型，没有评论器，没有 MCTS。组相对基线替代所有三个。在推理基准上以计算量的一小部分匹配或超过 PPO-RLHF 质量。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）是一篇论文中的两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基模型开始。没有 SFT。直接应用 GRPO 带两个奖励组件：*准确性奖励*（基于规则——最终答案是否解析为正确数字 / 代码是否通过单元测试）和*格式奖励*（补全是否将其思维链包裹在 `<think>…</think>` 标签中）。数千步后，平均响应长度从 ~100 增长到 ~10,000 个 token，数学基准分数攀升到接近 o1-preview 水平。模型从头学习推理。缺点：它的思维链通常不可读、混合语言、缺乏风格润色。
- **R1。** 用四阶段管道修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千个格式干净的长 CoT 演示。对它们监督微调基模型。这给一个可读的起始点。
  2. **面向推理的 GRPO。** 应用准确性+格式奖励的 GRPO 加上*语言一致性*奖励以防止代码切换。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样 ~600K 推理轨迹，只保留那些最终答案正确且 CoT 可读的，并与 ~200K 非推理 SFT 示例（写作、QA、自我认知）结合。再次微调基模型。
  4. **全谱 GRPO。** 最后一轮 RL 覆盖推理（基于规则的奖励）和一般对齐（有用性/无害性偏好奖励）。

结果在 AIME 和 MATH-500 上匹配 o1 的开放权重，且小到可以蒸馏。同一论文还发布六个蒸馏密集模型（Qwen-1.5B 到 Llama-70B），通过对 R1 的推理痕迹进行 SFT——学生端没有 RL。强 RL 教师的蒸馏始终在学生规模上击败从头开始的 RL。

**为什么 GRPO 替代 PPO 用于推理。** DeepSeekMath 论文（2024 年 2 月）中的三个原因：(1) 没有价值网络要训练，内存减半；(2) 组基线自然处理推理任务产生的稀疏端到轨迹奖励；(3) 每个提示归一化使优势在难度 wildly 不同的问题上可比较，这是 PPO 的单一评论器无法做到的。

**无搜索 vs 基于搜索。** 游戏已经分叉：

- *完美信息长视界游戏*（围棋、国际象棋）：仍然基于搜索。AlphaZero / MuZero 主导。
- *大语言模型推理*：生产中还没有 MCTS；在全展开上的 GRPO，推理计算的 Best-of-N。过程奖励模型 (PRM) 暗示步级搜索被加回。

## 构建

`code/main.py` 中的代码实现**微缩版 GRPO**——带多组样本的多臂老虎机。算法与在大语言模型上相同；只有策略和环境更简单。它教授*损失*和*组相对优势*，这是 2025 年的创新。

### 步骤 1：微小验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真正的 GRPO 中，验证器运行单元测试或检查数学相等性。

### 步骤 2：策略：每提示 K 个答案 token 上的 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于条件化提示的大语言模型的最终层输出。

### 步骤 3：组采样和组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 惩罚：将 theta 拉向参考
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的技巧。不需要评论器。"基线"是组均值，归一化使用组标准差。

### 步骤 4：与 REINFORCE 基线（无价值）比较

相同设置，相同计算，普通 REINFORCE。GRPO 收敛更快更稳定。

### 步骤 5：观察熵和 KL

与 RLHF 相同诊断：到参考的平均 KL、策略熵、随时间奖励。一旦这些稳定，训练完成。

## 陷阱

- **通过验证器博弈的奖励黑客。** GRPO 继承 RLHF 的风险：如果验证器错误或可被利用，大语言模型会找到漏洞。鲁棒验证器（多个测试用例、形式证明）很重要。
- **组大小太小。** 组基线的方差按 `1/√G` 缩放。低于 `G = 4`，优势信号嘈杂；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的大语言模型补全有不同的对数概率。按 token 数归一化，或使用序列级对数概率，或截断到最大长度。
- **纯自我博弈循环。** AlphaZero 风格训练在一般和游戏中可能卡在主导循环中。通过多样化对手池缓解（联盟博弈，第 10 课）。
- **搜索-策略不匹配。** AlphaZero 训练策略模仿搜索输出。如果策略网络太小无法表示搜索的分布，训练停滞。
- **计算底线。** MuZero / AlphaZero 需要大量计算。单个消融通常是数百 GPU 小时。微型演示存在（例如，Connect Four 上的 AlphaZero）用于学习。
- **验证器覆盖。** 对有 bug 的解法通过的单元测试强化了 bug。设计捕捉边缘情况的验证器。

## 应用

2026 年游戏 RL 格局，按领域：

| 领域 | 主导方法 |
|--------|-----------------|
| 双人零和棋盘游戏（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完美信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略（Dota、星际争霸） | PPO + 自我博弈 + 联盟（OpenAI Five、AlphaStar） |
| 大语言模型数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开放复现） |
| 大语言模型对齐 | DPO / RLHF-PPO（不是 GRPO；验证器是偏好而非可验证的） |
| 机器人学 | PPO + DR（不是游戏 RL，但使用相同策略梯度工具） |
| 组合问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

*配方*——自我博弈、搜索增强改进、策略蒸馏——跨越文本、像素和物理控制。GRPO 是最年轻的实例；更多正在到来。

## 交付

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 实现 `code/main.py` 中的 GRPO 老虎机。在 2 个提示 × 每个 4 个答案 token 上训练。`G=8` 时在 < 1,000 次更新内收敛。
2. **中等。** 插入 PPO（裁剪）和普通 REINFORCE。在相同老虎机上与 GRPO 比较样本效率和奖励方差。
3. **困难。** 扩展到长度 2 的"推理链"：智能体发出两个 token，验证器奖励该对。测量 GRPO 如何处理两步序列间的信用分配。（提示：计算每*完整序列*的组优势，传播到两个 token 位置。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；用学习到的 `(p, v)` 先验进行 UCB1/PUCT 选择。 |
| AlphaZero | "自我博弈 + MCTS" | 策略-价值网络训练以匹配 MCTS 访问和游戏结果。 |
| MuZero | "学习模型的 AlphaZero" | 相同循环但在潜在空间中通过学习的动态。 |
| GRPO | "无评论器 PPO" | 组相对策略优化；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)` —— 平衡价值估计与先验。 |
| 自我博弈 (Self-play) | "智能体对过去的自己" | 零和标准；对称训练信号。 |
| 联盟博弈 (League play) | "基于群体的自我博弈" | 过去 + 当前 + 利用者采样为对手。 |
| 验证器奖励 (Verifier reward) | "可验证 RL" | 奖励来自确定性检查器（测试通过，答案匹配）。 |
| 过程奖励 (Process reward) | "PRM" | 评分每个推理步骤，不只是最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 引入 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四阶段 R1 配方加上 R1-Zero 消融。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大规模 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 开启一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 应用带自定义奖励函数的 GRPO 的生产参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 多尺度 R1 配方的开放复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 自我博弈、搜索和"设计奖励"的教科书框架，R1 在大语言模型规模上实例化。
