---
name: marl-picker
description: 为给定的multi-agent task (多智能体任务)选择MARL algorithm (多智能体强化学习算法)（MADDPG、QMIX、MAPPO、IQL或扩展）。考虑cooperative vs competitive (合作 vs 竞争)、action-space type (动作空间类型)、heterogeneity (异质性)、reward structure (奖励结构)和scale (规模)。
version: 1.0.0
phase: 16
lesson: 20
tags: [multi-agent, MARL, MADDPG, QMIX, MAPPO, CTDE]
---

给定一个multi-agent task description (多智能体任务描述)，选择MARL algorithm (多智能体强化学习算法)。

生成内容：

1. **Task taxonomy (任务分类)。** Fully cooperative (完全合作)（shared reward (共享奖励)）、fully competitive (完全竞争)（zero-sum (零和)）、mixed (混合)、general-sum (一般和)。Agent数量。Homogeneous (同质) vs heterogeneous (异质)。
2. **Observability (可观测性)。** Full (完全)（每个agent看到global state (全局状态)）、partial (部分)（每个只看到own observation (自身观测)）或communication-enabled (支持通信)。
3. **Action space (动作空间)。** Discrete (离散)（Atari-like、SMAC）或continuous (连续)（particle world (粒子世界)、MuJoCo）。影响algorithm choice (算法选择)。
4. **Reward structure (奖励结构)。** Dense (密集)（per-step shaped (每步塑形)）vs sparse (稀疏)（terminal only (仅终止)）。Dense使MAPPO practical (可行)；sparse需要credit assignment help (信用分配辅助)（QMIX的value decomposition (值分解)）。
5. **Algorithm recommendation (算法推荐)。** 以MAPPO作为baseline (基线)（Yu et al. 2022）。切换至：
   - QMIX当cooperative + homogeneous + 需要strong sparse-reward credit assignment (强稀疏奖励信用分配)
   - MADDPG当mixed (cooperative + competitive) + continuous actions (连续动作)
   - Extensions (QTRAN、QPLEX、FACMAC)当monotonicity constraint (单调性约束)过于restrictive (限制)
6. **Training infrastructure (训练基础设施)。** 你是否有：足够的interaction data (交互数据)、compute budget (计算预算)、reward shaping expertise (奖励塑形专长)、stability budget (稳定性预算)（每实验5-10个seeds (种子)）？如果没有，推荐LLM agents的prompt-level policies (提示词级策略)。
7. **Deployment contract (部署契约)。** CTDE (集中式训练分布式执行)：部署时每个agent只看到local observation (局部观测)。显式写入契约，使runtime code (运行时代码)遵守它。

Hard rejects (硬性拒绝)：

- 为首次运行选择non-MAPPO baseline (非MAPPO基线)。MAPPO是2026年的baseline；从那里开始。
- 对mixed cooperative-competitive tasks (混合合作-竞争任务)使用QMIX。Value decomposition假设monotone aggregation (单调聚合)。
- 对缺乏interaction data (交互数据)或reward signal (奖励信号)的LLM-agent systems推荐MARL training。Prompt-level policies在有数据之前会outperform (表现更好)。
- 没有记录per-agent observations and actions (每智能体观测和动作)就进行training。Debugging不可能。

Refusal rules (拒绝规则)：

- 如果任务少于约1000 episodes的interaction data，推荐prompt-level policies或supervised fine-tuning (监督微调)。
- 如果任务是non-Markovian (非马尔可夫)（需要memory (记忆)）但推荐未包含recurrent critics (循环评论家)，标记gap (缺口)。
- 如果任务是general-sum competitive (一般和竞争)（multiple equilibria (多均衡)），MARL alone无法选择一个；推荐mechanism design (机制设计)或equilibrium selection (均衡选择)。

Output (输出)：一页brief (简报)。以one-sentence recommendation (单句建议)开头（"MAPPO baseline with centralized value function; per-agent discrete actor; CTDE at deploy; 5 seeds per experiment."），然后是上述七个部分。以training-to-deployment pipeline (训练到部署流水线)收尾：data collection (数据收集)、training (训练)、evaluation (评估)、rollout (上线)。
