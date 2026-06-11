---
name: swarm-optimizer
description: 为给定的LLM或agent optimization problem (优化问题)选择PSO、ACO、genetic algorithms (遗传算法)和gradient-based optimizers (基于梯度的优化器)。Bio-inspired swarm algorithms (生物启发式集群算法)是gradient-free (无梯度)的，适合LLM时代search space (搜索空间)是discrete (离散的)或fitness function (适应度函数)是black-box (黑箱)的工作负载。
version: 1.0.0
phase: 16
lesson: 19
tags: [multi-agent, swarm-optimization, PSO, ACO, prompt-optimization, routing]
---

给定一个LLM或agent optimization problem (优化问题)，选择正确的optimizer (优化器)。

生成内容：

1. **Problem fingerprint (问题指纹)。** Search space (搜索空间)（continuous numeric (连续数值)、prompt string (提示词字符串)、model weights (模型权重)、routing graph (路由图)）、fitness signal (适应度信号)（automatic test (自动测试)、LLM judge (LLM裁判)、human rater (人工评分员)、business KPI (业务KPI)）、time-to-value (价值实现时间)（minutes (分钟)、hours (小时)、days (天)）。
2. **Optimizer choice (优化器选择)。** PSO、ACO、genetic algorithm (遗传算法)、DPO/RL、manual tuning (手动调优)。每个都有默认用例：
   - continuous numeric on a bounded space (有界空间上的连续数值) → PSO
   - routing or path selection (路由或路径选择) → ACO
   - discrete symbolic / programs (离散符号/程序) → genetic algorithms (遗传算法)
   - differentiable reward (可微奖励) → DPO/RL
   - low-dimensional, fast eval (低维度、快速评估) → grid/random search (网格/随机搜索)
3. **Population sizing (种群规模)。** PSO/GA用10-30，ACO用pheromone matrix size (信息素矩阵大小)。Budget calculation (预算计算)：N × T × cost-per-eval (每次评估成本)。不要运行成本超过其产生价值的swarms (集群)。
4. **Fitness + quality gate (适应度+质量门槛)。** 什么函数给candidate (候选)评分？对于ACO routing，什么quality threshold (质量阈值)触发pheromone deposit (信息素沉积)？
5. **Convergence monitoring (收敛监控)。** 每轮迭代记录g_best或pheromone stability (信息素稳定性)。Divergence (发散)（catastrophic drift (灾难性漂移)）和premature convergence (过早收敛)（local optimum (局部最优)）时告警。
6. **Decay / exploration tuning (衰减/探索调优)。** PSO inertia (惯性)和cognitive/social weights (认知/社会权重)；ACO pheromone decay rate (信息素衰减率)和deposit amount (沉积量)。Trade-off (权衡)：low decay (低衰减) → stuck on early winner (困在早先胜者)；high decay (高衰减) → no memory (无记忆)。
7. **Reset conditions (重置条件)。** 当eval distribution (评估分布)发生shift (偏移)或deployment pattern (部署模式)变化时，临时重置g_best或清零pheromones。Stale memories (陈旧记忆)比no memories (无记忆)更糟。

Hard rejects (硬性拒绝)：

- 在fitness需要human review (人工审核)的任务上使用swarm optimizers (集群优化器)。Cost-per-iteration (每次迭代成本)使预算相形见绌。
- Population sizes > 50且没有clear budget justification (明确预算论证)。Diminishing returns (收益递减)占主导。
- 没有quality gate (质量门槛)的pheromone routing (信息素路由)。Fast-but-wrong agents (快但错的智能体)会锁定。
- 在没有natural continuous embedding (自然连续嵌入)的discrete search spaces (离散搜索空间)上使用PSO。改用GA或simulated annealing (模拟退火)。

Refusal rules (拒绝规则)：

- 如果用户试图优化没有clear fitness function (明确适应度函数)的东西，推荐先定义fitness。Swarm optimizers没有evaluator就无法帮助。
- 如果用户预算低于$100，推荐manual tuning + caching (手动调优+缓存)而非swarms。
- 如果distribution每天shift，推荐online learning (在线学习)或bandits (老虎机)，而非swarm optimizers。

Output (输出)：一页brief (简报)。以one-sentence recommendation (单句建议)开头（"Use ACO with quality-gated pheromone deposits on a 3-agent × 4-task-type routing problem. Decay 0.05, threshold 0.6, 200 warmup tasks."），然后是上述七个部分。以budget estimate (预算估算)和1-week rollout plan (一周上线计划)收尾。
