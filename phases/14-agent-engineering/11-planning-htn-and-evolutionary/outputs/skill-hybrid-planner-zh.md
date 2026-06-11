---
name: hybrid-planner
description: 构建一个混合规划器 —— ChatHTN 用于可证明可靠的计划，AlphaEvolve 用于带有机器可检验 evaluator (评估器) 的代码搜索 —— 并为问题选择正确的一种。
version: 1.0.0
phase: 14
lesson: 11
tags: [planning, htn, chathtn, alphaevolve, evolutionary-search]
---

给定一个问题类别（policy-bound workflow (策略约束工作流) vs 代码优化 vs 开放式任务），选择一种规划器并生成正确的脚手架。

决策：

1. 问题是否有硬 preconditions (前置条件) / 策略 / 调度约束？ -> HTN (ChatHTN)。
2. 问题是否有确定性的、机器可检验的 fitness function (适应度函数)？ -> Evolutionary (AlphaEvolve)。
3. 两者都不是？ -> 先尝试 ReAct (Lesson 01) 或 ReWOO (Lesson 02)。

对于 HTN，生成：

1. `Operator` 类型，带有 `preconditions`、`effects_add`、`effects_remove`。
2. `Method` 类型，带有 `task`、`preconditions`、`subtasks`。
3. 一个 planner (规划器)，先尝试 methods，回退到 LLM decomposition (LLM 分解)，并缓存成功的 LLM 分解。
4. 一个验证步骤，拒绝引用未知 operators 或 methods 的 LLM 分解。

对于 Evolutionary，生成：

1. 候选程序的 seed population (种子种群)。
2. 返回标量 fitness (适应度) 的确定性 evaluator (评估器)。
3. 一个 mutation operator (变异操作符)（LLM 驱动或基于规则）。
4. 一个 selection loop (选择循环)（保留 top-k，变异，重复）并带 early stopping (早停)。

硬性拒绝：

- ChatHTN 中 LLM 输出未经 operator-schema 验证就直接应用。soundness (可靠性) 声明失效。
- AlphaEvolve 中 evaluator 调用 LLM judge。Fitness 必须是确定性的；LLM judge 引入的随机噪声是循环无法恢复的。
- 将任一模式用于开放式任务（"写一篇博客文章"）。没有 evaluator，没有 preconditions -> 使用 ReAct。

拒绝规则：

- 如果领域没有清晰的 operator schema，拒绝 ChatHTN。建议 ReWOO 或普通 ReAct。
- 如果领域没有机器可检验的 fitness，拒绝 AlphaEvolve。建议 Self-Refine (Lesson 05)。
- 如果用户想要 "planner + LLM 做最终决定"，拒绝。Symbolic correctness (符号正确性) 与 LLM exploration (LLM 探索) 之间的分工是承重结构。

输出：`operators.py`、`methods.py`、`planner.py`（HTN）或 `evaluator.py`、`mutator.py`、`loop.py`（evolutionary），以及带有决策理由的 `README.md`。以 "what to read next" 结尾，指向 Lesson 25（如果 debate-style verification (辩论式验证) 适合该问题），或 Lesson 02（如果该任务实际上仍是 ReWOO 形状）。
