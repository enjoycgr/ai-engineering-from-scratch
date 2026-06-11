---
name: search-policy
description: 根据任务形状、token 预算和评估器质量选择搜索策略（ReAct、ToT、LATS、进化式）。
version: 1.0.0
phase: 14
lesson: 04
tags: [tree-of-thoughts, lats, mcts, search, value-function]
---

给定任务形状（单答案 / 多答案 / 开放式）、token 预算和可用评估器（scalar test / heuristic / self-eval），产出带具体参数的搜索策略建议。

产出内容：

1. Decision（决策）。以下之一：linear ReAct（线性 ReAct）、beam ToT（带 beam width k）、BFS ToT（带最大深度）、带剪枝的 DFS ToT、MCTS LATS（带迭代次数和 UCT c）、evolutionary search（仅当评估器是程序化且可检查时）。
2. Parameters（参数）。对于每种策略，具体的数值默认值：beam width、depth cap、branching factor K、每层级 rollout 数、UCT c（默认 1.4）、timeout。
3. Value function（价值函数）。精确指定什么给节点打分。选项：单元测试通过率、到目标的数值距离、带格式（sure/likely/impossible 或 1..10 或 vote）的 prompted LLM score，或环境奖励。
4. Token budget estimate（Token 预算估算）。最坏情况 token = branching_factor ^ depth * avg_prompt_tokens。显示该数字。如果超过用户预算，推荐更便宜的策略。
5. Failure modes（失败模式）。对于每种选定策略，列出前两种失败模式及其缓解措施（例如 LATS + noisy evaluator -> 按 CRITIC 添加 tool-grounded verification，Lesson 05）。

Hard rejects（硬拒绝）：

- 当评估器不可靠时推荐搜索（仅 self-eval，无 ground truth）。退回到 ReAct + CRITIC。
- 在没有充分理由时将 branching factor K 设高于 5。K=3-5 是论文默认值；K=10 爆炸式增加成本。
- 将 LATS 应用于聊天式任务。搜索对没有程序化目标的对话问答没有帮助。
- 没有机器可检查的 fitness 时进行进化式搜索。只有当 fitness 是程序化的（运行测试、测量速度、验证定理）时 AlphaEvolve 才有意义。

Refusal rules（拒绝规则）：

- 如果 token 预算 < 单次轨迹成本的 5 倍，拒绝搜索并推荐 ReAct + Reflexion（Lesson 03）。
- 如果 wall-clock 延迟预算 < 10 秒，拒绝 LATS 并推荐 ReAct。
- 如果任务是纯信息检索，拒绝搜索并推荐 ReWOO（Lesson 02）。

输出：一个推荐块（选定策略、参数、价值函数、预算估算）加一个 "what to read next" 注释：指向 Lesson 05（CRITIC）以了解评估器可靠性、Lesson 11（AlphaEvolve）以了解进化式变体，或 Lesson 30（eval-driven development）以了解基准级验证。
