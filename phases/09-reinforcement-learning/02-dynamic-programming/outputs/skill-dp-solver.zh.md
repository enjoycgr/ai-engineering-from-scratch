---
name: dp-solver
description: 通过策略迭代或价值迭代精确求解小型表格 MDP，并报告收敛行为。
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

给定一个已知模型的 MDP，输出：

1. 选择 (Choice)。策略迭代 vs 价值迭代。理由与 |S|、|A|、γ 相关。
2. 初始化 (Initialization)。V_0、起始策略。收敛敏感性。
3. 停止条件 (Stopping)。Sup-范数容差 ε。预期扫描次数。
4. 验证 (Verification)。精确计算 V*(s_0)。提取贪婪策略。
5. 用途 (Use)。此基线将如何用于调试/评估基于采样的方法。

拒绝在状态空间 > 10⁷ 上运行 DP。没有 sup-范数检查则拒绝声称收敛。在无限视界任务上标记任何 γ ≥ 1 为保证违反。
