---
name: eval-suite
description: 构建三层评估套件（静态基准、自定义离线、在线生产），包含 evaluator-optimizer 循环和 CI 门。
version: 1.0.0
phase: 14
lesson: 30
tags: [evaluation, ci, regression, benchmarks, llm-judge]
---

给定一个智能体产品，构建接入 CI 的三层评估套件。

产出：

1. **Static benchmark layer（静态基准层）**——至少一个相关基准（SWE-bench Verified 用于代码、BFCL V4 用于工具使用、WebArena 用于网页、OSWorld 用于桌面、GAIA 用于通才）。始终同时报告 +-审计分数。
2. **Custom offline layer（自定义离线层）**——至少一个 LLM-judge 评分标准，按领域特定维度评分（事实、语气、范围、拒绝质量）。至少一个基于执行的用例，探测智能体运行后的实际状态。至少一个基于轨迹的用例，带黄金路径。
3. **Online eval layer（在线评估层）**——会话回放、护栏触发的告警、通过 OTel GenAI spans（Lesson 23）的每步成本/延迟追踪。
4. **Evaluator-optimizer runner（评估器-优化器运行器）**——将智能体包装在提出 / 评判 / 精炼中，带轮次上限。
5. **CI gate（CI 门）**——>=5% 回归 vs 基线时失败构建。随时间追踪基线。
6. **Case mapping（用例映射）**——Phase 14 课程的每个护栏和每个学到的规则至少有一个用例。

硬性拒绝：

- 没有基线的评估套件。没有参考就无法检测回归。
- 事实任务上没有外部依据的 LLM-judge。那里需要 CRITIC 模式（Lesson 05）。
- 没有固定种子或快照状态的不稳定用例。误报侵蚀团队对评估的信任。

拒绝规则：

- 如果用户想要"只有快乐路径"，拒绝。每种故障模式（Lesson 26）都应该有一个用例。
- 如果用户想要"没有 CI 门"，拒绝面向付费用户的产品。否则评估漂移是 invisible 的。
- 如果用户想要"全是 LLM-judge"，拒绝事实和合规任务。那里需要基于执行或编程的评估器。

输出：`cases/benchmarks/`、`cases/custom/`、`cases/online/`、`runner.py`、`ci_gate.py`、`README.md` 解释评分标准、基线和 Phase 14 映射表。结尾附上"接下来读什么"，指向 Lesson 24（可观测性）、Lesson 26（故障模式）或 Lesson 23（OTel）了解底层。
