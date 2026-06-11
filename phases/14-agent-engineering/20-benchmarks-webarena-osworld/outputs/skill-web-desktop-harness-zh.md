---
name: web-desktop-harness
description: 构建 WebArena/OSWorld-style harness (类 WebArena/OSWorld 工具链)，带有 execution-based evaluation (基于执行的评估) 和 trajectory-efficiency metrics (轨迹效率指标)。
version: 1.0.0
phase: 14
lesson: 20
tags: [webarena, osworld, harness, trajectory-efficiency]
---

给定一个目标应用（web (网页) 或 desktop (桌面)）和一组带 gold trajectories (黄金轨迹) 的任务，构建一个 eval harness (评估工具链)。

产出：

1. 任务定义：`(tid, description, gold_steps, success_predicate, state_reset)`。
2. Runner (运行器)：运行 agent (智能体)，捕获每个 action (动作)，记录 step count (步骤数) + elapsed time (耗时) + success state (成功状态)。
3. Trajectory-efficiency metric (轨迹效率指标)：`agent_steps / gold_steps`。按任务和聚合报告。
4. 任务之间的 state reset (状态重置)——绝不在被另一个任务污染的状态上运行任务。
5. Failure-mode classifier (失效模式分类器)：对于每个 failure (失败)，标记它是 grounding miss ( grounding 失误)（错误元素）还是 planning miss (规划失误)（错误动作）。

Hard rejects：

- 任务之间没有 state reset (状态重置)。Cross-task contamination (跨任务污染) 使所有分数无效。
- 只报告 success rate (成功率)。Trajectory efficiency (轨迹效率) 是 2026 年的标准。
- 没有 DOM parity (DOM 对等性) 的仅 screenshot (截图) harness (工具链)。有些 agent (智能体) 使用 DOM+vision (视觉)；除非专门约束 surface (界面)，否则同时提供两者。

Refusal rules：

- 如果任务没有 gold trajectories (黄金轨迹)，拒绝。没有它们就无法测量 efficiency (效率)。
- 如果应用没有 pin (固定) 到特定版本，拒绝。Drift (漂移) 使跨运行比较无效。
- 如果 agent (智能体) 有 destructive tools (破坏性工具)（delete (删除)、publish (发布)），要求应用的 sandbox copy (沙箱副本)。

输出：`tasks.py`、`runner.py`、`failure_classifier.py`、`report.py`、`README.md`，解释 reset policy (重置策略)、gold-trajectory sourcing (黄金轨迹来源) 和 grounding-vs-planning split ( grounding 与规划的分割)。结尾附 "what to read next"，指向 Lesson 21（computer use models (计算机使用模型)）或 Lesson 30（eval-driven development (评估驱动开发)）。
