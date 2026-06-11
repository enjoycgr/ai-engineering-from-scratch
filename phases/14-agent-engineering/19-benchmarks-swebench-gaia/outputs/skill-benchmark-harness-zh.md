---
name: benchmark-harness
description: 为代码库构建 SWE-bench-style harness (类 SWE-bench 工具链)，带有 FAIL_TO_PASS / PASS_TO_PASS gating (门禁)、contamination checks (污染检查) 和 step-count metrics (步骤计数指标)。
version: 1.0.0
phase: 14
lesson: 19
tags: [swe-bench, gaia, agentbench, harness, evaluation]
---

给定一个代码库和一组 (bug, fix) 对，构建一个以真实 unit tests (单元测试) 为门禁并记录运维指标的 benchmark harness (基准测试工具链)。

产出：

1. 每个任务定义：`(tid, description, state_before, fail_to_pass_tests, pass_to_pass_tests, solution)`。
2. 一个 runner (运行器)，应用 agent (智能体) 的 patch (补丁)，在 sandbox (沙箱) 中运行仓库的 test suite (测试套件)，并记录：FTP pass count (FTP 通过数)、PTP pass count (PTP 通过数)、step count (步骤数)、tokens (令牌数)、wall-clock (挂钟时间)、cost (成本)。
3. 一个 contamination check (污染检查)：将 issue text (问题文本) 与产生的 patch (补丁) 进行模式匹配；标记 >=30% 重叠。
4. 一个 reporter (报告器)，以 JSON 形式发射 per-task (每个任务) 和 aggregate (聚合) 分数，外加 P50/P75/P95 step 和 cost。
5. 一个 CI job，每次 PR 上运行 harness (工具链) 并在 >=5% regression (回归) 时失败。

Hard rejects：

- 只报告单一 aggregate number (聚合数字) 的 harness (工具链)。要求 per-task results (每个任务的结果) + distributions (分布)。
- 不在 sandbox (沙箱) 中运行 tests (测试) 的 harness (工具链)。Agent-provided patches (智能体提供的补丁) 是不受信任的代码。
- 没有 PASS_TO_PASS gate (门禁) 的 harness (工具链)。破坏其他 tests (测试) 的 patch (补丁) 是比错过修复更糟糕的 regression (回归)。

Refusal rules：

- 如果用户只要求"FAIL_TO_PASS 分数"，拒绝。添加 PASS_TO_PASS；破坏现有 tests (测试) 是比错过修复更糟糕的 regression (回归)。
- 如果 tests (测试) 没有 pin (固定) 到特定 commit (提交)，拒绝。Tests (测试) 的 drift (漂移) 使跨运行的分数 incomparable (不可比)。
- 如果 tasks (任务) 与训练期间见过的 issue text (问题文本) 重叠，明确标记它。

输出：`tasks.py`、`harness.py`、`contamination.py`、`report.py`、`README.md`，解释 sandbox (沙箱)、gates (门禁)、contamination policy (污染策略)。结尾附 "what to read next"，指向 Lesson 30（在 harness (工具链) 之上进行 eval-driven development (评估驱动开发)）。
