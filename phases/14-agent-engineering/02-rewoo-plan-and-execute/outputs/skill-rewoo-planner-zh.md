---
name: rewoo-planner
description: 根据用户请求和工具目录生成经过验证的 ReWOO plan DAG（计划有向无环图）。
version: 1.0.0
phase: 14
lesson: 02
tags: [rewoo, plan-and-execute, planning, dag, distillation]
---

给定用户请求和工具目录（名称、输入 schema、描述），生成一个 ReWOO 计划：一个带有工具调用和证据引用（`#E1`、`#E2`...）的 DAG 步骤。在移交给执行器之前验证计划。

产出内容：

1. 一个 plan DAG（计划 DAG）。每个节点有 id（`E1`、`E2`...）、工具名称、参数字符串（可能包含 `#E<k>` 引用）和可选的 `parallel_group` 标签。
2. 验证输出。通过拓扑排序检查无环性；引用解析检查（每个 `#E<k>` 都有前面的生产者）；工具存在检查（每个工具名称都在目录中）；参数 schema 检查（每个参数匹配工具的输入 schema）。
3. 并行性提示。对于每个拓扑层级，列出可以并发执行的节点。
4. Planner/solver 拆分建议。如果计划少于 3 步，建议改用 ReAct。如果计划有无界循环需求（每一步都 replanning），建议带 replanner 的 Plan-and-Execute。如果计划超过 30 步或针对网页/移动端，建议带合成计划数据的 Plan-and-Act。

Hard rejects（硬拒绝）：

- 有环的计划。ReWOO 假设 DAG；有环是 ReAct 或 LATS 的问题。
- 引用拓扑序中尚不存在的 `#E<k>` 的计划。发出失败的特定边。
- 调用不在目录中的工具的计划。不要发明工具来凑数。
- 引用的参数类型与工具 schema 不匹配的计划（例如 `#E1` 替换为字符串但工具期望 int）。

Refusal rules（拒绝规则）：

- 如果任务是开放式探索（未知工具、未知步骤），拒绝并推荐 ReAct 或 LATS（Lesson 04）。
- 如果工具目录包含没有门控审批工具的破坏性工具，拒绝并指向 Lesson 09（权限、沙箱）。

输出：结构化计划（JSON 或 YAML）、验证报告、并行性映射，以及指向执行器（ReWOO Worker）、replaner（Plan-and-Execute）或更大轨迹采样循环（Plan-and-Act）的后续操作。

结尾附 "what to read next" 注释：如果任务类别之前尝试过，指向 Lesson 03（Reflexion）；如果计划会受益于搜索，指向 Lesson 04（LATS）。
