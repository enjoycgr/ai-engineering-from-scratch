---
name: minimal-workbench
description: 为任何仓库搭建最小可用的 agent workbench（工作台）—— 简短的 AGENTS.md router（路由器）、持久的 agent_state.json 和一个以项目当前 backlog 为键的 JSON task_board.json（任务板）。
version: 1.0.0
phase: 14
lesson: 32
tags: [workbench, agents-md, state, task-board, scaffold]
---

给定一个仓库路径和一个简短的 backlog，搭建最小可用的 agent workbench。

产出：

1. `AGENTS.md` 不超过 80 行。它必须路由到：state file、task board、deeper rules doc（即使为空）、和 verification command。此文件中不要放 prose tutorials。
2. `agent_state.json` 带有这些 keys：`active_task_id`、`touched_files`、`assumptions`、`blockers`、`next_action`。所有可选字段默认为空数组或空字符串，数组字段绝不使用 `null`。
3. `task_board.json` 作为 JSON task 数组。每个任务有 `id`、`goal`、`owner`（`builder` | `reviewer` | `human`）、`acceptance`（字符串列表）和 `status`（`todo` | `in_progress` | `done` | `blocked`）。
4. `docs/agent-rules.md` placeholder（占位符），每个 surface 一个 H2，以便后续课程填充。

Hard rejects（硬性拒绝）：

- `AGENTS.md` 超过 80 行或少于 10 行。太长 agent 会跳过；太短则无法承载路由信息。
- 引用 chat history 而非仓库的 state file。仓库是 system of record。
- 没有 `acceptance` 的 task board。没有 acceptance criteria 的任务会变成 "looks good" 的橡皮图章。
- `owner` 为 `agent` 或 `model` 的任务。Owners 是 roles（角色），不是 entities（实体）。

Refusal rules（拒绝规则）：

- 如果仓库完全没有 verification command，拒绝写入 `AGENTS.md`，直到提供一个或 stub 一个。指向缺失 gate 的 router 比没有 router 还糟。
- 如果 backlog 有超过 12 个 open tasks，拒绝并要求用户拆分。超过一屏的 board 会滑入 planning theater（规划表演）。
- 如果项目在 tracked files 中带有 secrets，拒绝写入 state file，并首先将 secret leak 作为 blocking finding（阻塞性发现）提出。

输出结构：

```
<repo>/
├── AGENTS.md
├── agent_state.json
├── task_board.json
└── docs/
    └── agent-rules.md
```

结尾附上 "what to read next"，指向：

- Lesson 33 了解如何将 rules placeholder 转化为 executable constraints（可执行约束）。
- Lesson 34 了解 durable state schema。
- Lesson 36 了解每个任务的 scope contract（范围合约）。
