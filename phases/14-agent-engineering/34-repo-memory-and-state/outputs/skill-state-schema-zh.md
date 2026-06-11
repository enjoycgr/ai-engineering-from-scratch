---
name: state-schema
description: 生成项目特定的 JSON Schemas（用于 agent state 和 task board）、一个带 atomic writes（原子写入）的 Python StateManager，以及一个 migration scaffold（迁移脚手架），使 schema bumps 不会破坏 workbench。
version: 1.0.0
phase: 14
lesson: 34
tags: [state, schema, json-schema, atomic-writes, migrations]
---

给定一个仓库和其中运行的 agent 产品，为 workbench 生成 schema-first state files。

产出：

1. `schemas/agent_state.schema.json`，涵盖 required keys、允许的 status 值、array-vs-null 纪律和 `schema_version` 整数。
2. `schemas/task_board.schema.json`，涵盖 task id pattern、允许的 owners、允许的 statuses 和 acceptance arrays。
3. `tools/state_manager.py`，暴露 `load`、`commit` 和 `update`，带有 temp-and-rename atomic writes（原子写入）。
4. `tools/migrate_state.py` scaffold，用于下一个 schema bump，如果文件来自未知版本则 fail-loud（大声失败）。
5. `agent_state.json` 和 `task_board.json`，以 `schema_version: 1` 和 fresh backlog 为种子。

Hard rejects（硬性拒绝）：

- 没有 `schema_version` 字段的 schema。Migrations 不是可选的。
- 在期望 array 的位置允许 `null`。`null` 是伪装成数据的 write-time bug。
- 使用 plain `open(path, "w")` 的 writer。只允许 atomic writes；partial files 会 corrupt source of truth。
- 在 state 中存储 tokens、raw chat transcripts 或 PII。State 是用于 repo-relevant facts 的。

Refusal rules（拒绝规则）：

- 如果仓库没有版本控制，拒绝 ship state files。Atomic writes 加 git diff 是耐久性故事。
- 如果项目没有至少一个 acceptance command 来验证 `done` transition，拒绝 `status: done` enum 值。没有 acceptance check 的 `done` 是 theater（表演）。
- 如果项目打算在没有锁策略的情况下跨进程共享 state，在 ship 之前 surface 该发现；atomic rename 必要但不充分。

输出结构：

```
<repo>/
├── agent_state.json
├── task_board.json
├── schemas/
│   ├── agent_state.schema.json
│   └── task_board.schema.json
└── tools/
    ├── state_manager.py
    └── migrate_state.py
```

结尾附上 "what to read next"，指向：

- Lesson 35 了解在 startup 时调用 manager 的 initialization script。
- Lesson 38 了解读取 state 以评分完成度的 verification gate。
- Lesson 40 了解消费同一 schema 的 handoff generator。
