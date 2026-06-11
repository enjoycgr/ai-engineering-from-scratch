---
name: scope-contract
description: 为每个任务生成包含允许/禁止通配符（allowed/forbidden globs）、验收标准和回滚计划的范围契约，以及一个 CI 就绪的、通配符感知的检查器，在每次智能体（agent）差异（diff）时运行。
version: 1.0.0
phase: 14
lesson: 36
tags: [scope, contract, globs, diff-check, ci]
---

给定任务描述和仓库结构，生成一个范围契约（scope contract）和一个差异感知检查器（diff-aware checker）。

产出：

1. 任务的 `scope_contract.json`，包含字段：`task_id`、`goal`、`allowed_files`（通配符 globs）、`forbidden_files`（通配符 globs）、`acceptance_criteria`、`rollback_plan`、`approvals_required`。
2. `tools/scope_check.py`，接收契约路径和触碰文件列表，返回 `ScopeReport` 并在任何违规时以非零状态退出。
3. CI 步骤（`.github/workflows/scope-check.yml` 或等效物），针对合并差异（merge diff）运行检查器。
4. `outputs/scope/closed/<task_id>.json` 归档约定，以便契约随变更历史一起交付。

硬性拒绝：

- 没有 `forbidden_files` 的契约。负面空间（negative space）是契约的一部分。
- 对代码目录列出原始路径而非通配符（globs）的契约。重构会使原始路径在一夜之间失效。
- `rollback_plan` 字段为空或写"see runbook"。必须明确写出。
- 将审批列为"case by case"。审批边界（Approval boundaries）必须是可枚举的。

拒绝规则：

- 如果任务描述没有限定仓库的某个区域，拒绝仅凭描述编写 `allowed_files`。请提供任务所在的目录。
- 如果仓库没有测试命令，拒绝添加 `acceptance_criteria`，直到提供或存根了一个。无法验证的契约只是一个愿望。
- 如果智能体运行时无法兑现审批边界（没有人工介入回路 human-in-the-loop），在交付前暴露这个缺口；对需要审批的行为发生范围蔓延将是最主要的失败。

输出结构：

```
<repo>/
├── scope_contract.json
├── outputs/scope/closed/
│   └── T-XXX.json
├── tools/
│   └── scope_check.py
└── .github/
    └── workflows/
        └── scope-check.yml
```

以"what to read next"结尾，指向：

- Lesson 37，了解将运行命令链接回契约的运行时反馈（runtime feedback）。
- Lesson 38，了解消费范围报告的验证门（verification gate）。
- Lesson 39，了解审查关闭契约归档的审阅者智能体（reviewer agent）。
