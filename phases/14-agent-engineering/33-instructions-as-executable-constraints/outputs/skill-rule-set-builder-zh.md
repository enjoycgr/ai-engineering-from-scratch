---
name: rule-set-builder
description: 采访项目所有者，将现有 prose instructions（散文式指令）分类到五个操作类别中，并发出一份 versioned agent-rules.md 和一个 Python checker stub（存根）。
version: 1.0.0
phase: 14
lesson: 33
tags: [rules, instructions, constraints, checker, workbench]
---

给定一个仓库和任何现有 prose instructions（`AGENTS.md`、`CONTRIBUTING.md`、onboarding docs），产生一个 workbench 可以执行的五类别 rule set。

五个类别：

1. `startup` — 工作开始前必须为真的条件。
2. `forbidden` — 绝对不能发生的事。
3. `definition_of_done` — 证明任务完成的条件。
4. `uncertainty` — 不确定时 agent 做什么。
5. `approval` — 什么需要人类签字。

产出：

1. `docs/agent-rules.md`，每条规则一个 `##` heading。每条规则携带 `category`、`check` 和一行 description。
2. `tools/rule_checker.py`，`RuleChecker` 类暴露每个 `check` 对应的一个方法。每个方法接受一个 `TurnTrace` dataclass 并返回 `bool`。
3. `tools/rule_report.py` runner，加载 rules，在 trace 上运行 checker，发出 `rule_report.json`。
4. 一份 migration notes 文件：哪些 prose lines 变成了哪条 rule，哪些作为 aspirational 被丢弃，原因是什么。

Hard rejects（硬性拒绝）：

- 没有 `check` 字段的 rules。Aspirational-only rules 属于 onboarding docs，不属于 workbench rule set。
- 单一的 "be careful" rule。指定 category 和 check，或删除它。
- 需要 LLM calls 的 checks。Rule checks 必须是确定性的且便宜的，以便每轮都能运行。
- 超过 200 行的 rule files。按 category 拆分为 `agent-rules.{startup,forbidden,done,uncertainty,approval}.md`，并从父 index 路由。

Refusal rules（拒绝规则）：

- 如果 agent 产品无法提供 `TurnTrace`（没有 instrumentation），拒绝接入 checker，直到至少记录了 `read_state_file`、`edited_files` 和 `tests_exit_code`。
- 如果现有 instructions 大部分是 aspirational（>50%），在发出 rules 前 surface 该发现。Rule set 看起来会很薄；这是正确的。
- 如果一条规则是因为单个 past incident 添加的，附上 incident id，以便 future review 决定它是否仍然需要。

输出结构：

```
<repo>/
├── docs/
│   └── agent-rules.md
├── tools/
│   ├── rule_checker.py
│   └── rule_report.py
└── docs/migration-notes.md
```

结尾附上 "what to read next"，指向：

- Lesson 36 了解扩展 forbidden 类别的 per-task scope contracts。
- Lesson 38 了解消费 rule report 的 verification gates。
- Lesson 39 了解评分 rule compliance 的 reviewer agent。
