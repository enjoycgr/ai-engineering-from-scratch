---
name: workbench-audit
description: 在 agent 开始工作前，审计仓库的七个 agent workbench surface（工作台表面），报告哪些缺失、部分存在或健康。
version: 1.0.0
phase: 14
lesson: 31
tags: [workbench, audit, reliability, agent-engineering]
---

给定一个仓库路径和将在其中运行的 agent 产品，审计七个 workbench surface（工作台表面）并生成一份 readiness report（就绪报告）。

七个 surface：

1. Instructions（指令）：agent 首先读取的根文件（如 `AGENTS.md`），简短，并路由到更深层的规则。
2. State（状态）：一个 durable（持久的）、machine-readable（机器可读的）文件，记录任务、已触碰文件、阻塞项、下一步行动。
3. Scope（范围）：每个任务的一份合约，列出允许文件、禁止文件、acceptance criteria（验收标准）、rollback plan（回滚计划）。
4. Feedback（反馈）：一个 runner，捕获 command、stdout、stderr、exit code，并将结果 feed 回循环。
5. Verification（验证）：一个 gate，运行 tests、lint、type-check、smoke run，并确认 acceptance criteria。
6. Review（审查）：以不同角色的第二次审查，builder 不能批改自己的作业。
7. Handoff（交接）：一份 artifact，总结改变了什么、为什么、还有什么待办，以及 next best action（下一步最佳行动）。

产出：

- 每个 surface 的评分：0 缺失，1 部分存在，2 健康。将每个评分与你观察到的文件或流程关联。
- 按 leverage（杠杆效应）排序的三个优先级：如果首先添加哪个缺失的 surface，能消除最多的 failure modes。
- 一份 `workbench_audit.json` machine-readable（机器可读）报告，外加一份 `workbench_audit.md` human-readable（人类可读）摘要。
- 对最弱 surface 的一份 starter patch：将评分从 0 提升到 1 的最小文件变更。

Hard rejects（硬性拒绝）：

- 没有文件路径或流程引用的 "Healthy" 评分。没有证据的审计会腐烂。
- 单个合并的 "agent config" surface。合并 surface 会隐藏任务中断时哪个 surface 失败了。
- 因为 tests 慢而跳过 verification。如果 verification 不在 workbench 上，builder 就会自己批改作业。

Refusal rules（拒绝规则）：

- 如果仓库完全没有 test command，拒绝 verification 评分，并将其作为 blocking finding（阻塞性发现）提出。
- 如果仓库没有版本控制历史，拒绝 handoff 评分，并将其作为阻塞性发现提出。
- 如果 agent 产品以 root 运行或拥有无限制的文件访问权限，在定义 sandbox 或写入列表之前拒绝 scope 评分。

输出结构：

```
workbench-audit/
├── workbench_audit.json
├── workbench_audit.md
├── patches/
│   └── <weakest-surface>.patch
└── README.md
```

结尾附上 "what to read next"，指向：

- Lesson 32 了解 minimal repo layout。
- Lesson 33 深入了解 instructions surface。
- Lesson 38 了解 verification gate。
