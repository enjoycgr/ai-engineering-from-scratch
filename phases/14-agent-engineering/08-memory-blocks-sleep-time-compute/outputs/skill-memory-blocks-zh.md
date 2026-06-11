---
name: memory-blocks
description: 生成 Letta 风格的三层记忆系统（core 块、recall、archival），带一个位于关键路径之外的 sleep-time consolidation agent。
version: 1.0.0
phase: 14
lesson: 08
tags: [memory, letta, blocks, sleep-time, consolidation]
---

给定目标 runtime、主模型和一个（可能更强的）sleep-time 模型，产出带显式块类型的三层记忆系统和异步合并。

产出：

1. `Block` 类型，带 `label`、`value`、`limit`、`description`、`version`、`history`。每次写入都会 bump version 并记录旧值。暴露 `near_limit(threshold=0.8)`。
2. `BlockStore`，至少包含三个默认块：`human`（关于用户的事实）、`persona`（智能体自我概念）和 `task`（当前范围）。允许用户自定义块。
3. `Recall` 存储 —— 按会话分页的轮次日志。每轮自动写入。尾部在超出上限时驱逐，但仍可检索。
4. `Archival` 存储 —— 至少两个后端（vector、KV）。Insert 返回记录 id。矛盾时 invalidate 而非 delete。
5. `PrimaryAgent`，处理轮次，仅发出原始写入。Critical path 上不做摘要或合并。
6. `SleepTimeAgent`，在轮次间运行：摘要超出阈值的块、失效矛盾的归档记录、将 `learned_context` 写入共享块。

硬性拒绝：

- 任何在用户面对轮次中同步运行的记忆操作，除了直接查找。Summary、consolidation、invalidation 属于 sleep-time pass。
- 矛盾时删除归档记录。Invalidate 以保留可审计的历史。
- 未经审查就写入 Persona 或 Safety 块。这些块全局塑造行为；静默写入掩盖 bug。

拒绝规则：

- 如果 runtime 无法在会话间持久化块，拒绝交付一个被描述为"有记忆"的产品。降级声明。
- 如果 sleep-time agent 没有 trace 输出，拒绝。静默 consolidation 是调试死区。
- 如果用户要求"不做 invalidation，始终信任最新写入"，拒绝任何历史声明重要的领域（合规、医疗、法律）。

输出：每个组件一个文件，外加 `README.md`，说明默认块、sleep-time 节奏和矛盾解决策略。结尾附带"接下来读什么"的指引：如果智能体需要基于图的记忆推理，指向第 09 课；如果产品需要在记忆操作上打 OTel span，指向第 23 课。
