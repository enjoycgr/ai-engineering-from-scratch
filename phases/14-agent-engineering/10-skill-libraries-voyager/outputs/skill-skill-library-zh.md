---
name: skill-library
description: 生成 Voyager 风格的技能库，支持注册、基于相似度的检索、组合式执行和失败驱动的精炼。
version: 1.0.0
phase: 14
lesson: 10
tags: [voyager, skills, library, composition, refinement]
---

给定目标 runtime 和一个领域，产出支持 Voyager 三个组件的技能库：curriculum hook、可检索 skill store、迭代精炼。

产出：

1. `Skill` 类型，带 `name`、`description`、`code`、`version`、`tags`、`depends_on`、`history`。每次写入都记录先前代码。
2. `SkillLibrary`，带 `register(skill, dedup=True)`（新建或 bump version）、`search(query, top_k, tag_filter)`、`get(name)`、`topo_order(name)`（依赖解析）、`execute(name, context)`（拓扑运行）。
3. 检索**必须**使用 embedding 相似度或 BM25，而非用 LLM 对完整 library 打分。允许对 top-k 短名单做 LLM re-rank。
4. 执行**必须**按 skill 捕获异常，并将它们 surface 到 trace 中，作为 refinement loop 可消费的反馈。
5. 精炼 hook：失败的 `execute` 之后，runtime 收集 (task, skill_name, error, env_state)，传给模型，并在重写后的 skill 上调用 `register`。Version bump；history 保留旧代码。

硬性拒绝：

- 技能库中的 skills 是散文字符串而非代码。Skills 是可执行的。散文属于 `description`。
- 组合时没有拓扑排序。无 cycle detection 的深度优先在 skill DAG 上会断裂。
- 静默版本覆盖。每次精炼**必须** bump `version` 并将旧代码推入 `history` 以供审计。

拒绝规则：

- 如果目标 runtime 没有 skill 执行的 sandbox，拒绝交付 skills 会触及生产系统的领域。先要求 sandbox（第 09 课原则）。
- 如果用户要求"每次失败都自动重试但不精炼"，拒绝。没有精炼的重试会放大 bug，而非修复它。
- 如果 library 超过约 200 个 skills 且仍是扁平检索，拒绝称其为"生产就绪"。先添加 tag filters 和分层命名空间。

输出：`skill.py`、`library.py`、`execute.py`、`refine.py`，以及 `README.md`，解释 dedup 规则、检索后端、精炼 prompt 和版本策略。结尾附带"接下来读什么"的指引：指向第 17 课了解 Claude Agent SDK 集成，第 16 课了解 OpenAI Agents SDK tool 翻译，或第 30 课了解技能库质量评估。
