---
name: hybrid-memory
description: 生成 Mem0 风格的三存储记忆系统（向量 + KV + 图），带融合打分器、作用域分类和时序失效机制。
version: 1.0.0
phase: 14
lesson: 09
tags: [memory, mem0, vector, graph, kv, fusion, scope]
---

给定目标 runtime、向量后端（Qdrant、pgvector、Chroma、sqlite-vec）、KV 后端（Postgres、Redis、dict）和图后端（Neo4j、内存边），产出融合记忆系统。

产出：

1. 三个 store class 封装在 `add(text, user_id, session_id, scope, importance, tags)` facade 之后。写入时 extractor 将 `text` 分解为记录、KV 三元组和图三元组。没有 store 是可选的。
2. 融合打分器 `score = w_rel * relevance + w_imp * importance + w_rec * recency`。三个权重都暴露为配置。按产品调优，而非按调用调优。
3. 作用域分类 (Scope taxonomy)：`user`、`session`、`agent`。检索**必须**尊重 scope。用户查询绝不应泄漏另一用户的记录。
4. 时序失效 (Temporal invalidation)。矛盾时标记旧边/记录失效；永不删除。暴露 `search(query, as_of=timestamp)` 供历史查询。
5. Extractor 接口。默认可以是 LLM 驱动；允许确定性 regex fallback 用于测试。限制每次 `add()` 的图边数以防止爆炸。

硬性拒绝：

- 将单存储记忆描述为"Mem0 风格"。仅向量、仅 KV、仅图的产品没问题，但不是混合记忆。不要误命名。
- 不带 per-scope 权重或显式 `scope=` 过滤器的跨 scope 检索。Scope leak (作用域泄漏) 是合规和隐私事故。
- 矛盾时删除。Invalidate 并打时间戳。删除掩盖 bug 并破坏审计。

拒绝规则：

- 如果用户要求"不要 importance weighting"，拒绝。在百万记录上的扁平 relevance ranking 是检索失败的定时炸弹。
- 如果图后端没有 conflict detector，拒绝将结果系统称为"Mem0 风格"。降级名称。
- 如果产品涉及 PII（医疗、法律、HR），拒绝交付 extractor 未经产品负责人审计的系统。

输出：每个存储一个文件，外加 `memory.py`（facade）、`config.py`（权重）、`README.md` 解释融合权重、作用域策略、extractor 契约和失效语义。结尾附带"接下来读什么"的指引：如果智能体需要学习新技能，指向第 10 课；如果记忆操作需要 OTel span，指向第 23 课；如果检索需要处理不可信输入，指向第 27 课。
