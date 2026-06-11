# 混合记忆：向量 + 图 + KV (Mem0)

> Mem0 (Chhikara 等, 2025) 将记忆视为三种并行存储 —— 向量 (vector) 用于语义相似度，KV 用于快速事实查找，图 (graph) 用于实体关系推理。一个 scoring layer (打分层) 在检索时将三者融合。这是 2026 年外部记忆的生产标准。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 分钟

## 学习目标

- 解释为什么单一存储（仅向量、仅图、仅 KV）对智能体记忆是不够的。
- 说出 Mem0 的三个并行存储以及各自优化的查询类型。
- 描述 Mem0 的 fusion scoring (融合打分) —— relevance (相关性)、importance (重要性)、recency (时效性) —— 以及为什么它是加权和而非层级结构。
- 用标准库实现一个玩具级三存储记忆系统：`add()` 同时写入三个存储，`search()` 融合结果。

## 问题背景

一种存储对三类查询中的两类都是错的：

- **Semantic similarity (语义相似度)** —— "上周我们讨论过 agent drift 的什么？" 向量 (vector) 获胜；KV 和图都会漏掉。
- **Fact lookup (事实查找)** —— "用户的手机号是多少？" KV 获胜；向量浪费，图杀鸡用牛刀。
- **Relationship reasoning (关系推理)** —— "哪些客户共享同一个 billing entity？" 图 (graph) 获胜；向量和 KV 无法回答。

生产级智能体在一次会话中发出全部三类查询。单存储记忆对其中两类永远是错的。Mem0 的贡献是将三者封装在一个 `add`/`search` 表面背后，用一个 scoring function (打分函数) 融合它们。

## 核心概念

### 三种并行存储

Mem0 (arXiv:2504.19413, 2025 年 4 月) 在 `add(text, user_id, metadata)` 时：

1. 从文本提取候选事实（LLM 驱动的步骤）。
2. 将每个事实写入 vector store (向量存储)（embedding）供语义搜索。
3. 将每个事实写入 KV store，键为 (user_id, fact_type, entity)，实现 O(1) 查找。
4. 将每个事实写入 graph store (图存储)（Mem0g）作为 typed edge (类型化边)，供关系查询。

在 `search(query, user_id)` 时：

1. Vector store 返回 embedding cosine 的 top-k。
2. KV store 返回查询派生的 (user_id, type, entity) 的直接命中。
3. Graph store 返回查询实体可达的 subgraph (子图)。
4. Scoring layer (打分层) 将三者融合。

### Fusion scoring (融合打分)

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance (相关性)** —— 向量 cosine、KV 精确匹配、图 path weight。
- **Importance (重要性)** —— 写入时打标或学习（某些事实更重要：姓名、ID、政策）。
- **Recency (时效性)** —— 自上次写入或读取以来的指数衰减。

权重按产品调优。Chat agent 提高 `w_recency`；合规 agent 提高 `w_importance`；检索 agent 提高 `w_relevance`。

### Mem0g 与时序推理

Mem0g 添加 conflict detector (冲突检测器)。当新事实与现有边矛盾时，现有边被标记 invalid (失效) 但不删除。时序查询（"用户三月份住在哪个城市？"）遍历 valid-at-time (当时有效) 的子图。

这是 Letta invalidation 模式泛化出的合规级行为。

### 基准数字

Mem0 论文报告（2025）：

- **LoCoMo**（长对话记忆）：91.6
- **LongMemEval**（长程情景记忆）：93.4
- **BEAM 1M**（百万词元记忆基准）：64.1

对比基线（全上下文 128k LLM、扁平向量存储、扁平 KV）全部落后 10 分以上。仅靠基准不足以 justify 选择 —— 运维形态才是 —— 但数字表明融合设计不是舍入误差。

### 作用域分类 (Scope taxonomy)

Mem0 按 scope (作用域) 拆分记忆：

- **User memory (用户记忆)** —— 跨会话持久化，键为 `user_id`。
- **Session memory (会话记忆)** —— 单线程内持久化。
- **Agent memory (智能体记忆)** —— 每个智能体实例的状态。

每次写入选择一个 scope。检索可以跨 scope 查询，带 per-scope weights。不加思考地混合 scope 会导致"助手把 Bob 的项目告诉了 Alice"这类事故。

### 该模式何时出错

- **Embedding drift (嵌入漂移)。** 前几百次查询看起来正确的向量结果，随语料库增长而退化。定期对 top-N 常用记录做 re-embedding。
- **KV schema creep (KV 模式蔓延)。** `(user_id, type, entity)` 看起来简单，直到每个团队都加自己的 `type`。每季度审计 type 集合。
- **Graph explosion (图爆炸)。** 一个嘈杂的 extractor 每条消息添加 50 条边。限制每次 `add` 的图写入数；丢弃低置信度边。

## 动手实现

`code/main.py` 用标准库实现了三存储模式：

- `VectorStore` —— 以词元重叠相似度作为 embedding 替身。
- `KVStore` —— 以 `(user_id, fact_type, entity)` 为键的字典。
- `GraphStore` —— typed edge (subject, relation, object, valid)。
- `Mem0` —— 顶层 facade，带 `add()`、`search()`、fusion scoring 和 scope-aware retrieval。
- 一段多用户、多会话对话的完整 trace。

运行：

```
python3 code/main.py
```

输出展示三条独立的召回路径以及融合后的 top-k。在 `main()` 顶部翻转 scoring 权重，观察排名变化。

## 如何使用

- **Mem0 (Apache 2.0)** —— 生产就绪。用 Postgres + Qdrant + Neo4j 自托管，或使用托管云。
- **Letta** —— 三层 core/recall/archival；自带 vector 和 graph 后端。
- **Zep** —— 商业替代方案，带 temporal KG (时序知识图) 和 fact extraction。
- **Custom builds** —— 当需要精确控制 extractor（合规）或 fusion weights（语音 agent 中 recency 占主导）时。

## 产物输出

`outputs/skill-hybrid-memory.md` 生成一个三存储记忆脚手架，带 fusion scorer、scope taxonomy 和时序失效机制。

## 练习

1. 将玩具向量相似度替换为真实 embedding 模型（sentence-transformers、Ollama、OpenAI embeddings）。在合成长对话上测量 recall@10。写入 1000 条后排名是否漂移？
2. 添加时序查询：`search(query, as_of=timestamp)`。仅返回在该时间戳或之前有效的记录。哪个存储需要最多改动？
3. 实现 conflict detector：如果传入事实与图边矛盾，invalidate 旧边并记录两者。在"用户住在 Berlin" -> "用户住在 Lisbon"上测试。
4. 将 fusion scorer 扩展为包含 `user_feedback` 维度（对检索记录点赞）。如何防止 gaming（智能体只返回它已经喜欢的记录）？
5. 阅读 Mem0 文档 (`docs.mem0.ai`)。将玩具实现移植到 `mem0` 客户端调用。在相同 20 条测试查询上比较检索质量。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Hybrid memory | "向量加图加 KV" | 三种存储并行写入，检索时融合 |
| Fact extraction | "记忆摄入" | LLM 步骤，将文本拆分为 (entity, relation, fact) 元组 |
| Fusion scoring | "相关性排序" | relevance + importance + recency 的加权和 |
| Scope | "记忆命名空间" | user / session / agent —— 决定谁能看到什么 |
| Mem0g | "记忆图" | 带时序有效性的 typed edge，用于关系查询 |
| Temporal invalidation | "软删除" | 标记矛盾边失效；永不删除 |
| Embedding drift | "检索腐化" | 语料库增长时向量质量退化；定期 re-embed |

## 延伸阅读

- [Chhikara 等, Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) —— 原始论文
- [Mem0 docs](https://docs.mem0.ai/platform/overview) —— 生产级 API、SDK、托管云
- [Packer 等, MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) —— 虚拟上下文前身
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) —— 三层兄弟设计
