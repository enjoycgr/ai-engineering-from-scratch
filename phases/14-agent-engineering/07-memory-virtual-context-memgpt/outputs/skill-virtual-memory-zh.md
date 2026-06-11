---
name: virtual-memory
description: 为任何目标 runtime 搭建 MemGPT 风格的两层记忆系统（主上下文 + 归档存储 + 记忆工具），具备正确的驱逐策略、引用机制和不可信输入处理。
version: 1.0.0
phase: 14
lesson: 07
tags: [memory, memgpt, virtual-context, archival, citations]
---

给定目标 runtime（Python、Node、Rust）、模型供应商（Anthropic、OpenAI、本地）和存储后端（内存、SQLite、vector DB、KV、graph），产出正确的 MemGPT 风格记忆系统。

产出：

1. `MainContext` 类型，带 `core` 字典（命名持久化段落）和 `messages` 列表（FIFO）。超出大小上限时自动驱逐；被驱逐的轮次仍可通过 `conversation_search` 检索。
2. `ArchivalStore`，支持 insert 和 search。记录**必须**携带 `id`、`text`、`tags`、`session_id`、`turn_id`、`created_at`。每次写入返回存储 id 供引用。
3. 五个匹配 MemGPT 表面的记忆工具：`core_memory_append`、`core_memory_replace`、`archival_memory_insert`、`archival_memory_search`、`conversation_search`。将它们呈现给模型时，附带 `description` 文本，说明何时使用每个工具。
4. 引用契约：每次归档检索**必须**返回记录 id 与文本，智能体**必须**在最终回答中引用它们。无引用的回答是 soft failure (软性失败)。
5. 一个 consolidation hook（v1 中可为空操作），以便第 08 课的 sleep-time agents 可即插即用，无需重新布线。暴露 `list_records_since(timestamp)` 和 `delete(id)`。

硬性拒绝：

- 用全 prompt LLM 打分来搜索归档。使用正确的检索后端（BM25、向量相似度）。允许对 top-k 短名单做 LLM re-ranking，但不能对完整语料库做。
- 主上下文无驱逐策略。无界主上下文会默默超出窗口。
- 将检索到的内容当作用户指令存储。所有归档内容都是 untrusted text (不可信文本)（第 27 课）。将其作为 observation 传给模型，而非 system prompt。
- 提供 `core_memory_clear` 工具以 wiping all sections。Core 是 load-bearing；清空是 foot-gun。支持 `replace` 而非 `clear`。

拒绝规则：

- 如果用户要求"不要引用，只要答案"，拒绝任何需要来源归因的领域（医疗、法律、政策、金融）。提供折中：引用以脚注形式呈现，而非内联。
- 如果用户要求"将所有检索到的内容不加过滤地写回归档"，拒绝并指向第 27 课。检索内容是 attacker-reachable (攻击者可触及的)；批量写回就是 memory poisoning (记忆投毒)。
- 如果 runtime 没有持久化层，拒绝交付一个被描述为"长期记忆"的智能体。降级产品描述，而非实现。

输出：每个组件一个文件（`main_context.*`、`archival_store.*`、`memory_tools.*`、`agent.*`），以及一份 `README.md`，解释驱逐策略、引用契约，以及在哪里接入第 08 课（sleep-time consolidation）和第 09 课（Mem0 融合）。结尾附带"接下来读什么"的指引：如果智能体需要三层或异步 consolidation，指向第 08 课；如果需要 vector+KV+graph 融合，指向第 09 课。
