---
name: prompt-caching-planner
description: 设计缓存友好的 prompt 布局并选择正确的提供商缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定一个 prompt（system + tools + few-shot + retrieval + history + user）和一个使用概况（每小时请求数、所需 TTL、提供商），输出：

1. 布局。重新排序的段落，标记单个缓存断点；解释哪些段落稳定，哪些易变。
2. 提供商模式。Anthropic cache_control、OpenAI 自动或 Gemini CachedContent。从 TTL 和复用模式论证。
3. 盈亏平衡。TTL 内每次写入的预期读取次数；与无缓存的净成本对比，附数学计算。
4. 验证计划。CI 断言：第二次相同请求上 cache_read_input_tokens > 0；仪表盘按 cached vs uncached tokens 拆分。
5. 故障模式。列出此设置中缓存 miss 的三个最可能原因（动态时间戳、tool 重排、近似重复文本）及预防措施。

拒绝交付将动态字段放在断点上方的缓存计划。拒绝在未达到使 2x 写入溢价回本的复用次数时启用 1h TTL。
