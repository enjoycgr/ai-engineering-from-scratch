---
name: groupchat-selector
description: 为任务配置AutoGen/AG2风格的GroupChat selector (群聊选择器)，命名selector variant (选择器变体)、termination (终止条件)和anti-hot-speaker (防热发言者)规则。
version: 1.0.0
phase: 16
lesson: 10
tags: [multi-agent, groupchat, autogen, ag2, speaker-selection]
---

给定一个任务和一个agent roster (智能体名册)，生成一个GroupChat configuration (群聊配置)：selector choice (选择器选择)、selector inputs (选择器输入)、termination rules (终止规则)和guardrails (护栏)。

生成内容：

1. **Selector variant (选择器变体)。** Round-robin (轮询)（便宜、公平、context-blind (无视上下文)）、LLM-selected (LLM选择)（context-aware (上下文感知)、昂贵）或custom (自定义)（LLM + rule-based fallback (基于规则的回退)）。
2. **Selector inputs (选择器输入)。** 如果是LLM-selected：最近N条消息、agent specialties (智能体专长)、turn counts (轮次计数)。如果是custom：explicit rules (显式规则)。
3. **Termination rules (终止规则)。** Max rounds (最大轮次)、TERMINATE token、goal-reached verifier (目标达成验证器)或组合。
4. **Hot-speaker mitigation (热发言者缓解)。** Per-agent turn cap (每智能体轮次上限)、selector input中的speaker-balance score (发言者平衡分数)、K次连续轮次后的forced rotation (强制轮换)。
5. **Context bloat mitigation (上下文膨胀缓解)。** Projection plan (投影计划)（每角色的scoped views (作用域视图)）、summarization checkpoints (摘要检查点)、每agent的context cap (上下文上限)。
6. **Observability (可观测性)。** 记录selector's input (选择器输入)、selector's choice (选择器选择)、per-turn agent latency (每轮智能体延迟)。

Hard rejects (硬性拒绝)：

- 任何没有记录selector's input/output (选择器输入/输出)的LLM-selected config (LLM选择配置)。调试将变得不可能。
- 没有max_rounds cap (最大轮次上限)的配置。
- 在reasoning tasks (推理任务)上的symmetric chats (对称聊天)（无specialization (特化)）——改用debate (辩论)（第07课）。

Refusal rules (拒绝规则)：

- 如果任务具有已知的DAG structure (DAG结构)，拒绝GroupChat并推荐LangGraph static graph (静态图)以获得determinism (确定性)。
- 如果任务需要strict audit trails (严格审计跟踪)，拒绝GroupChat；推荐带checkpointer (检查点)的LangGraph。
- 如果agent数量超过5-6个，拒绝flat GroupChat (扁平群聊)并推荐nested groups (嵌套组)或hierarchical pattern (层级模式)。

Output (输出)：一页GroupChat config brief (群聊配置简报)。以cost estimate (成本估算)收尾（LLM-selected每轮产生一个selector call (选择器调用)）。
