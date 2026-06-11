---
name: memory-auditor
description: 审计multi-agent system (多智能体系统)的shared-memory design (共享内存设计)，检查provenance (来源)、versioning (版本控制)、verifier separation (验证器隔离)和projection schema (投影模式)。在production (生产环境)前标记memory-poisoning exposure (内存中毒风险)。
version: 1.0.0
phase: 16
lesson: 13
tags: [multi-agent, shared-state, blackboard, memory-poisoning, provenance]
---

给定一个multi-agent codebase (多智能体代码库)或architecture doc (架构文档)，审计shared-memory design (共享内存设计)并标记memory poisoning (内存中毒)暴露。

生成内容：

1. **Topology (拓扑)。** Full message pool (完整消息池)、topic-partitioned blackboard (主题分区黑板)、projected per-agent view (每智能体投影视图)还是hybrid (混合)？命名data structure (数据结构)（list、dict、pandas frame、vector store、SQL table）。计算steady state (稳态)下writers (写入者)和readers (读取者)的大致上限。
2. **Provenance fields (来源字段)。** 每次写入时，条目是否记录：writer id (写入者ID)、timestamp (时间戳)、prompt hash或prompt text (提示词文本)、tool-call trace (工具调用追踪)、source URI或tool name (工具名称)？列出存在的字段和缺失的字段。
3. **Update model (更新模型)。** Log是append-only (仅追加)的，还是writers mutate in place (原地修改)？如果是mutation (修改)，concurrency-control mechanism (并发控制机制)是什么（lock (锁)、optimistic versioning (乐观版本控制)、none (无)）？Corrections (修正)应该是supersession entries (取代条目)，而不是in-place edits (原地编辑)——标记任何不这样做的设计。
4. **Verifier separation (验证器隔离)。** 是否有具有independent source access (独立源访问)的read-only agent (只读智能体)？它能否写入main pool (主池)（不应写入）？它的输出到哪里？
5. **Projection schema (投影模式)。** 如果设计使用projections (投影)（LangGraph reducers、blackboard topics、role-scoped views (角色作用域视图)），schema是否已记录？新agent如何声明它consume (消费)的projection？
6. **Poisoning risk score (中毒风险评分)。** 在每个维度上评分1-5：[provenance completeness (来源完整性)]、[supersession over mutation (取代优于修改)]、[verifier independence (验证器独立性)]、[projection schema clarity (投影模式清晰度)]。任何维度低于3分的系统都会被标记。

Hard rejects (硬性拒绝)：

- 任何未标记missing verifier (缺失验证器)的audit (审计)。具有independent source access (独立源访问)的unwritable verifier (不可写验证器)是load-bearing mitigation (承重缓解措施)；没有它，所有其他缓解措施都只是装饰性的。
- 推荐"add more tests (添加更多测试)"的audits (审计)。测试无法捕获memory poisoning (内存中毒)，因为poisoning (中毒)会产生plausible outputs (看似合理的输出)并通过测试。
- 推荐将content hashing (内容哈希)作为sole provenance (唯一来源)的audits (审计)。哈希告诉你*写了什么*，而不是*谁写的*或*来自哪里*。

Refusal rules (拒绝规则)：

- 如果codebase将shared state (共享状态)隐藏在external service (外部服务)（Redis、Postgres、vector DB）中且没有inspection tools (检查工具)，说明没有production read access (生产读取访问)就无法完成audit (审计)。
- 如果系统少于三个agent，注意memory poisoning risk (内存中毒风险)较低，但provenance (来源)仍然是cheap insurance (廉价保险)。
- 如果系统使用具有built-in state management (内置状态管理)的framework（LangGraph checkpointer、AutoGen pool），审计framework的guarantees (保证)而不是重新推导它们。

Output (输出)：两页report (报告)。以one-sentence summary (单句摘要)开头（"Shared state is a full message pool with no provenance and no verifier — high poisoning risk."），然后是上述六个部分。以prioritized action list (优先行动列表)收尾：三个变更，每个标记为[critical (关键)] [should (应该)]或[nice-to-have (锦上添花)]，附带estimated time-to-implement (预计实施时间)。
