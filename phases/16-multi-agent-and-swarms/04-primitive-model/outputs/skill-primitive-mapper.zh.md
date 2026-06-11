---
name: primitive-mapper
description: 将任何multi-agent framework (多智能体框架)或代码库映射到四个primitive axes (原语维度)（agent、handoff、shared state、orchestrator）。
version: 1.0.0
phase: 16
lesson: 04
tags: [multi-agent, primitives, framework-comparison, architecture]
---

给定一个multi-agent framework (多智能体框架)（或使用了某个框架的代码库），生成four-primitive mapping (四原语映射)，使读者能够在一个段落内理解该框架。

生成内容：

1. **Agent definition (智能体定义)。** agent (智能体)是如何构建的？有哪些参数？它携带什么state (状态)？命名确切的class (类)或factory (工厂)。
2. **Handoff mechanism (交接机制)。** 它使用三种handoff patterns (交接模式)中的哪一种——function return (函数返回)、graph edge (图边)或speaker selection (发言者选择)？如果是hybrid (混合)，哪种是主要的？展示触发一次handoff (交接)的最小代码。
3. **Shared state model (共享状态模型)。** Full message pool (完整消息池)还是projected view (投影视图)？In-memory (内存中)还是durable (持久化)（checkpointed (检查点)）？是否对concurrent writers (并发写入者)是thread-safe (线程安全)的？谁reconciles conflicts (协调冲突)？
4. **Orchestrator type (编排器类型)。** Static (静态)、LLM-selected (LLM选择)、handoff-driven (交接驱动)还是queue-driven (队列驱动)？如果是LLM-selected，默认使用哪个model (模型)？如果是static，graph (图)是cyclic (循环)还是DAG (有向无环图)？
5. **Cross-axis tradeoffs (跨维度权衡)。** 每个维度一句话：determinism (确定性)、scalability ceiling (可扩展性上限)、debuggability (可调试性)、typical failure mode (典型故障模式)。

Hard rejects (硬性拒绝)：

- 任何声称某个abstraction (抽象)是"new (全新)"的映射，但未展示它不会collapse (坍缩)为四个primitives (原语)之一。如果你无法reduction (归约)它，精确命名gap (缺口)而不是发明第五个primitive (原语)。
- 只引用marketing docs (营销文档)的framework comparisons (框架比较)。始终引用框架仓库或官方cookbook ( cookbook)中的具体代码示例。
- 诸如"Framework X is better for agents (框架X更适合agent)"的表述，未指明框架优化了哪个primitive (原语)。

Refusal rules (拒绝规则)：

- 如果框架是closed-source (闭源)的，且公开文档未暴露agent-handoff-state-orchestrator surface (智能体-交接-状态-编排器接口)，说明没有internals (内部实现)就无法进行mapping (映射)。
- 如果用户提供了一个代码库但没有framework (框架)（hand-rolled agents (手写agent)），改为映射custom implementation (自定义实现)，并标出哪个primitive (原语)是under-designed (设计不足的)。
- 如果框架早于2024年（原始AutoGen v0.2、pre-Swarm）且不再维护，包含一行注释说明其继任者是否保留了该mapping (映射)。

Output (输出)：一页framework brief (框架简报)。以single-sentence summary (单句摘要)开头（"Framework X fixes handoff as graph edge and exposes shared state via a reducer."），然后是上述五个部分，最后是一段closing paragraph (收尾段落)，说明该框架的primitives (原语)最适合哪个production project (生产项目)。
