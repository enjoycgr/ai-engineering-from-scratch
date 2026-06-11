---
name: swarm-fit
description: 判断一个任务是否适合swarm (集群)架构（decentralized (去中心化)）或supervisor (监督者)架构（centralized (集中化)）。
version: 1.0.0
phase: 16
lesson: 09
tags: [multi-agent, swarm, decentralized, langgraph, matrix]
---

给定一个任务及其throughput (吞吐量) / determinism (确定性)需求，推荐swarm (集群)或supervisor (监督者)并列出具体的queue (队列)和guardrail (护栏)选择。

生成内容：

1. **Task independence check (任务独立性检查)。** Subtasks (子任务)是独立的还是相互依赖的？只有当independence (独立性)很高时，swarm (集群)才适合。
2. **Duration distribution (持续时间分布)。** Uniform (均匀) vs variable (可变)。Swarm (集群)主要在variable-duration workloads (可变持续时间工作负载)上获胜。
3. **Ordering requirement (排序需求)。** Strict (严格)、relaxed (宽松)或none (无)。Swarm (集群)不保留order (顺序)；supervisor (监督者)保留。
4. **Debuggability need (可调试性需求)。** 高（金融、医疗）→ supervisor (监督者)。中等 → swarm (集群)配合per-task trace IDs (每任务追踪ID)。
5. **Queue choice (队列选择)。** In-memory (`queue.Queue`)用于演示；Kafka / Redis Streams / NATS / durable DB-backed (持久化数据库 backed)用于production (生产环境)。
6. **Worker design requirements (工作者设计要求)。** 必须是idempotent (幂等的)；必须emit per-task trace (发出每任务追踪)；必须handle back-pressure (处理背压)。
7. **Anti-starvation plan (反饥饿计划)。** Priority aging (优先级老化)、worker specialization (工作者特化)、bounded queue (有界队列)。
8. **Observability plan (可观测性计划)。** Per-task IDs (每任务ID)、start/end events (开始/结束事件)、result pool schema (结果池模式)。

Hard rejects (硬性拒绝)：

- 对具有hard ordering requirements (严格排序需求)的任务推荐swarm (集群)。
- 没有idempotent workers (幂等工作者)的swarm (集群)。
- 在生产环境中没有durable queue (持久化队列)的swarm (集群)。

Refusal rules (拒绝规则)：

- 如果任务每秒少于10个independent units (独立单元)，拒绝swarm (集群)并推荐supervisor (监督者)。在低throughput (吞吐量)下，swarm (集群)的开销不合理。
- 如果observability requirements (可观测性需求)需要single coherent trace (单一连贯追踪)（audit (审计)、compliance (合规)），拒绝swarm (集群)并推荐LangGraph deterministic graph (确定性图)。

Output (输出)：一页architectural brief (架构简报)。以fit verdict (适配判定)开头，以针对目标throughput (吞吐量)的具体message broker recommendation (消息代理推荐)收尾。
