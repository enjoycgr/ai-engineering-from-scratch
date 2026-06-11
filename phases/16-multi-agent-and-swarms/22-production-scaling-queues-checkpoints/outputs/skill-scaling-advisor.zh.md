---
name: scaling-advisor
description: 为multi-agent production system (多智能体生产系统)提供durable-execution choice (持久执行选择)建议。根据具体负载和state-retention needs (状态保留需求)在FastAPI + Postgres、LangGraph runtime、Temporal、Restate或custom之间选择。
version: 1.0.0
phase: 16
lesson: 22
tags: [multi-agent, production, scaling, durable-execution, queues, checkpoints]
---

给定一个multi-agent production deployment plan (多智能体生产部署计划)，推荐durable-execution substrate (持久执行底层)。

生成内容：

1. **Load profile (负载概况)。** Concurrent agent-runs (并发智能体运行)（p50、p99）。Per-run duration (单次运行持续时间)（秒到小时）。需要human-in-the-loop waits (人工介入等待)的runs比例。Deploy frequency (部署频率)。
2. **State profile (状态概况)。** Per-run state size (单次运行状态大小)（KB到MB）。Retention requirement (保留需求)（seconds of checkpoint history (检查点历史秒数)或full audit log (完整审计日志)）。Determinism (确定性)：runs能否从checkpoints deterministically replay (确定性重放)，还是只能从logs (日志)replay？
3. **Side-effect profile (副作用概况)。** 哪些side effects需要exactly-once (恰好一次)（payments (支付)、external APIs (外部API)、email (邮件)）？哪些可以容忍at-least-once (至少一次)（pure tool reads (纯工具读取)）？Exactly-once需要outbox pattern (发件箱模式)。
4. **Recommendation tier (推荐层级)。**
   - Tier 1 (Bedi's rule)：FastAPI + Postgres。约100个concurrent runs (并发运行)以下、sub-hour durations (低于一小时的持续时间)、simple retries (简单重试)。
   - Tier 2：LangGraph runtime或Temporal。Hour-long runs (长达一小时的运行)、interrupt/resume (中断/恢复)、structured retries (结构化重试)。
   - Tier 3：Custom with outbox + event sourcing (自定义配合发件箱+事件溯源)。Specialized needs (特殊需求)、high throughput (高吞吐量)、strict audit (严格审计)。
5. **Deploy model (部署模型)。** Single version (单版本)还是rainbow/canary (彩虹/金丝雀)？Rainbow (彩虹)对long-running stateful workloads (长时间运行有状态工作负载)是必需的。
6. **Async / thread boundary (异步/线程边界)。** 哪些是async (异步)的（LLM calls、tool I/O）以及哪些是threads/processes (线程/进程)（CPU-bound post-processing (CPU密集型后处理)、embedding (嵌入)）。
7. **Observability (可观测性)。** Per-run traces (每次运行追踪)、super-step audit (超级步骤审计)、retry counter (重试计数器)。Traces的storage (存储)（与checkpoint store分离）。

Hard rejects (硬性拒绝)：

- 对10-concurrent-run prototype (10并发运行原型)推荐Temporal。Ceremony cost (仪式成本) > value (价值)。
- Thread-per-job LLM call architectures (每任务线程LLM调用架构)。I/O-bound + 1MB/thread无法scale (扩展)。
- 对paid side effects (付费副作用)没有outbox pattern的设计。Duplicate charges (重复扣费)很昂贵。
- 对multi-hour agent runs (数小时智能体运行)使用single-version deploys (单版本部署)。每次code push都会让用户丢失state (状态)。

Refusal rules (拒绝规则)：

- 如果负载未知且未测试，推荐Tier 1加load testing (负载测试)。Premature optimization (过早优化)浪费时间。
- 如果用户想要tokenized / blockchain-persistent system (代币化/区块链持久化系统)，说明durable-execution engines通常不解决该问题（编写自己的event sourcing (事件溯源)）；推荐tokenized flows的法律审查。
- 如果团队没有on-call engineer (值班工程师)，Temporal / LangGraph runtime maintenance是under-provisioned (配置不足)的；推荐Tier 1直到有值班人员。

Output (输出)：两页brief (简报)。以one-sentence recommendation (单句建议)开头（"Tier 1 (FastAPI + Postgres + outbox) for current load; escalate to LangGraph runtime when p99 run duration exceeds 10 min or concurrent runs exceed 200."），然后是上述七个部分。以90-day upgrade path (90天升级路径)收尾：metrics to watch (关注指标)、escalation threshold (升级阈值)、runbook outline (运行手册大纲)。
