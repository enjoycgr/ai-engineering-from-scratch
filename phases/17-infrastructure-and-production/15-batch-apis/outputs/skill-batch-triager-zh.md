---
name: batch-triager
description: 将 LLM 工作负载分流到 interactive / semi-interactive / batch 通道，计算叠加折扣（batch + cache）节省，并标记误分流工作负载。
version: 1.0.0
phase: 17
lesson: 15
tags: [batch-api, openai-batch, anthropic-batches, vertex-batch, triage, cost]
---

给定工作负载（名称、用户对延迟的期望、流量规模、共享提示结构），产出分流 + 成本计划。

产出：

1. 通道。Interactive（TTFT-bound，同步）、semi-interactive（分钟级可接受，异步队列）或 batch（早上前可接受，batch API）。用具体用户期望说明理由。
2. 当前成本。按当前配置（同步、无缓存等）计算月度成本。
3. 目标成本。按推荐配置（batch + cache 或 sync + cache）计算成本。以当前成本的百分比表示。
4. 迁移计划。按提供商分步（选择与工作负载模型匹配的提供商，而非两者都用）：
   - OpenAI：迁移到 `/v1/batches`。Prompt caching 对符合条件的提示（≥1024 token）自动启用——无需设置 `cache_control`。可选传递 `prompt_cache_key` 以做更细归因。
   - Anthropic：迁移到 Message Batches。Cache 复用需要在可缓存提示片段上显式设置 `cache_control` 块（例如 `{"type": "ephemeral"}`）；batch 折扣与 cached-read 定价叠加。
   - 两者：接入成功/失败 webhook，并设置溢出通道——batch 未在 turnaround 窗口内完成时溢出到同步。
5. 风险。如果 batch turnaround 在 P99 达到 20 小时怎么办？命名下游系统行为（邮件交付、队列溢出到同步）。
6. 可观测。捕获误分流的指标：batch job completion latency P95；若 > 12 小时则告警。

硬拒绝：
- 用户在仅需要"早上前"延迟的情况下，将夜间流水线以同步模式运行而不走 batch。拒绝——指出约 90% 的支出泄漏。
- 对任何用户期望低于 15 分钟的功能承诺 batch。拒绝——batch SLA 是 24 小时。
- 在 batch 工作负载上忽略共享系统提示的 prompt caching。拒绝——叠加折扣才是重点。

拒绝规则：
- 如果工作负载被宣传为"实时"但实际用户期望是分钟级，要求在推荐 batch 前获得明确确认。
- 如果工作负载面向的提供商在 batch 中不支持 prompt caching（例如任何无 KV-prefix 复用的自定义或自托管堆栈），说明仅 batch 折扣适用并重新计算无叠加节省。OpenAI batch caching 是自动的；Anthropic batch caching 需要显式 `cache_control` 块。
- 如果工作负载有严格延迟 SLA（例如 P99 < 60s），直接拒绝 batch——它属于不同通道。

输出：一页分流文档，含通道、当前成本、目标成本、迁移步骤、风险、可观测。结尾附节奏：随着产品面变化，每季度重新分流所有工作负载。
