# Batch API（批处理 API）—— 五折优惠已成为行业标准

> 每家主流提供商都推出了异步 batch API（批处理 API），提供 50% 折扣， turnaround（周转时间）约 24 小时。OpenAI、Anthropic、Google 以及大多数推理平台（Fireworks batch tier、Together batch）均采用相同模式。将 batch 与 prompt caching（提示缓存）叠加使用，夜间流水线成本可降至同步无缓存成本的约 10%。规则极其简单：如果不是交互式任务，就应该走 batch。内容生成流水线、文档分类、数据提取、报告生成、批量标注、目录打标签——任何可容忍 24 小时延迟的任务，在未迁移到 batch 之前都是在浪费钱。2026 年的生产模式是将每个新的 LLM（大语言模型）工作负载分流到三条通道：交互式（同步+缓存）、半交互式（异步队列+兜底）、batch（夜间+缓存输入叠加）。那些假装是交互式但实际上可容忍数分钟延迟的工作负载，浪费最为严重。

**Type:** Learn
**Languages:** Python（stdlib，toy batch-vs-sync cost simulator）
**Prerequisites:** Phase 17 · 14（Prompt & Semantic Caching）
**Time:** ~45 minutes

## Learning Objectives

- 说出三家提供商的 batch API（OpenAI、Anthropic、Google）以及通用的 50% 折扣 + 24 小时 turnaround 保证。
- 计算在夜间分类工作负载上叠加 batch + cached-input（缓存输入）的成本，并与同步无缓存基线对比。
- 将工作负载分流为 interactive（交互式）/ semi-interactive（半交互式）/ batch（批处理）并说明理由。
- 说出两个陷阱：partial interactivity（部分交互性，用户期望快于 24 小时）和 output-schema drift（输出格式漂移，各提供商 batch 文件格式不同）。

## The Problem

你的团队运行一个夜间报告生成流水线。50,000 份文档，每份总结，聚类摘要，起草执行简报。同步运行每晚耗时 4 小时，成本 $2,000。你听说了 batch API。

Batch 给你 50% 折扣。你还对系统提示（所有 50k 次调用共享）启用了 prompt caching（提示缓存）。叠加后，账单降至每晚 $180——约为基线的 9%。同样的流水线，三个配置变更。

Batch 是 LLM 成本工具箱中最便宜但没人用的杠杆。原因主要是组织性的：团队一想到"实时"，而实际 SLA 只是"早上前完成"。这节课讲的是不要把 90% 的账单留在桌上。

## The Concept

### The three batch APIs

**OpenAI Batch API**：上传 JSONL 文件，包含请求列表。承诺 24 小时 turnaround（实际通常约 2-8 小时）。输入和输出 token 均享 50% 折扣。`/v1/batches` 端点。符合条件的缓存输入还可叠加 cached-input pricing。

**Anthropic Message Batches**：JSONL 上传。24 小时 turnaround。50% 折扣。支持 `cache_control`——缓存写入显式，读取在 batch 内自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS 输入。Gemini 同样约 50% 折扣。与 Vertex 流水线集成。

### Semantic: asynchronous, not slow

Batch 是"我承诺 24 小时内返回"——不是"这需要 24 小时"。典型 P50 为 2-6 小时。提供商将你的 batch 调度到 GPU 库存利用率较低的低谷时段。

### Stack with caching

50k 文档摘要，使用相同的 4K-token 系统提示：

- Synchronous uncached：50000 × ($input × 4000 + $output × 200)，按全价计费。
- Synchronous cached：系统提示在首次写入后缓存；剩余 49999 次输入便宜约 10 倍。
- Batch cached：以上全部再叠加 50% 折扣。

叠加效果：batch + cache = 同步无缓存账单的约 10%。任何夜间运行且共享系统提示的工作负载都应使用此方案。

### Workload triage

**Interactive** — 用户等待响应。TTFT（首token时间）至关重要。同步调用+prompt caching。不能走 batch。

**Semi-interactive** — 用户提交任务，数分钟后回来查看。异步队列，batch 不可用时兜底同步。中等规模 RAG 索引即属此类。

**Batch** — 用户期望"早上前"或"下小时前"拿到结果。内容流水线、大规模分类、离线分析。永远走 batch，永远叠加缓存。

常见错误：因为流水线是生产环境就把所有任务归为 interactive。Production 不是延迟规格——SLA 才是。

### The partial-interactivity trap

某些功能看起来是 interactive，但可容忍 5-10 分钟。例如：夜间客户健康报告带"刷新"按钮。用户点击刷新，等 10 分钟没问题。团队却按同步实现。50 个并发刷新成本是批量处理后邮件交付的 10 倍。

要问的问题："24 小时对这个用户意味着什么？"如果答案是"他们不会注意到"，那就走 batch。

### The output-schema trap

各提供商 batch 文件格式不同：

- OpenAI：JSONL，每行一个请求。
- Anthropic：JSONL，每行一个消息；响应格式内嵌。
- Vertex：BigQuery 表或 GCS 前缀，TFRecord。

写一个跨提供商的"统一 batch client"意味着每个提供商都要适配代码。宣传多提供商 batch 的网关（Portkey、LiteLLM 某些层级）仍然是对原始格式的薄封装。

### Numbers you should remember

- 各提供商 batch 折扣：输入+输出统一 50%。
- Turnaround SLA：保证 24 小时，典型 P50 为 2-6 小时。
- Batch + cached input 叠加：约为同步无缓存成本的 10%。
- Workload triage 规则：如果可容忍 24 小时延迟，永远走 batch。

## Use It

`code/main.py` 计算 50k 文档工作负载在 sync、sync+cache、batch、batch+cache 四种配置下的成本。以金额和百分比报告节省。

## Ship It

本课产出 `outputs/skill-batch-triager.md`。根据工作负载特征，分流到 interactive/semi/batch 并估算节省。

## Exercises

1. 运行 `code/main.py`。对于 100k 文档流水线，3K-token 系统提示，500-token 输出，计算 full stack（batch + cache）vs sync 基线的节省。
2. 选一个你熟悉的真实产品，挑出三个功能。将每个分流为 interactive/semi/batch。
3. 用户抱怨报告花了 3 小时。这是 batch 误分流还是 legitimate interactive？写出决策标准。
4. 你的 batch API 返回 SLA 是 24 小时，但 P99 是 20 小时。如何向用户传达——边缘情况下下游系统应如何表现？
5. 计算盈亏平衡点：共享前缀多长时，batch + cache 比自己预留 GPU 夜间运行更便宜？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Batch API | "async discount" | 50% off with 24h turnaround |
| JSONL | "batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | Guarantee, not typical; typical is 2-6h |
| Workload triage | "interactivity decision" | Interactive / semi / batch routing decision |
| Output schema | "response format" | Per-provider JSONL layout; not portable |
| Stacked discount | "batch + cache" | ~10% of uncached sync bill when both apply |

## Further Reading

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format and `/v1/batches` semantics.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — batch format and `cache_control` interaction.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
