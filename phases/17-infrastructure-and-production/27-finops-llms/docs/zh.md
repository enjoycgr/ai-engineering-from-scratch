# FinOps for LLMs — 单位经济学与多租户归因

> 传统 FinOps 在 LLM 支出上失效。成本是 token 交易，而非资源运行时间。标签无法自动映射 —— API 调用是交易，不是资产。工程决策（提示设计、上下文窗口、输出长度）就是财务决策。2026 年的操作手册要求在第一天就接入三个归因维度：per-user（`user_id`）用于席位定价和扩展，per-task（`task_id` + `route`）用于产品表面成本和优先级排序，per-tenant（`tenant_id`）用于单位经济学和续约。四个 token 层 —— prompt、tool、memory、response —— 合并到一个桶里会隐藏支出。多租户产品的执行阶梯：per-tenant rate limit（预期峰值的 2-3 倍，清晰的 429 + retry-after）；daily spend cap（合同上限的 1.5-3 倍；触发速率收紧 + 告警）；kill switch（spend z-score > 4，自动暂停 + 告警值班）。归因模式：tag-and-aggregate、telemetry-joiner（trace-ID → billing；最高精度）、sampling-and-extrapolation、model-based allocation、event-sourced、real-time streaming。单位指标：每次 resolved query 的成本、每次 generated artifact 的成本 —— 不是 $/M tokens。事后打标签总是遗漏；在请求创建时就接入。

**类型：** Learn
**语言：** Python（stdlib，带 kill switch 的玩具级成本归因模拟器）
**前置条件：** Phase 17 · 13（可观测性）、Phase 17 · 14（缓存）
**时间：** ~60 分钟

## 学习目标

- 解释为什么传统 FinOps（标签 + 层级）在 LLM 支出上失效，并说出三个新的归因维度。
- 列举四个 token 层（prompt、tool、memory、response）以及为什么单桶计费会隐藏成本。
- 为多租户产品设计执行阶梯（rate → spend cap → kill switch）。
- 选择单位指标（每次 resolved query / artifact 的成本）而非 $/M tokens。

## 问题

你的账单显示 $40,000。你不知道：
- 哪个租户花了这笔钱。
- 哪个产品功能驱动了它。
- 是否有单个用户滥用。
- 是 prompt bloat、tool call 还是 memory amplification 的锅。

Provider 端的 tag-and-aggregate 对云资源（EC2、S3）有效，因为标签会传播到明细项。LLM API 调用不会自动打标签 —— 你必须在调用点 stamp user/task/tenant 并全程携带。事后归因总是遗漏边缘情况。

## 概念

### 三个归因维度

**Per-user** (`user_id`)：谁在花什么钱。驱动席位定价、扩展对话、识别重度用户。

**Per-task** (`task_id` + `route`)：哪个产品表面花什么钱。驱动功能优先级排序、砍掉昂贵功能的决策。

**Per-tenant** (`tenant_id`)：哪个客户盈利。驱动单位经济学、续约定价、层级阈值。

第一天就在调用点接入所有三个。事后接入总是更差。

### 四个 token 层

| 层 | 示例 | 典型占总成本比例 |
|----|------|----------------|
| Prompt | system + user input | 40-60% |
| Tool | 反馈的 tool-call 结果 | 20-40%（agent 工作负载） |
| Memory | 先前对话 / 检索文档 | 10-30% |
| Response | 模型输出 | 10-30% |

将四层合并会让优化盲目。在归因 schema 中拆分它们。

### 执行阶梯

1. **Rate limit** per tenant。预期峰值的 2-3 倍。返回 429 并带 `Retry-After`。租户感受到摩擦；没有意外账单。

2. **Daily spend cap** per tenant。合同上限的 1.5-3 倍。触发：收紧 rate limit + 告警 customer-success。

3. **Kill switch**。相对于租户基线的 spend z-score > 4。自动暂停租户；告警值班；升级给 ops + CS。

### 归因模式

- **Tag-and-aggregate**：stamp 元数据头；稍后聚合。简单；粗略。
- **Telemetry joiner**：通过 trace ID 将 trace 与 billing 关联。最高精度。成熟团队的做法。
- **Sampling + extrapolation**：采样 5-10%，乘以系数。对粗略支出具有成本效益；遗漏尾部。
- **Model-based allocation**：回归推断成本驱动因素。用于无标签的遗留数据。
- **Event-sourced**：将成本作为流中的事件（Kafka / Kinesis）。实时。
- **Real-time streaming**：仪表盘亚秒级更新。

### Cost per X 是单位指标

$/M tokens 是 vendor 话术。产品指标：

- 每次 resolved support ticket 的成本。
- 每次 generated article 的成本。
- 每次 successful agent task 的成本。
- 每次 user-session-minute 的成本。

将成本与产品结果挂钩。否则优化没有锚点。

### 成本归因 trace 形状

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

每次调用都发出。存储在数据湖。按维度聚合。Phase 17 · 13 的可观测性栈就是它的归宿。

### 复合节省栈

栈：cache + batch + route + gateway。四个全上：
- Cache L2（Phase 17 · 14）：输入成本约便宜 10 倍。
- Batch（Phase 17 · 15）：50% 折扣。
- Route to cheap model（Phase 17 · 16）：成本降低 60%。
- Gateway efficiency（Phase 17 · 19）：冗余 + 重试。

最佳情况叠加：约 naive 基线的 5-10%。大多数团队启用了 2-3 个杠杆；很少堆叠全部四个。

### 应该记住的数字

- 归因维度：per-user、per-task、per-tenant。
- 四个 token 层：prompt、tool、memory、response。
- Kill switch：spend z-score > 4。
- 单位指标：每次 resolved query 的成本，不是 $/M tokens。
- 叠加优化：可能达到基线的 ~5-10%。

## 使用

`code/main.py` 模拟了一个带三级执行阶梯的多租户 LLM 服务。注入了一个滥用租户并演示 kill switch 触发。

## 交付

本节课产出 `outputs/skill-finops-plan.md`。给定产品和规模，设计归因 schema 和执行阶梯。

## 练习

1. 运行 `code/main.py`。Kill switch 在什么 z-score 触发？如何选择阈值？
2. 设计一个 per-tenant、per-task 成本仪表盘。你先构建哪 5 个视图？
3. 你的最大租户单位经济学为负。按客户影响排序提出三个干预措施。
4. 计算一个支持产品的每次 resolved ticket 成本：3M tokens/ticket，约 800 tickets/天，GPT-5 缓存费率。
5. 论证事后打标签是否可能有效。什么时候可以接受？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Per-user attribution | "用户级成本" | 每次调用都 stamp `user_id` |
| Per-task attribution | "功能成本" | `task_id` + `route` 识别产品表面 |
| Per-tenant attribution | "客户成本" | `tenant_id`；驱动单位经济学 |
| Four token layers | "成本层" | prompt + tool + memory + response |
| Rate limit | "429 护栏" | 在网关强制执行的 per-tenant 上限 |
| Daily spend cap | "每日上限" | 租户范围的预算加告警 |
| Kill switch | "自动暂停" | Spend z-score > 4 触发自动暂停 |
| Cost per resolved | "产品单位指标" | 成本与产品结果挂钩，而非 token |
| Telemetry joiner | "trace-to-billing" | 最高精度归因模式 |
| Stacked optimization | "cache+batch+route+gateway" | 复合节省至约基线的 5-10% |

## 延伸阅读

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
