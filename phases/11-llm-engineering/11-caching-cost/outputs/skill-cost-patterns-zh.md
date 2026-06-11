---
name: skill-cost-patterns
description: LLM 成本优化的决策框架 — 缓存策略、速率限制、模型路由和预算控制
version: 1.0.0
phase: 11
lesson: 11
tags: [caching, cost-optimization, rate-limiting, model-routing, budget, llm-ops]
---

# LLM Cost Optimization Patterns

构建需要控制成本的 LLM 应用时，应用此决策框架。

## 何时优化

**立即优化当：**
- 月度 LLM 支出超过 $500 或基础设施预算的 10%
- 消费者产品的每次查询成本超过 $0.01
- 你的 system prompt 超过 1,000 token 且随每个请求发送
- 超过 30% 的查询是重复或近似重复的
- 你从 100 扩展到 10,000+ 日活用户

**还不需要优化当：**
- 你的 DAU 少于 100，仍在验证产品-市场契合度
- 月度支出低于 $100 且增长缓慢
- 仍在迭代 prompt 设计（缓存会锁定你到一个 prompt）

## 缓存策略选择

### Exact caching (精确缓存)

**使用当：** temperature=0、完全相同的 prompt 重复、需要确定性输出。

```python
key = sha256(json.dumps({"model": m, "messages": msgs, "temp": 0}))
```

- 实现：30 分钟
- 命中率：大多数应用 10-25%，FAQ 机器人 40-60%
- 延迟：<1ms（字典查找）
- 风险：底层数据变化时响应过时

**跳过当：** temperature > 0、每个查询都唯一、需要实时数据。

### Semantic caching (语义缓存)

**使用当：** 用户用不同措辞问相同问题、FAQ -heavy 产品、客户支持。

- 实现：2-4 小时（embedding + 相似度 + 存储）
- 命中率：在精确缓存之上额外 15-35%
- 延迟：10-50ms（embedding + ANN 搜索）
- 风险：false positives（对相似但不同的问题返回错误的缓存答案）

**阈值指导：**
- 0.98+：非常保守，几乎没有 false positives，命中率较低
- 0.95：事实性 Q&A 的良好平衡
- 0.90：激进，命中率更高但有错误答案风险
- 0.85：仅适用于低风险应用（建议、自动补全）

**跳过当：** 每个查询都有唯一上下文（代码生成）、响应必须反映最新数据、查询空间无界。

### Provider prompt caching

**使用当：** system prompt > 1,024 tokens（OpenAI）或模型特定最小值，相同前缀重复发送。

| Provider | Action | Savings |
|----------|--------|---------|
| Anthropic | 在 system message 上添加 `cache_control: {"type": "ephemeral"}` | 缓存命中 90%（写入时付 25% 溢价） |
| OpenAI | 无需操作（自动） | 缓存前缀 50% |
| Google | 使用带显式 TTL 的 Context Caching API | 缓存上下文约 75% |

**跳过当：** system prompt 每请求都变化、prompt 低于最小长度。

## 模型路由规则

### 基于关键词（简单、快速）

```
simple:  <= 5 个词 OR 匹配 FAQ 关键词 -> gpt-4o-mini ($0.15/$0.60)
medium:  一般查询、摘要        -> claude-sonnet ($3/$15)
complex: "analyze", "compare", "debug"     -> gpt-4o ($2.50/$10)
```

- 实现：1 小时
- 准确率：70-80%
- 节省：模型成本的 40-60%

### 基于 embedding（更准确）

每个类别嵌入 50-100 个带标签的查询。通过最近邻分类新查询。

- 实现：4-8 小时
- 准确率：85-92%
- 节省：模型成本的 50-70%
- 额外成本：分类 embedding 约 $0.02/1M token（可忽略）

### 基于 ML（生产级）

在历史查询/模型对上训练一个小型分类器（逻辑回归或小型 BERT）。

- 实现：1-2 周
- 准确率：90-95%
- 节省：模型成本的 60-75%
- 需要：来自生产流量的带标签训练数据

## 速率限制配置

### 按层级的令牌桶参数

| Tier | Bucket Size | Refill Rate | Max RPM | Daily Cap |
|------|-------------|-------------|---------|-----------|
| Free | 50K tokens | 500/sec | 10 | 50K |
| Pro | 500K tokens | 5K/sec | 60 | 500K |
| Enterprise | 5M tokens | 50K/sec | 300 | 5M |

### 实现清单

1. 将桶存储在 Redis 中（而非内存），用于多实例应用
2. 使用原子操作（MULTI/EXEC）防止竞态条件
3. 拒绝响应中返回 `Retry-After` header
4. 将拒绝请求作为指标跟踪（>5% 拒绝 = 层级限制过紧）
5. 实现优雅降级：优先拒绝昂贵模型请求，保留便宜模型访问

## 预算控制

### 三阈值断路器

| Threshold | Action | Reversible |
|-----------|--------|------------|
| 月度预算的 70% | 记录警告，通过 Slack/PagerDuty 告警团队 | 是（自动） |
| 月度预算的 85% | 将所有流量路由到最便宜的模型 | 是（自动，下个计费周期） |
| 月度预算的 95% | 仅提供缓存响应，拒绝新的 LLM 调用 | 是（手动重置或下个周期） |

### 每用户成本跟踪

跟踪每个用户的累计成本。标记超过中位数 10 倍的用户。常见原因：
- 合法的超级用户（升级他们的层级）
- Prompt injection 循环（机器人发送自动化请求）
- 低效的集成（客户端在每个错误上重试）

## 成本跟踪字段

用以下字段记录每次 API 调用：

```json
{
  "timestamp": "2026-04-02T10:30:00Z",
  "model": "gpt-4o",
  "input_tokens": 1523,
  "output_tokens": 487,
  "cached_input_tokens": 1024,
  "latency_ms": 1847,
  "cost_usd": 0.006142,
  "user_id": "user_abc123",
  "cache_status": "partial_hit",
  "request_category": "customer_support",
  "complexity_class": "medium",
  "routed_from": "gpt-4o"
}
```

### 需要仪表板展示的关键指标

- **Cost per query** (P50, P95, P99) —— 按模型、按功能、按用户层级
- **Cache hit rate** —— 精确 vs 语义，随时间趋势
- **Model distribution** —— 每个模型的流量占比、每个模型的成本
- **Budget burn rate** —— 当前支出 vs 按当前速率的预计月度支出
- **Rejection rate** —— 被速率限制的请求百分比，按层级

## 常见错误

| Mistake | Why it hurts | Fix |
|---------|-------------|-----|
| 用 temperature > 0 做缓存 | 非确定性输出，陈旧缓存给出错误的多样性 | 只缓存 temp=0 的调用，或接受缓存响应失去随机性 |
| Semantic cache 阈值太低 | 对表面相似查询返回错误答案 | 从 0.95 开始，仅在测量 false positive 率后才降低 |
| 没有缓存失效机制 | 底层数据变化时响应过时 | 设置 TTL（动态数据 1 小时，静态数据 24 小时），数据更新时失效 |
| 将所有流量路由到最便宜的模型 | 质量下降，用户感知 | 按复杂度路由，测量每层级质量，设置最低质量阈值 |
| 没有每用户限制 | 一个滥用用户烧完整个预算 | 始终实现每用户配额，即使很宽松 |
| 忽略输出 token | 输出成本是输入的 2-5 倍 | 适当设置 max_tokens，使用 stop sequences，压缩输出 |
| 在 prompt 稳定前缓存 | 缓存填满旧 prompt 的响应 | 仅在 prompt 最终确定后启用缓存，prompt 变更时刷新缓存 |

## 定价参考（截至 2026 年 4 月）

| Model | Input ($/1M) | Output ($/1M) | Cached Input ($/1M) | Best For |
|-------|-------------|--------------|--------------------|---------|
| gpt-4.1-nano | $0.10 | $0.40 | $0.025 | 高容量简单任务 |
| gpt-4o-mini | $0.15 | $0.60 | $0.075 | 简单路由、分类 |
| gemini-2.5-flash | $0.15 | $0.60 | $0.0375 | 预算型多模态 |
| claude-haiku-3.5 | $0.80 | $4.00 | $0.08 | 快速中层级任务 |
| o4-mini | $1.10 | $4.40 | $0.275 | 预算型推理 |
| gemini-2.5-pro | $1.25 | $10.00 | $0.3125 | 长上下文、多模态 |
| gpt-4o | $2.50 | $10.00 | $1.25 | 通用、function calling |
| claude-sonnet-4 | $3.00 | $15.00 | $0.30 | 质量/成本平衡 |
| claude-opus-4 | $15.00 | $75.00 | $1.50 | 最高质量、复杂推理 |
