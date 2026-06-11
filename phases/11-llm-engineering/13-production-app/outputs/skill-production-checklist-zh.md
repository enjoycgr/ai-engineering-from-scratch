---
name: skill-production-checklist
description: 交付 LLM 应用到生产环境的决策框架 —— 涵盖每个组件的具体阈值和 pass/fail 标准
version: 1.0.0
phase: 11
lesson: 13
tags: [production, deployment, llm, architecture, scaling, cost, observability, guardrails]
---

# 生产级 LLM 检查清单

交付 LLM 应用时，按顺序完成此检查清单。每个部分都有 pass/fail 标准和具体阈值。

## 1. 安全（上线阻塞项）

此处每项必须在任何部署前通过。

| 检查 | 通过标准 | 验证方式 |
|-------|--------------|---------------|
| API keys 在环境变量中 | 代码库中零硬编码 keys | `grep -r "sk-" --include="*.py"` 返回空 |
| Input guardrails 启用 | Prompt injection 模式被拦截 | 发送 "Ignore all previous instructions" —— 返回被拦截响应 |
| PII 脱敏 | SSN、信用卡、邮箱模式被捕获 | 发送 "My SSN is 123-45-6789" —— PII 在 LLM 调用前被脱敏 |
| Output 过滤 | 危险内容被拦截 | 模型无法返回 `DROP TABLE`、`rm -rf`、`exec()` 模式 |
| 限流 | 每用户请求上限强制执行 | 同一用户在 10 秒内发送 100 个请求 —— 最后 50+ 被拒绝 |
| 所有端点需认证 | 无未认证的 LLM 访问 | `curl /v1/chat` 不带 token 返回 401 |
| CORS 限制 | 仅允许生产域名 | `Origin: evil.com` 请求被拒绝 |
| 最大 input tokens | 超限请求被拒绝 | 发送 50K token input —— 返回 413 或截断 |

## 2. 可靠性（首周存活）

这些防止你的第一次 on-call 事故。

| 检查 | 通过标准 | 验证方式 |
|-------|--------------|---------------|
| 带退避的重试 | 5xx 时 3 次重试，指数延迟 | 中途 kill LLM mock —— 日志中可见重试 |
| Fallback model chain | 链中有 2+ 个模型 | 主模型不可用 —— 响应仍从 fallback 返回 |
| 请求超时 | 所有外部调用最大 30s | 慢 LLM mock（60s）—— 请求在 30s 超时 |
| 优雅降级 | 缓存/RAG 故障不拖垮服务 | 停止缓存 —— 请求仍成功（更慢、更贵） |
| 健康检查端点 | 返回依赖状态 | `GET /health` 返回 `{"status": "healthy", "cache": ..., "llm": ...}` |
| 流式传输工作 | 首 token 低于 500ms | 测量 time-to-first-token，持续 < 500ms |
| 错误消息安全 | 内部错误绝不泄露给用户 | 强制 500 —— 用户看到通用错误，而非堆栈跟踪 |

## 3. 成本控制（首月经济）

这些防止 $50K 的意外账单。

| 检查 | 通过标准 | 验证方式 |
|-------|--------------|---------------|
| 单请求成本追踪 | 每个请求记录 token 数 + USD 成本 | 请求日志有 `input_tokens`、`output_tokens`、`cost_usd` 字段 |
| Semantic cache 启用 | 重复模式命中率 > 20% | 1000 个测试请求后缓存统计显示命中率 |
| 缓存 TTL 配置 | 条目过期（默认：1 小时） | 插入条目 —— TTL 后不再返回 |
| 单用户成本追踪 | 按 user_id 聚合成本 | 仪表盘/API 显示按成本排序的前 10 用户 |
| 成本告警 | 每日预算 80% 时告警 | 设置 $10 日预算，发送 $8.50 请求 —— 告警触发 |
| 按成本路由模型 | 低复杂度查询使用更便宜模型 | 简单问题路由到 gpt-4o-mini，复杂问题到 gpt-4o |
| 最大 output tokens 设置 | 响应按 template 上限截断 | max_output_tokens=512 的 template —— 响应永不超过它 |

**成本估算公式：**
```
Monthly LLM cost = DAU x queries_per_user x 30 x (1 - cache_hit_rate) x (avg_input_tokens x input_price + avg_output_tokens x output_price) / 1,000,000
```

**按规模的基准阈值：**

| DAU | 目标单请求成本 | 月度预算 |
|-----|-------------------|----------------|
| 1K | < $0.005 | < $750 |
| 10K | < $0.003 | < $4,500 |
| 100K | < $0.001 | < $15,000 |

## 4. 可观测性（生产环境调试）

你无法修复你看不见的东西。

| 检查 | 通过标准 | 验证方式 |
|-------|--------------|---------------|
| 结构化 JSON 日志 | 每个请求产生一行 JSON 日志 | 日志包含：request_id、user_id、model、tokens、latency_ms、cost |
| 请求追踪 | 端到端追踪带组件耗时 | 单个请求显示：guardrail (5ms) + cache (2ms) + llm (3200ms) + eval (1ms) |
| 延迟追踪 | 测量 P50、P95、P99 | 1000 个请求后：P50 < 2s、P99 < 10s |
| 错误率监控 | 错误被计数和分类 | 仪表盘显示：0.5% API 错误、0.1% guardrail 拦截、0.01% 超时 |
| 缓存指标 | 命中率、未命中率、条目数可见 | `GET /v1/cache/stats` 返回当前数值 |
| A/B 测试指标 | 每变体质量指标被记录 | 每个请求记录 prompt_template + version 用于对比 |
| Eval 日志 | 每请求记录质量信号 | 存储响应长度、延迟、模型、template version 用于离线分析 |

## 5. Prompt 管理

Prompts 是代码。像对待代码一样对待它们。

| 检查 | 通过标准 | 验证方式 |
|-------|--------------|---------------|
| 版本化 templates | 每个 template 有名称 + 版本字符串 | Template 变更创建新版本，旧版本保留 |
| A/B 测试支持 | 按确定性用户哈希分流 | 同一用户在实验内始终看到相同变体 |
| 回滚能力 | < 1 分钟回退到上一版本 | 变更实验配置 —— 流量立即切换 |
| Template 验证 | 渲染前验证变量 | Template 中缺失变量引发清晰错误，而非 KeyError |
| System prompt 分离 | System 和 user 消息在独立字段中 | System prompt 不拼接进 user 消息 |

## 6. 扩展就绪

上线时不需要。10 倍增长时需要。

| 检查 | 通过标准 | 验证方式 |
|-------|--------------|---------------|
| 异步 LLM 调用 | API 调用不阻塞线程 | 50 并发请求 —— server CPU 保持 < 30% |
| 连接池 | HTTP 连接被复用 | 网络追踪显示与 LLM 提供商的持久连接 |
| 水平扩展 | 无状态 server 设计 | 负载均衡后 2 个实例 —— 所有请求成功 |
| 队列支持 | 非实时任务进入队列 | 摘要请求返回 job_id，结果通过轮询获取 |
| 负载测试通过 | 100 并发用户，< 5% 错误率 | `wrk` 或 `locust` 测试在目标并发下通过 |

## 新项目的实现顺序

1. **第 1 天：** API server + prompt templates + 单次 LLM 调用带重试
2. **第 2 天：** Input guardrails + output guardrails + 错误处理
3. **第 3 天：** Semantic cache + 单请求成本追踪
4. **第 4 天：** 流式传输（SSE）+ 健康检查端点
5. **第 5 天：** 结构化日志 + 请求追踪 + eval 日志
6. **第 2 周：** A/B 测试 + prompt 版本管理 + 回滚
7. **第 3 周：** Fallback model chain + 优雅降级
8. **第 4 周：** 负载测试 + 异步优化 + 水平扩展

## 快速诊断

如果生产环境出问题，按此顺序检查：

1. **用户抱怨错误？** 检查健康端点，然后日志中的错误率，然后 LLM 提供商状态页面
2. **响应慢？** 检查 P99 延迟，然后缓存命中率，然后追踪中的 LLM 响应时间
3. **成本飙升？** 检查单请求成本趋势，然后缓存命中率，然后按成本排序的 top 用户，然后查找增加 token 数的 prompt template 变更
4. **质量下降？** 检查是否部署了新 prompt 版本，检查 RAG 检索准确率是否变化，检查模型提供商是否变更了默认模型版本
5. **安全事件？** 检查 guardrail 拦截率（突然下降 = guardrails 被禁用），检查请求日志中的异常模式，立即轮换 API keys
