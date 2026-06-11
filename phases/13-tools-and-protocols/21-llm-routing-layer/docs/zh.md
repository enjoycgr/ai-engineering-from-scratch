# LLM 路由层 —— LiteLLM、OpenRouter、Portkey

> 提供商锁定很昂贵。不同的工具调用工作负载适合不同的模型。路由网关提供一个 API 表面、重试、故障转移、成本追踪和护栏。三种原型主导 2026：LiteLLM（开源自托管）、OpenRouter（托管 SaaS）、Portkey（生产级，2026 年 3 月开源）。本课命名决策标准并走过 stdlib 路由网关。

**类型：** Learn
**语言：** Python（stdlib，路由 + 故障转移 + 成本追踪器）
**前置要求：** Phase 13 · 02（函数调用），Phase 13 · 17（网关）
**时间：** ~45 分钟

## 学习目标

- 区分自托管、托管和生产级路由选项。
- 实现一个在提供商失败时按定义优先顺序重试的故障转移链。
- 跨提供商追踪每次请求成本和 token 使用。
- 针对给定生产约束决定 LiteLLM、OpenRouter 和 Portkey 之间选择。

## 问题

提供商路由重要的场景：

1. **成本。** Claude Sonnet 成本是 Haiku 的 3 倍。对于分类任务，Haiku 足够；对于综合任务，Sonnet 值得。按请求路由。

2. **故障转移。** OpenAI 有个糟糕小时。每次请求都失败。你想要无需重新部署的自动故障转移到 Anthropic。

3. **延迟。** 实时聊天 UI 需要快速首 token 时间。批量摘要器不需要。按延迟 SLA 路由。

4. **合规性。** 欧盟用户必须留在欧盟区域。按区域路由。

5. **实验。** 对相同工作负载 A/B 两个模型。按测试桶路由。

为每个集成手工编写所有这些都是重复的。路由网关提供一个 OpenAI 兼容的 API 并处理其余。

## 概念

### OpenAI 兼容代理形状

每个人都说 OpenAI 形状。路由网关暴露 `/v1/chat/completions`，接受 OpenAI schema，内部代理到 Anthropic / Gemini / Cohere / Ollama / 任何。客户端不关心。

### 模型别名

代替 `claude-3-5-sonnet-20251022`，你的代码说 `our_smart_model`。网关将别名映射到真实模型。当 Anthropic 交付 Claude 4 时，你在服务端更改别名；你的代码不触碰任何东西。

### 故障转移链

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

网关在配置中定义这个。重试计入预算，因此故障转移级联不会爆炸成本。

### 语义缓存

相同或近相同的提示命中缓存而非提供商。重复智能体循环上的节省可以是 30% 到 60%。键是基于嵌入的；近相同提示共享缓存槽。

### 护栏

网关级：

- **PII 编辑。** 在发送提示前进行基于正则表达式或 ML 的传递。
- **策略违规。** 拒绝带有禁止内容的提示。
- **输出过滤器。** 编辑补全中的泄漏。

Portkey 和 Kong 都交付了固执己见的护栏。LiteLLM 让它们可选。

### 每键速率限制

一个 API key = 一个团队。每键预算防止一个团队消耗共享配额。大多数网关支持这一点。

### 自托管 vs 托管的权衡

| 因素 | LiteLLM（自托管） | OpenRouter（托管） | Portkey（生产级） |
|--------|----------------------|----------------------|----------------------|
| 代码 | 开源，Python | 托管 SaaS | 开源（2026 年 3 月）+ 托管 |
| 设置 | 部署代理 | 注册 | 任一 |
| 提供商 | 100+ | 300+ | 100+ |
| 计费 | 你自己的 key | OpenRouter 积分 | 你自己的 key |
| 可观测性 | OpenTelemetry | 仪表板 | 完整 OTel + PII 编辑 |
| 最适合 | 想要完全控制的团队 | 快速原型设计 | 开箱即用的合规性 |

当你拥有 SRE 团队并想要数据主权时，LiteLLM 获胜。当你想要单一订阅且无需基础设施时，OpenRouter 获胜。当你需要开箱即用的护栏和合规性时，Portkey 获胜。

### 成本追踪

每次请求携带 `provider`、`model`、`input_tokens`、`output_tokens`。乘以网关维护的每模型每 token 价格（从定价表中提取）。每用户 / 每团队 / 每项目聚合。

### MCP 加路由

网关可以路由 LLM 调用 AND MCP 采样请求。当采样请求的 modelPreferences 偏好特定模型时，网关转换到正确的后端。这是 Phase 13 · 17（MCP 网关）和本课的路由网关有时合并为一个服务的地方。

### 路由策略

- **静态优先。** 列表中的第一个；错误时故障转移。
- **负载均衡。** 轮询或加权。
- **成本感知。** 选择满足延迟 / 质量的最便宜模型。
- **延迟感知。** 选择最近 N 分钟中最快的模型。
- **任务感知。** 提示分类器将编码路由到一个模型，摘要到另一个。

## 使用它

`code/main.py` 在约 150 行中实现了一个路由网关：接受 OpenAI 形状请求、转换到每提供商 stub、运行优先故障转移链、追踪每次请求成本并对输入应用 PII 编辑。用三个场景运行它：正常请求、触发故障转移的主提供商中断、PII 泄漏被编辑捕获。

看点：

- `ROUTES` dict：别名 -> 具体提供商的优先排序列表。
- 故障转移循环在 5xx 上重试。
- 成本追踪器将 token 使用乘以每模型费率。
- PII 编辑器在转发前编辑 SSN 形状模式。

## 交付它

本课产出 `outputs/skill-routing-config-designer.md`。给定工作负载配置文件（延迟、成本、合规性），该技能选择 LiteLLM / OpenRouter / Portkey 并产出路由配置。

## 练习

1. 运行 `code/main.py`。触发中断场景；确认故障转移到第二个提供商且成本正确归属。

2. 添加语义缓存：提示的 SHA256 是查找键；缓存命中即时返回。在重复调用上测量成本节省。

3. 添加一个提示分类器，将 "code ..." 提示路由到偏好智能的别名，将 "summarize ..." 提示路由到偏好速度的别名。

4. 设计每团队预算：每个团队有每月支出上限；网关在上限到达时拒绝请求。选择执行粒度（每次请求或窗口化）。

5. 并排阅读 LiteLLM、OpenRouter 和 Portkey 文档。命名每个交付而另外两个没有的一个功能。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| 路由网关 | "LLM 代理" | 许多提供商前面的单 API 表面层 |
| OpenAI 兼容 | "说 OpenAI schema" | 接受 `/v1/chat/completions` 形状，转换到任何后端 |
| 模型别名 | "our_smart_model" | 你的代码中的名称，网关映射到具体模型 |
| 故障转移链 | "重试列表" | 失败时尝试的提供商排序列表 |
| 语义缓存 | "提示嵌入缓存" | 键是提示的嵌入；近重复共享缓存命中 |
| 护栏 | "输入/输出过滤器" | 编辑 PII，拒绝策略违规 |
| 每键速率限制 | "团队预算" | 限定到 API key 的配额 |
| 成本追踪 | "每次请求支出" | 聚合 token 使用 x 每模型价格 |
| LiteLLM | "开放代理" | 可自托管的 OSS 路由网关 |
| OpenRouter | "托管 SaaS" | 基于积分计费的托管网关 |
| Portkey | "生产选项" | 内置护栏的开源 + 托管 |

## 延伸阅读

- [LiteLLM — docs](https://docs.litellm.ai/) —— 自托管路由网关
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) —— 托管路由 SaaS
- [Portkey — docs](https://portkey.ai/docs) —— 带护栏的生产路由
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) —— 决策指南
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) —— 供应商调查
