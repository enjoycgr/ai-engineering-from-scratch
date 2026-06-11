# Chaos Engineering for LLM Production

> 2026 年，面向 LLM 的混沌工程已自成一派。在生产环境运行实验前的先决条件：已定义 SLI/SLO、trace+metric+log 可观测性、自动回滚、运维手册、值班制度。架构包含四个平面：control（实验调度器）、target（服务、基础设施、数据存储）、safety（护栏 + 中止 + 流量过滤）、observability（指标 + 链路 + 日志）、feedback（反馈至 SLO 调整）。护栏是强制性的：burn-rate alert（燃尽速率告警）在每日错误预算燃尽速率 > 预期 2 倍时暂停实验；suppression windows（抑制窗口）+ trace-ID correlation（链路追踪 ID 关联）用于消除告警噪音。节奏：每周小规模金丝雀 + SLO 审查；每月 game day（演练日）+ 事后分析；每季度跨团队韧性审计 + 依赖映射。LLM 专用实验：内存过载、网络故障、提供商中断、畸形提示词、KV cache eviction storm（KV 缓存驱逐风暴）。工具：Harness Chaos Engineering（LLM 衍生推荐、blast-radius 自动缩小、MCP 工具集成）；LitmusChaos（CNCF）；Chaos Mesh（CNCF Kubernetes 原生）。

**类型：** Learn
**语言：** Python（stdlib，玩具级混沌实验运行器）
**前置条件：** Phase 17 · 23（AI SRE）、Phase 17 · 13（可观测性）
**时间：** ~60 分钟

## 学习目标

- 说出混沌工程的五个先决条件（SLI/SLO、可观测性、回滚、运维手册、值班），并解释跳过任何一个都会破坏实践。
- 绘制四个平面（control、target、safety、observability）及反馈至 SLO 的循环图。
- 列举五个 LLM 专用实验（内存过载、网络故障、提供商中断、畸形提示词、KV 驱逐风暴）。
- 根据技术栈选择工具 —— Harness、LitmusChaos、Chaos Mesh。

## 问题

传统技术栈中的混沌测试已成熟。LLM 技术栈增加了新的故障模式。一个包含毒字符的 4K token 提示词会让 tokenizer 卡住 12 秒。上游提供商返回 429；你的网关重试；你的服务因重试放大的并发而 OOM。突发负载下的 KV cache eviction storm 导致 re-prefill 级联，使计算饱和。

这些都不会出现在单元测试中。混沌工程让你在用户发现之前发现它们。

## 概念

### 先决条件

在生产环境运行混沌工程前，必须具备：

1. **SLI/SLO** —— 已定义的服务等级指标和目标。
2. **可观测性** —— trace、metric、log，已接入仪表盘。
3. **自动回滚** —— Phase 17 · 20 的策略开关回滚。
4. **运维手册** —— 结构化的，Phase 17 · 23。
5. **值班** —— 有人响应。

缺少任何一个，混沌都会变成真实事件。

### 四个平面 + 反馈

**Control plane（控制平面）** —— 实验调度器（Litmus workflow、Chaos Mesh schedule、Harness UI）。

**Target plane（目标平面）** —— 服务、Pod、节点、负载均衡器、数据存储。

**Safety plane（安全平面）** —— 终止开关、抑制窗口、blast-radius（影响范围）限制、error-budget（错误预算）关卡。

**Observability plane（可观测性平面）** —— 常规指标 + trace-ID 关联，以区分混沌诱导故障与自然故障。

**Feedback loop（反馈循环）** —— 发现反馈至 SLO 调整、运维手册更新、代码修复。

### 护栏是强制性的

- **Burn-rate alert（燃尽速率告警）**：如果每日错误预算燃尽速率超过预期的 2 倍，暂停实验。
- **Suppression windows（抑制窗口）**：实验期间在影响范围内静默非实验告警。
- **Trace-ID correlation（链路追踪 ID 关联）**：所有实验诱导的错误都携带标签，以便值班人员去重。

### 五个 LLM 专用实验

1. **内存过载** —— 通过发送高并发的长上下文请求强制 KV cache preemption storm（KV 缓存抢占风暴）。观察：服务是优雅降级还是崩溃？

2. **网络故障** —— 切断推理网关与提供商之间的连接。观察：fallback 是否在 SLA 内启动？（Phase 17 · 19）

3. **提供商中断模拟** —— OpenAI 100% 返回 429。观察：路由是否故障转移至 Anthropic？（Phase 17 · 16、19）

4. **畸形提示词** —— 注入使 tokenizer 停滞的负载（例如深度嵌套 unicode、超大 UTF-8 码点）。观察：单个请求是否会锁住一个 worker？

5. **KV 驱逐风暴** —— 通过饱和 vLLM block budget 强制驱逐。观察：LMCache 是否恢复，还是服务降级？

### 节奏

- **每周** —— 在 staging 进行小规模金丝雀实验，可能 5% 生产环境。
- **每月** —— 针对特定场景的 scheduled game day；跨团队参与；事后分析。
- **每季度** —— 跨团队韧性审计；依赖映射更新。

### 工具

- **Harness Chaos Engineering** —— 商业产品；AI 衍生实验推荐；blast-radius 自动缩小；MCP 工具集成。
- **LitmusChaos** —— CNCF 毕业项目；基于 Kubernetes workflow。
- **Chaos Mesh** —— CNCF 沙箱项目；Kubernetes 原生 CRD 风格。
- **Gremlin** —— 商业产品；广泛支持。
- **AWS FIS** / **Azure Chaos Studio** —— 托管云服务。

### 从小处开始

第一个实验：在稳定流量下 kill 一个 decode replica 的 pod。观察重路由和恢复。如果成功且看起来安全，升级到网络混沌。

第一个 LLM 专用实验：注入一个提供商 429 持续 5 分钟。观察 fallback。大多数团队会发现他们的 fallback 并未经过充分测试。

### 应该记住的数字

- 四个平面：control、target、safety、observability。
- Burn-rate 暂停：预期每日预算燃尽的 2 倍。
- 节奏：每周金丝雀、每月 game day、每季度审计。
- 五个 LLM 实验：内存、网络、提供商、畸形提示词、KV 风暴。

## 使用

`code/main.py` 模拟三个带安全平面关卡的混沌实验。报告哪些实验会触发 burn-rate 中止。

## 交付

本节课产出 `outputs/skill-chaos-plan.md`。给定技术栈和成熟度，选择前三个实验和工具。

## 练习

1. 运行 `code/main.py`。哪个实验触发了 burn-rate 关卡？为什么？
2. 为基于 vLLM 的 RAG 服务设计前五个混沌实验。包含成功标准。
3. 你的 burn-rate 告警暂停了一个实验。如何确定根本原因 —— 混沌还是自然故障？
4. 论证混沌是否应该在生产环境运行，还是仅在 staging。什么时候生产环境是正确的答案？
5. 说出三种通用网络混沌无法复现的 LLM 专用故障模式。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SLI / SLO | "服务目标" | 指标 + 目标；必需的先决条件 |
| Blast radius | "范围" | 实验影响的服务/用户集合 |
| Burn-rate alert | "预算关卡" | 错误预算燃尽速率 > 预期 2 倍时触发 |
| Game day | "月度演练" | 定期跨团队混沌演练 |
| LitmusChaos | "CNCF workflow" | CNCF 毕业项目 Kubernetes 混沌工具 |
| Chaos Mesh | "CNCF CRD" | CNCF 沙箱项目 Kubernetes 原生混沌工具 |
| Harness CE | "商业 AI 辅助" | 具备 AI 推荐的 Harness 混沌工程 |
| Malformed prompt | "tokenizer 炸弹" | 使 tokenization 停滞的输入 |
| KV eviction storm | "抢占级联" | 大规模驱逐触发 re-prefill |

## 延伸阅读

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
