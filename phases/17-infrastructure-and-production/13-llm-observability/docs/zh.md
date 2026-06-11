# LLM 可观测性栈选择

> 2026 年的可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示管理、会话回放捆绑在一起。网关/遥测工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 是 MIT 许可核心，具有强大的开源平衡（云端每月 50K 事件免费）。Phoenix 是 Elastic License 2.0 下的 OpenTelemetry 原生 —— 非常适合漂移/RAG 可视化，但不是持久生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比单体可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 领域领先，$39/用户/月，仅 Enterprise 可自托管。Helicone 是基于代理的，15-30 分钟设置，每月 100K 请求免费，但在智能体追踪上深度较浅。常见生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens）通过 OpenTelemetry 粘合。

**类型:** 学习
**语言:** Python（标准库，玩具追踪采样模拟器）
**前置知识:** Phase 17 · 08（推理指标），Phase 14（智能体工程）
**时间:** 约 60 分钟

## 学习目标

- 区分开发平台（捆绑：评估 + 提示 + 会话）与网关/遥测工具（仅追踪 + 指标）。
- 将六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到其许可、定价和最佳用例。
- 解释 OpenTelemetry 粘合模式，让你可以将网关工具与单独的评估平台组合。
- 说出 2026 年成本差异化因素（Arize AX 的零拷贝方法 vs 单体摄取）并陈述约 100 倍倍数。

## 问题背景

你发布了一个 LLM 功能。它工作了。你无法看到提示失败、工具循环、延迟回归、成本飙升或提示缓存命中率。你 Google "LLM 可观测性"，得到八个工具，都声称以三个不同价格点解决同一个问题。

它们没有解决同一个问题。LangSmith 回答"为什么这个 LangGraph 运行失败了？"Phoenix 回答"我的 RAG 流水线是否在漂移？"Helicone 回答"哪个应用在烧 token？"Langfuse 回答"我能自托管整个东西吗？"不同工具，不同受众。

选择涉及四个维度：栈（LangChain？原始 SDK？多供应商？）、许可容忍度（仅 MIT？Elastic 可接受？商业可接受？）、预算（免费层？$100/月？$1000/月？）和自托管（必须？nice-to-have？绝不？）。

## 核心概念

### 两个类别

**开发平台** 将可观测性与评估、提示管理、数据集版本控制、会话回放捆绑在一起。你运行实验，看哪个提示有效，将新提示与旧赢家进行数据集回归。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具** 检测推理调用 —— 提示、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简。可通过 OpenTelemetry 与单独的评估工具组合。

### Langfuse —— 开源平衡

- 核心 Apache / MIT 许可；通过 Docker 自托管。
- 云端免费层：每月 50K 事件。付费：团队 $29/月。
- 评估、提示管理、追踪、数据集。合理覆盖所有四个开发平台功能。
- 最佳场景：你想要 LangSmith 级功能但必须自托管或保持开源许可。

### Phoenix（Arize）—— 遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管 trivial。
- 擅长 RAG 和漂移可视化。嵌入空间散点图作为一等功能提供。
- 不是为持久生产后端设计的 —— 主要是开发时可观测性。
- 最佳场景：RAG 流水线开发、漂移调试，与单独网关配对用于生产。

### Arize AX —— 规模化方案

- 商业。通过 Iceberg/Parquet 零拷贝数据湖集成。
- 声称比单体可观测性（Datadog 级）规模化时约便宜 100 倍。计算方式：你将追踪存储在自己的 S3 Parquet 上；Arize 直接读取。
- 最佳场景：>10M 追踪/天，现有数据湖，想要 LLM 专用仪表板而不支付 Datadog 价格。

### LangSmith —— LangChain/LangGraph 优先

- 商业，$39/用户/月。仅 Enterprise 可自托管。
- 对于 LangChain 和 LangGraph 栈是最佳类别。如果你不在两者上，吸引力较低。
- 最佳场景：团队致力于 LangChain，愿意付费。

### Helicone —— 基于代理的最小可行方案

- 通过将你的 `OPENAI_API_BASE` 交换为 Helicone 代理，15-30 分钟设置。
- MIT 许可；每月 100K 请求免费，付费 $20/月+。
- 包括故障转移、缓存、速率限制 —— 也充当网关。
- 在智能体/多步追踪上深度较浅。
- 最佳场景：快速启动，单栈应用，需要网关 + 可观测性合一。

### Opik（Comet）—— 开源开发平台

- Apache 2.0，完全开源。
- 与 Langfuse 类似的功能集，具有 Comet 血统。
- 最佳场景：已在 Comet 上的 ML 团队，想要在同一面板中获得 LLM 可观测性。

### SigNoz —— OpenTelemetry 优先的全栈 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 加 LLM。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年底发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。正在出现的生产模式：

1. 从每个 LLM 调用发出带 GenAI 约定的 OTel。
2. 路由到网关（Helicone / Portkey）用于日常。
3. 双发到评估平台（Phoenix / Langfuse）用于回归。
4. 归档到数据湖（Iceberg）用于通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误层插桩

在你的智能体框架内部插桩（例如，添加 LangSmith 追踪）会将你与该框架耦合。在 HTTP/OpenAI-SDK 层（通过 OpenLLMetry 或你的网关）插桩是可移植的。

### 采样 —— 你无法保留一切

在 >1M 请求/天时，完整追踪保留成本超过 LLM 调用本身。按规则采样：100% 错误，100% 高成本，5% 成功。始终保留聚合；为长尾保留原始。

### 你应该记住的数字

- Langfuse 云端免费：每月 50K 事件。
- LangSmith：$39/用户/月。
- Helicone 免费：每月 100K 请求。
- Arize AX 声称：规模化时比单体约便宜 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

## 动手实践

`code/main.py` 模拟 1M 追踪天跨保留策略（100% 摄取、采样、采样 + 错误）。报告存储成本和每种策略下丢失的内容。

## 交付成果

本课产出 `outputs/skill-observability-stack.md`。给定栈、规模、预算、许可姿态，选择工具。

## 练习

1. 你的 LangChain 团队想要开源自托管可观测性。选择 Langfuse 或 Opik 并论证。
2. 在 5M 追踪/天且 Datadog 报价 $150K/月时，计算 Arize AX 的盈亏平衡。
3. 设计你的组织指南应在每次 LLM 调用上强制要求的 OpenTelemetry GenAI 属性集。
4. 论证 Phoenix 单独是否足以用于生产。何时不足？
5. Helicone 有 20ms 代理开销。在 P99 TTFT 300 ms 时，这是否可接受？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| OpenLLMetry | "LLM 的 OTel" | LLM 的开源 OpenTelemetry 插桩 |
| GenAI 约定 | "OTel 属性" | LLM 调用的标准 OTel 属性名 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态系统捆绑的商业平台 |
| Langfuse | "开源 LangSmith" | 具有类似功能集的 MIT 开源 |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生开发/评估平台 |
| Arize AX | "规模化可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | 收集 LLM 遥测 + 网关功能的 HTTP 代理 |
| Opik | "Comet LLM" | Comet 的 Apache 2.0 开源开发平台 |
| 会话回放 | "追踪重放" | 重放完整智能体会话及工具调用 |
| 评估 | "离线测试" | 在标注数据集上运行候选模型/提示 |

## 延伸阅读

- [SigNoz — 2026 年顶级 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 设置 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)
