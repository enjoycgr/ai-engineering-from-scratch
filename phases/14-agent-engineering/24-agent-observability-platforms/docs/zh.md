# Agent Observability: Langfuse, Phoenix, Opik

> 三款开源 agent observability platforms（智能体可观测性平台）主导 2026 年。Langfuse（MIT）—— 每月 600 万+ SDK 安装量，tracing（追踪）+ prompt management（提示词管理）+ evals（评估）+ session replay（会话回放）。Arize Phoenix（Elastic 2.0）—— 深度 agent-specific evals（智能体专属评估），RAG relevancy（RAG 相关性），OpenInference auto-instrumentation（自动仪器）。Comet Opik（Apache 2.0）—— automated prompt optimization（自动提示词优化），guardrails（护栏），LLM-judge hallucination detection（幻觉检测）。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 分钟

## Learning Objectives

- 命名三款顶级开源 agent observability platforms（智能体可观测性平台）及其 licenses（许可证）。
- 区分每款最强之处：Langfuse（prompt mgmt + sessions 提示词管理 + 会话），Phoenix（RAG + auto-instrumentation 自动仪器），Opik（optimization + guardrails 优化 + 护栏）。
- 解释为什么 89% 的组织在 2026 年前报告已部署 agent observability（智能体可观测性）。
- 实现一个 stdlib trace-to-dashboard pipeline（追踪到仪表盘流水线）with LLM-judge evaluation（LLM 评委评估）。

## The Problem

OTel GenAI（Lesson 23）为你提供 schema。你仍然需要平台来摄取 spans（跨度）、运行 evaluations（评估）、存储 prompt versions（提示词版本）和 surface regressions（发现回退）。三款竞争者各自强调生命周期中的不同部分。

## The Concept

### Langfuse（MIT）

- 每月 600 万+ SDK 安装量，GitHub 19k+ stars。
- 功能：tracing（追踪）、带 versioning + playground 的 prompt management（提示词管理）、evaluations（评估）（LLM-as-judge、用户反馈、自定义）、session replays（会话回放）。
- 2025 年 6 月：原先商业模块（LLM-as-a-judge、annotation queues 标注队列、prompt experiments 提示词实验、Playground）在 MIT 下开源。
- 最强之处：端到端可观测性 with tight prompt-management loop（紧密的提示词管理闭环）。

### Arize Phoenix（Elastic License 2.0）

- 更深度的 agent-specific evaluation（智能体专属评估）：trace clustering（追踪聚类）、anomaly detection（异常检测）、RAG retrieval relevancy（RAG 检索相关性）。
- 原生 OpenInference auto-instrumentation（自动仪器）。
- 与托管 Arize AX 配对用于生产环境。
- 无 prompt versioning（提示词版本控制）—— 定位为 alongside broader platforms（ alongside 更广泛平台）的 drift/behavioral-regression tool（漂移/行为回退工具）。
- 最强之处：RAG relevancy（RAG 相关性）、behavioral drift（行为漂移）、anomaly detection（异常检测）。

### Comet Opik（Apache 2.0）

- 通过 A/B experiments（A/B 实验）进行 automated prompt optimization（自动提示词优化）。
- Guardrails（护栏）（PII redaction PII 脱敏、topical constraints 主题约束）。
- LLM-judge hallucination detection（幻觉检测）。
- Comet 自己的测量 benchmark：Opik 在 23.44s 内完成 logging + evals，而 Langfuse 需要 327.15s（约 14 倍差距）—— 将 vendor benchmarks 视为 directional（方向性参考）。
- 最强之处：optimization loop（优化闭环）、automated experimentation（自动实验）、guardrail enforcement（护栏执行）。

### 行业数据

根据 Maxim（2026 年现场分析）：89% 的组织已部署 agent observability（智能体可观测性）；质量问题是最主要的生产障碍（32% 的受访者提及）。

### 如何选择

| 需求 | 选择 |
|------|------|
| 带 prompt management（提示词管理）的一体化方案 | Langfuse |
| 深度 RAG evaluation + drift（RAG 评估 + 漂移） | Phoenix |
| Automated optimization + guardrails（自动优化 + 护栏） | Opik |
| 开放许可，无 ELv2 | Langfuse (MIT) 或 Opik (Apache 2.0) |
| Datadog / New Relic 集成 | 任何 —— 都导出 OTel |

### 该模式在何处出错

- **无 eval strategy（评估策略）。** Tracing without evaluation（无评估的追踪）只是昂贵的日志。
- **自研 LLM-judge 无 grounding（依据）。** CRITIC pattern（Lesson 05）适用 —— judges 需要 external tools（外部工具）进行 factual verification（事实验证）。
- **Prompt versions 未绑定到 traces。** 当生产回退时，你无法 bisect（二分）到导致问题的 prompt。

## Build It

`code/main.py` 实现一个 stdlib trace collector + LLM-judge evaluator：

- 摄取 GenAI-shaped spans（符合 GenAI 规范的跨度）。
- 按 session（会话）分组，标记 failed runs（失败运行）（guardrail trips 护栏触发、low-confidence evals 低置信度评估）。
- 一个 scripted LLM-judge（脚本化 LLM 评委），按 rubric（评分标准）为 agent responses（智能体响应）打分。
- Dashboard-like summary（类仪表盘摘要）：failure rate（失败率）、top failure reasons（主要失败原因）、eval score distribution（评估分数分布）。

运行方式：

```
python3 code/main.py
```

输出：per-session eval scores（每会话评估分数）和 failure categorization（失败分类），匹配 Langfuse/Phoenix/Opik 会显示的内容。

## Use It

- **Langfuse** 自托管或云端；通过 OTel 或其 SDK 接入。
- **Arize Phoenix** 自托管；auto-instrument OpenInference。
- **Comet Opik** 自托管或云端；automated optimization loop（自动优化闭环）。
- **Datadog LLM Observability** 用于已运行 Datadog 的混合 ops+ML 团队。

## Ship It

`outputs/skill-obs-platform-wiring.md` 选择一款平台并将 traces + evals + prompt versions 接入现有 agent。

## Exercises

1. 将一周的 OTel traces 导出到 Langfuse 云端（免费层）。哪些 sessions（会话）失败了？为什么？
2. 为你的领域编写 LLM-judge rubric（评分标准）（factual correctness 事实正确性、tone 语气、scope adherence 范围遵守）。在 50 条 traces 上测试。
3. 对比 Langfuse prompt versioning 与 Phoenix trace clustering。哪个更快告诉你问题所在？
4. 阅读 Opik guardrail docs。将 PII redaction guardrail（PII 脱敏护栏）接入你的某次 agent run。
5. 在你的语料库上 benchmark 三款。忽略 vendor-published numbers（厂商发布数字）；测量你自己的。

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Tracing | "Spans collector（跨度收集器）" | 摄取 OTel / SDK spans；按 session 索引 |
| Prompt management | "Prompt CMS（提示词 CMS）" | 与 traces 绑定的 versioned prompts（版本化提示词） |
| LLM-as-judge | "Automated eval（自动评估）" | 单独的 LLM 按 rubric（评分标准）为 agent output 打分 |
| Session replay | "Trace playback（追踪回放）" | 逐步走过 past runs（过往运行）用于调试 |
| RAG relevancy | "Retrieval quality（检索质量）" | 检索到的 context 是否与 query 匹配 |
| Trace clustering | "Behavioral grouping（行为分组）" | 聚类相似运行以进行 drift detection（漂移检测） |
| Guardrail enforcement | "Policy at log time（日志时策略）" | 对记录内容的 PII/toxicity/scope 检查 |

## Further Reading

- [Langfuse docs](https://langfuse.com/) —— tracing、evals、prompt mgmt（提示词管理）
- [Arize Phoenix docs](https://docs.arize.com/phoenix) —— auto-instrumentation（自动仪器）、drift（漂移）
- [Comet Opik](https://www.comet.com/site/products/opik/) —— optimization + guardrails（优化 + 护栏）
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 三款平台都消费的 schema
