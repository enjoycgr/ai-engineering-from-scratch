# Load Testing LLM APIs — 为什么 k6 和 Locust 会撒谎

> 传统的负载测试工具并非为流式响应、可变输出长度、token 级指标或 GPU 饱和场景而设计。有两个陷阱最常让团队中招。GIL 陷阱：Locust 的 token 级测量在客户端 Python GIL 下运行 tokenization，在高并发下与请求生成竞争；tokenization 积压导致报告的 inter-token latency（token 间延迟）膨胀——瓶颈在客户端，而非服务器。Prompt-uniformity 陷阱（提示词均一性陷阱）：循环测试中使用完全相同的提示词，只测试了 token 分布上的一个点；真实流量具有可变长度和多样的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 修复了这个问题。2026 年工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——支持流式感知、通过 TestRun/PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试，最适合 CI/CD 门禁；Vegeta 用于 Go 恒定速率饱和测试；Locust 2.43.3 仅在与 LLM-Locust 扩展配合时使用。负载模式：steady-state（稳态）、ramp（渐进）、spike（尖峰）、soak（浸泡）。

**类型：** Build
**语言：** Python（stdlib，玩具级真实提示词生成器 + 延迟采集器）
**前置条件：** Phase 17 · 08（推理指标），Phase 17 · 03（GPU 自动扩缩容）
**时间：** ~75 分钟

## 学习目标

- 解释两种反模式（GIL 陷阱、提示词均一性陷阱），它们会让通用负载测试工具在测试 LLM API 时给出虚假结果。
- 为给定目的选择工具：LLMPerf（基准测试）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成测试）、GenAI-Perf（NVIDIA 参考测试）。
- 设计四种负载模式（steady、ramp、spike、soak），并说出每种模式能捕获的故障模式。
- 使用输入 token 的 mean + stddev 构建真实的提示词分布，而非固定长度。

## 问题

你用 k6 对 LLM 端点做了 500 并发用户的测试。它扛住了。你上线了。生产环境 200 个真实用户时服务就挂了——P99 TTFT 爆炸，GPU 满载。

发生了两件事。第一，k6 发送了 500 条完全相同的提示词——你的请求合并（request-coalescing）和前缀缓存（prefix caching）让你看起来在处理 500 个并发 decode，实际上你只处理了一个。第二，k6 不会按人眼体验的方式追踪流式响应的 inter-token latency；它只看到一条 HTTP 连接，而不是 500 个 token 以不同间隔到达。

LLM 的负载测试是一门独立学科。

## 概念

### GIL 陷阱（Locust）

Locust 使用 Python，在客户端 GIL 下运行 tokenization。高并发下，tokenizer 会排在请求生成后面。报告的 inter-token latency 包含了客户端 tokenization 的积压。你以为服务器慢，其实是测试工具的问题。

修复方案：LLM-Locust 扩展将 tokenization 移到独立进程，或使用编译语言编写的测试工具（k6、LLMPerf 使用 tokenizers.rs）。

### 提示词均一性陷阱

所有已知的负载测试工具都允许你配置一条提示词。在 10,000 次循环测试中，每次都发送完全相同的提示词。服务器每次看到相同的前缀——prefix cache hit 接近 100%，吞吐量看起来很好。

修复方案：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——长度多样，内容多样。

### 四种负载模式

1. **Steady-state（稳态）**——恒定 RPS 持续 30-60 分钟。捕获：基线性能回归。
2. **Ramp（渐进）**——在 15 分钟内将 RPS 从 0 线性提升到目标值。捕获：容量断点、预热异常。
3. **Spike（尖峰）**——突然提升到 3-10x RPS 持续 2 分钟后恢复。捕获：autoscaling（自动扩缩容）延迟、队列饱和、cold start（冷启动）影响。
4. **Soak（浸泡）**——稳态持续 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性数据溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）——Python 前端，Rust 后端 tokenization。支持 mean/stddev 提示词。支持流式感知。性能测试的默认最佳工具。

**NVIDIA GenAI-Perf**——NVIDIA 的参考工具。使用 Triton 客户端；指标覆盖全面。注意它的 ITL 排除了 TTFT；LLMPerf 的 ITL 包含 TTFT。同一台服务器用两种工具会得出不同的 TPOT。

**LLM-Locust**（TrueFoundry）——修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm**——大规模合成基准测试工具。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go，编译型，无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRD 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门禁和 SLA 测试。

**Vegeta**——Go 编写，比 k6 更简单。恒定速率 HTTP 饱和测试。不是 LLM 专用工具，但适合网关 / 限流测试。

**Locust 2.43.3 原版**——对 LLM 存在 GIL 陷阱。仅在与 LLM-Locust 扩展配合时使用。

### CI 中的 SLA 门禁

在 PR 上运行 k6：

- 在基线 RPS 下各运行 30-50 次迭代。
- 门禁条件：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 违反即中断构建。

### 真实提示词分布

从真实流量样本构建（如果有的话）或从已发布分布构建（例如聊天用 ShareGPT 提示词，代码用 HumanEval）。将 mean + stddev 喂给 LLMPerf。绝对避免 loop-with-one-prompt。

### 应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：支持流式感知指标。
- 典型 LLMPerf 运行：在并发 X 下运行 100-1000 个请求。
- 典型 CI 门禁：每个 PR 30-50 次迭代。
- 四种模式：steady、ramp、spike、soak。

## 使用

`code/main.py` 模拟了一个带真实提示词分布的负载测试，测量有效 TPOT，并演示了统一提示词陷阱。

## 交付

本节课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。比较 uniform vs realistic 分布——差距在哪里？
2. 编写 k6 CI 门禁脚本：TTFT P95 < 800 ms，100 并发，运行 5 分钟。
3. 你的 soak 测试显示内存每小时增长 50 MB。说出三种原因以及用来区分它们的观测手段。
4. Spike 测试从 10 RPS 到 100 RPS。如果使用了 Karpenter + vLLM production-stack（Phase 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；同一台服务器 LLMPerf 报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| LLMPerf | "LLM 测试工具" | Anyscale 基准测试工具，支持流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考测试工具 |
| LLM-Locust | "Locust for LLMs" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准测试" | 大规模合成工具 |
| k6 Operator | "K8s k6" | 基于 CRD 的分布式 k6 |
| GIL trap | "Python 客户端开销" | Tokenization 积压导致报告的延迟膨胀 |
| Prompt-uniformity trap | "单提示词谎言" | 循环使用相同提示词命中缓存，夸大吞吐量 |
| Steady-state | "恒定负载" | N 分钟内平坦的 RPS |
| Ramp | "线性上升" | 在持续时间内从 0 到目标值 |
| Spike | "突发测试" | 突然倍增然后恢复 |
| Soak | "长时间测试" | 数小时，用于检测泄漏 |

## 延伸阅读

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
