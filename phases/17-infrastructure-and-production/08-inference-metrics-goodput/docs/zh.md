# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定一次推理部署是否正常工作。TTFT 是 prefill 加队列加网络。TPOT（等同于 ITL）是逐 token 的内存受限解码成本。端到端延迟是 TTFT 加上 TPOT 乘以输出长度。吞吐量是集群每秒处理的 token 总数。但对产品真正重要的是 goodput —— 即同时满足所有 SLO 的请求比例。高吞吐量但低 goodput 意味着你在处理那些永远无法及时送达用户的 token。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数据：TTFT 均值 162 ms，TPOT 均值 7.33 ms，端到端均值 1,093 ms。始终报告 P50、P90、P99 —— 不要只报均值。注意测量陷阱：GenAI-Perf 在 ITL 计算中排除 TTFT，而 LLMPerf 包含 TTFT；同一运行中两个工具对 TPOT 的结果不一致。

**类型:** 学习
**语言:** Python（标准库，玩具百分位计算器和 goodput 报告器）
**前置知识:** Phase 17 · 04（vLLM 服务内部原理）
**时间:** 约 60 分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、端到端、吞吐量和 goodput，并说出每个指标衡量的具体组成部分。
- 解释为什么均值是 LLM 服务的错误统计量，以及如何阅读 P50/P90/P99。
- 构建一个多约束 SLO（例如 TTFT<500 ms 且 TPOT<15 ms 且 端到端<2 s），并据此计算 goodput。
- 说出两个在同一运行中对 TPOT 结果不一致的基准测试工具，并解释原因。

## 问题背景

"我们的吞吐量是每秒 15,000 个 token。" 那又怎样？如果 40% 的请求端到端延迟超过 2 秒，用户就会放弃会话。仅凭吞吐量无法告诉你产品是否正常工作。

推理存在多个延迟维度，每个维度的失效方式各不相同。Prefill 是计算密集型的，随提示长度增加而增加。Decode 是内存密集型的，随 batch size 增加而增加。队列延迟是运营问题。网络延迟是物理距离问题。你需要为每个维度设置不同的指标，需要百分位数，还需要一个单一的复合指标来回答"用户是否得到了他们期望的结果"——这就是 goodput。

## 核心概念

### TTFT —— 首 token 时间

`TTFT = queue_time + network_request + prefill_time`

当提示较长时，prefill 占主导。在 H100 上运行 Llama-3.3-70B FP8 时，32k 提示的纯 prefill 时间约为 800 ms。队列时间是调度器在负载下的行为。网络请求是包含 TLS 的传输时间。TTFT 是用户在收到任何流式响应之前感受到的延迟。

### TPOT / ITL —— 逐 token 延迟

同一个量有多种名称。`TPOT`（time per output token，逐输出 token 时间）、`ITL`（inter-token latency，token 间延迟）、`decode latency per token`（逐 token 解码延迟）—— 都是同一个概念。它是首 token 之后连续流式 token 之间的时间间隔。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在相同的 Llama-3.3-70B H100 堆栈上使用 chunked prefill 时，TPOT 均值约为 7 ms。如果不使用 chunked prefill，当相邻序列进行长 prefill 时，TPOT 可能飙升至 50 ms。关注 P99，而非均值。

### 端到端延迟

`端到端 = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 token），端到端延迟由 TPOT 主导。对于短输出且提示较长时，端到端延迟由 TTFT 主导。报告按输出长度条件化的端到端延迟。

### 吞吐量

`吞吐量 = total_output_tokens / elapsed_time`

聚合指标。告诉你集群效率。无法告诉你单个请求的健康状况。

### Goodput —— 你真正关心的指标

`goodput = 同时满足 (TTFT <= a) 且 (TPOT <= b) 且 (端到端 <= c) 的请求比例`

SLO 是多约束的。只有当所有约束都满足时，请求才是"良好"的。Goodput 就是良好请求的比例。高吞吐量但 goodput 只有 60% 是失败的。较低吞吐量但 goodput 达到 99% 才是目标。

2026 年，goodput 被用于 MLPerf Inference v6.0 提交和 AI 平台提供商的内部 SLA 跟踪。

### 为什么均值是错误的统计量

LLM 延迟分布是右偏的。一个包含长 prefill 邻居的解码批次可能以 TPOT ~7 ms 输出 500 个 token，而以 TPOT ~60 ms 输出 20 个 token。TPOT 均值是 9 ms。TPOT P99 是 65 ms。用户经常遇到 P99 —— 这就是他们离开的原因。

始终报告三元组 (P50, P90, P99)。对于用户体验，P99 才是你需要优化的。

### 参考数据 —— 2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct

- TTFT 均值：162 ms
- TPOT 均值：7.33 ms
- 端到端均值：1,093 ms
- TPOT P99：根据 chunked-prefill 配置，在 10-25 ms 之间变化

这些是 NVIDIA 发布的参考点。它们随模型大小（70B 会显示 3-5 倍）、硬件（H100 与 B200 约 3 倍）和负载而变化。

### 测量陷阱

两个最常用的 2026 年基准测试工具在同一运行中对 TPOT 结果不一致：

- **NVIDIA GenAI-Perf**：在 ITL 计算中排除 TTFT。ITL 从第 2 个 token 开始。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始。

对于一个 TTFT 为 500 ms、100 个输出 token 在 700 ms 内完成解码的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择会改变数字。

始终说明使用哪个工具。始终公布定义。

### 构建 SLO

2026 年一个合理的面向消费者的 70B 聊天模型 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- 对于 <300 token 的输出，端到端 P99 <= 3 s。
- Goodput 目标 >= 99%。

企业级 SLO 会收紧 TTFT（200-400 ms）并放宽端到端。关键是把它们写下来，测量三个指标，并将 goodput 作为单一复合指标跟踪。

### 如何测量

- 运行真实流量或真实合成流量（LLMPerf 使用 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准测试运行目标为 2 倍峰值并发。
- 运行 30-50 次迭代，对合并样本取百分位数。
- 发布时注明工具名称、工具版本、模型、硬件、并发数、提示分布。

## 动手实践

`code/main.py` 是一个玩具 goodput 计算器。生成合成延迟分布，应用 SLO，并计算 goodput。同时展示同一 trace 上 GenAI-Perf 与 LLMPerf 的 TPOT 差异。

## 交付成果

本课产出 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，它生成一个基于 goodput 而非吞吐量来控制部署的 CI/CD 就绪基准测试方案。

## 练习

1. 运行 `code/main.py`。生成一个带有 1% 尾部尖峰的分布。当你将 TPOT P99 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 某厂商报价"Llama 3.3 70B H100 上 15,000 tok/s"。在相信之前，先问三个问题。
3. 为什么 chunked prefill 保护 TPOT P99 但不保护 TPOT 均值？
4. 为语音助手构建一个消费者 SLO（首 token 是被听到的，不是被读到的）。哪个指标对用户最可见？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出两个工具在其他三个指标上的分歧。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| TTFT | "首 token 时间" | 队列 + 网络 + prefill；长提示下由 prefill 主导 |
| TPOT | "逐输出 token 时间" | 首 token 之后的内存受限逐 token 解码成本 |
| ITL | "token 间延迟" | 大多数工具中与 TPOT 相同（并非全部 —— 见 GenAI-Perf） |
| 端到端 | "端到端" | TTFT + TPOT * 输出长度；再加上响应侧网络 |
| 吞吐量 | "tok/s" | 集群效率；没有延迟百分位数就毫无意义 |
| Goodput | "SLO 达标率" | 同时满足所有 SLO 约束的请求比例 |
| P99 | "尾部" | 1/100 的最坏情况延迟；用户体验指标 |
| 多约束 SLO | "联合约束" | 三个延迟边界的 AND；任一违反则请求失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具对 ITL 是否包含 TTFT 存在分歧 |

## 延伸阅读

- [NVIDIA NIM — LLM 基准测试指标](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义。
- [Anyscale — LLM 服务基准测试指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量方案。
- [BentoML — LLM 推理指标](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 真实部署上的应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准测试工具。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业公认的基于 goodput 的基准测试。
