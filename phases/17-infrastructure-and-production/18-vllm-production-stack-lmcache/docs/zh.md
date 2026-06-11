# vLLM Production Stack with LMCache KV Offloading

> vLLM 的 production-stack（生产栈）是 Kubernetes 部署的参考实现——router（路由器）、engines（引擎）和 observability（可观测性）全部集成在一起。LMCache 是 KV-offloading（KV 卸载）层，它将 KV cache（KV 缓存）从 GPU 内存中提取出来，在查询和引擎之间复用（先存入 CPU DRAM，再下沉到 disk/Ceph）。vLLM 0.11.0 的 KV Offloading Connector（2026 年 1 月）通过 Connector API（v0.9.0+）实现了异步和可插拔。Offload latency（卸载延迟）对用户不可见。即使没有共享前缀，LMCache 也有价值——当 GPU 的 KV slot 耗尽时，被抢占的请求可以从 CPU 恢复，而无需重新计算 prefill（预填充）。在 4 台 a3-highgpu-4g 上部署 16x H100（80GB HBM）的公开基准测试显示：当 KV cache 超过 HBM 时，native CPU offload（原生 CPU 卸载）和 LMCache 都能显著提升 throughput（吞吐量）；在 KV footprint 较低时，所有配置都与基线持平，仅有少量 overhead（开销）。

**Type:** Learn
**Languages:** Python（stdlib，toy KV-spill simulator）
**Prerequisites:** Phase 17 · 04（vLLM Serving Internals）, Phase 17 · 06（SGLang/RadixAttention）
**Time:** ~60 minutes

## Learning Objectives

- 画出 vLLM production-stack 的分层：router、engines、KV offload、observability。
- 解释 KV Offloading Connector API（v0.9.0+）以及 0.11.0 的 asynchronous（异步）路径如何隐藏 offload latency。
- 量化 LMCache CPU-DRAM 何时有帮助（KV > HBM）vs 何时增加 overhead（KV 小到足以放入 HBM）。
- 在给定部署约束下，选择 native vLLM CPU offload 还是 LMCache connector。

## The Problem

你的 vLLM serving 显示 GPU HBM 占用 100%，并发一上升就出现 preemption（抢占）事件。请求被驱逐、重新排队，同一 2K-token 提示在一分钟内被重新 prefill 四次。GPU 计算浪费在冗余 prefill 上；goodput（有效吞吐量）远低于 raw throughput（原始吞吐量）。

增加 GPU 成本线性增长。增加 HBM 不可能。但 CPU DRAM 很便宜——一个 socket 有 512 GB+，latency 比 HBM 差几个数量级，但对"临时温热"的 KV cache 来说足够。

LMCache 将 KV cache 提取到 CPU DRAM，使被抢占的请求快速恢复，且跨引擎的重复前缀无需每个引擎都重新 prefill。

## The Concept

### vLLM production-stack

`github.com/vllm-project/production-stack` 是参考 Kubernetes 部署：

- **Router** — cache-aware（Phase 17 · 11）。消费 KV 事件。
- **Engines** — vLLM workers。每张 GPU 或每个 TP/PP 组一个。
- **KV cache offload** — LMCache 部署或 native connector。
- **Observability** — Prometheus scrape、Grafana dashboards、OTel traces。
- **Control plane** — service discovery、config、rolling updates。

以 Helm chart + operator 形式交付。

### The KV Offloading Connector API（v0.9.0+）

vLLM 0.9.0 引入了 Connector API，用于可插拔的 KV cache backends。Engine 将 block 卸载到 connector；connector 存储它们（RAM、disk、object storage、LMCache）。请求需要 block 时，connector 加载回来。

vLLM 0.11.0（2026 年 1 月）增加了 asynchronous offload path——offload 可以在后台进行，因此 engine 在常见情况下不会阻塞。End-to-end latency 和 throughput 仍取决于 workload shape、KV cache hit rate 和 system pressure；vLLM 官方说明指出，custom-kernel offload 在低 hit rate 下可能降低 throughput，且 async scheduling 与 speculative decoding（投机解码）存在已知的交互问题。

### Native CPU offload vs LMCache

**Native vLLM CPU offload**：engine-local。将 KV block 存储在 host RAM 中。实现快，零网络跳数。不跨引擎。

**LMCache connector**：cluster-scale。将 block 存储在共享 LMCache server（CPU DRAM + Ceph/S3 tier）中。Block 可被任何引擎访问。16x H100 基准测试已发布。

单个 engine 有 HBM pressure 时选 native。多个引擎共享前缀（RAG 共用系统提示、多租户共享模板）时选 LMCache。

### Benchmark behavior

16x H100（80 GB HBM）分布在 4 台 a3-highgpu-4g 上的测试：

- Low KV footprint（短提示、低并发）：所有配置与基线持平，LMCache 增加约 3-5% overhead。
- Moderate footprint：LMCache 开始在跨引擎前缀复用上发挥作用。
- KV exceeds HBM：native CPU offload 和 LMCache 都大幅提升 throughput；LMCache 增益更大，因为跨引擎共享。

### When LMCache is decisive

- 多租户 serving，系统提示在租户间共享。
- RAG，文档块在查询间重复。
- 同一基座上的 fine-tuned variants（LoRA），基座模型 KV 复用减少冗余计算。
- Preemption-heavy 工作负载：从 CPU 恢复比重新 prefill 更便宜。

### When NOT to enable

- HBM pressure 小——付 overhead 却无收益。
- 短上下文（<1K token）——transfer time > re-prefill。
- 单租户单提示工作负载——无复用可捕获。

### Integration with disaggregated serving

Phase 17 · 17 的 disaggregated serving + LMCache 叠加：KV 从 prefill pool 传输到 decode pool 后若未使用，落入 LMCache；后续查询从 LMCache 拉取。Phase 17 · 11 的 cache-aware router 可以路由到本地或 LMCache 共享 cache 匹配的引擎。

### Numbers you should remember

- vLLM 0.9.0：Connector API 发布。
- vLLM 0.11.0（2026 年 1 月）：asynchronous offload path；end-to-end latency 影响取决于 workload、KV hit rate 和 system pressure（非绝对保证）。
- 16x H100 benchmark：KV footprint 超过 HBM 时 LMCache 有帮助。
- 小 HBM pressure：3-5% overhead，无收益。

## Use It

`code/main.py` 模拟有/无 LMCache 的 preemption-heavy 工作负载。报告 re-prefills avoided、throughput gain 和 break-even HBM utilization。

## Ship It

本课产出 `outputs/skill-vllm-stack-decider.md`。根据 workload shape 和 vLLM 部署，决定 native vs LMCache vs 两者都不选。

## Exercises

1. 运行 `code/main.py`。LMCache 在何种 HBM utilization 下开始划算？
2. 某租户在 200 查询/小时内共享 6K-token 系统提示。计算该租户的预期 LMCache 节省。
3. LMCache server 是单点故障。设计 HA 策略（replicas、回退 native）。
4. LMCache 存储到 Ceph 机械硬盘。4K-token KV 在 70B FP8 下为 500 MB，读取时间 vs re-prefill 是多少？
5. 论证 vLLM 0.11.0 的 asynchronous path 是否"免费"——overhead 藏在哪里？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Production-stack | "the reference deployment" | vLLM's Kubernetes Helm chart + operator |
| Connector API | "KV backend interface" | vLLM 0.9.0+ pluggable KV store interface |
| Native CPU offload | "engine-local spill" | Store KV in host RAM of same engine |
| LMCache | "cluster KV cache" | Cross-engine KV cache server on CPU DRAM + disk |
| 0.11.0 async | "non-blocking offload" | Offload hidden behind engine stream |
| Preemption | "evict to make room" | KV cache shuffle when HBM full |
| Prefix reuse | "same system prompt" | Multiple queries share beginning; cache hit |
| Ceph tier | "disk tier" | Durable storage below DRAM in the cache hierarchy |

## Further Reading

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — asynchronous path details.
