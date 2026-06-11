# Disaggregated Prefill/Decode（分离式预填充/解码）—— NVIDIA Dynamo 与 llm-d

> Prefill（预填充）是 compute-bound（计算密集型）；decode（解码）是 memory-bound（内存密集型）。两者运行在同一张 GPU 上会造成一种资源浪费。Disaggregation（分离式部署）将它们拆分到独立池，通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在池间传输 KV cache。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 自动匹配 prefill:decode 比例以满足 SLO。NVIDIA 公布的吞吐量提升大致如下：developer.nvidia.com（2025-06）显示在 GB200 NVL72 + Dynamo 上，DeepSeek-R1 MoE 在中等延迟区间提升约 6 倍；Dynamo 产品页（developer.nvidia.com，无日期）宣传 GB300 NVL72 + Dynamo 相比 Hopper 可达 50 倍 MoE 吞吐量。"30 倍"是社区对 Blackwell + Dynamo + DeepSeek-R1 全栈报告的汇总；我们未找到单一原始来源明确写出 30 倍，因此将其视为方向性说法。llm-d（Red Hat + AWS）是 Kubernetes-native：prefill / decode / router 作为独立 Service，带按角色 HPA。llm-d 0.5 增加了 hierarchical KV offloading（分层 KV 卸载）、cache-aware LoRA routing、UCCL 网络、scale-to-zero。经济性：多个客户披露的内部汇总显示，在 $200 万级别 inference（推理）支出中，切换到 Dynamo 的 disaggregated 部署并保持相同 SLA 可节省 30–40%（即每年 $60–80 万）；具体 $200 万→$60–80 万是内部合成数字，非单一公开案例——仅作为数量级参考，不作为引用来源。短提示（<512 token，短输出）不足以抵消传输成本。

**Type:** Learn
**Languages:** Python（stdlib，toy disaggregated-vs-colocated simulator）
**Prerequisites:** Phase 17 · 04（vLLM Serving Internals）, Phase 17 · 08（Inference Metrics）
**Time:** ~75 minutes

## Learning Objectives

- 解释为何 prefill 和 decode 需要不同的最优 GPU 配置，并量化 colocation（同机部署）下的资源浪费。
- 画出 disaggregated 架构图：prefill pool、decode pool、KV transfer via NIXL、router。
- 说出 disaggregation 不划算的条件（短提示、短输出）。
- 区分 NVIDIA Dynamo（stack-above）与 llm-d（Kubernetes-native），并匹配各自适用的运维场景。

## The Problem

你在 8 张 H100 上运行 Llama 3.3 70B。混合工作负载（长提示+短输出）下，GPU 在 decode 阶段空闲，因为大部分计算已消耗在 prefill。不同工作负载（短提示+长输出）下则相反。Colocated prefill + decode 意味着两种资源都过度配置。

预算影响：20-40% 的 GPU 时间浪费在错误资源上。你购买 H100 计算来跑 memory-bound decode，或购买 H100 HBM 带宽来跑 compute-bound prefill。两者都是昂贵的浪费。

Disaggregation 将 prefill 和 decode 拆分到按各自瓶颈优化的独立池。KV cache 通过高带宽互连从 prefill pool 传输到 decode pool。

## The Concept

### Why the bottlenecks differ

**Prefill** — 在完整输入提示上一次性运行 transformer。矩阵乘法占主导；compute-bound。H100 FP8 提供约 2000 TFLOPS 有效吞吐量。Batch efficiency 良好——一次前向处理多个 token。

**Decode** — 逐 token 生成，每次迭代读取完整权重。Memory-bandwidth-bound。HBM3 提供约 3 TB/s。Batch efficiency 仅在并发高时才好——权重读取在 batch 中摊销。

Colocating them：你购买同时优化两者的 GPU。H100 两者都擅长但成本相同。大规模时，你希望 prefill pool 用 H100/计算密集型；decode pool 用 H200/内存密集型，或 aggressive quantization（激进量化）。

### The architecture

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输层。可用时使用 RDMA/InfiniBand，否则回退 TCP。Transfer latency 真实存在——70B FP8 下 4K-token 提示的 KV cache 传输通常 20-80 ms。这就是短提示不值得 disaggregation 的原因：传输税超过节省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 位于 vLLM、SGLang、TRT-LLM 之上作为 orchestrator（编排器）。
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比例。
- Rust 核心，Python 扩展性。
- 吞吐量提升：NVIDIA 报告 GB200 NVL72 + Dynamo 上 DeepSeek-R1 MoE 中等延迟区间提升 6 倍（developer.nvidia.com，2025-06）；社区"高达 30 倍"的全栈 Blackwell + Dynamo + DeepSeek-R1 说法缺乏单一原始来源，应视为方向性。
- GB300 NVL72 + Dynamo：相比 Hopper 高达 50 倍 MoE 吞吐量（Dynamo 产品页，developer.nvidia.com，无日期）。

**llm-d**（Red Hat + AWS，Kubernetes-native）：
- Prefill / decode / router 作为独立 Kubernetes Services。
- 按角色 HPA，信号为 queue depth（prefill）/ KV utilization（decode）。
- `topologyConstraint packDomain: rack` 将 prefill+decode 组打包到同一机架，实现高带宽 KV transfer。
- llm-d 0.5（2026）：hierarchical KV offloading、cache-aware LoRA routing、UCCL 网络、scale-to-zero。

想要托管 stack-above orchestrator 选 Dynamo。想要 Kubernetes-native 原语且已投入 CNCF 生态选 llm-d。

### Economics

内部合成（非单一公开案例——数量级参考）：

- 每年 $200 万 colocated serving 支出。
- 切换到 Dynamo 的 disaggregated 部署。
- 相同请求量，相同 P99 延迟 SLA。
- 报告节省：每年 $60–80 万（30–40% 降幅）。
- 无新硬件。

该数字由多个客户披露合成，非单一可引用案例；最接近的公开数据点是 Baseten 的 Dynamo KV routing 实现 2 倍更快 TTFT / 61% 更高吞吐量（baseten.co，2025-10），以及 VAST + CoreWeave 在 40–60% KV hit rate 下预测 tokens/$ 提升 60–130%（vastdata.com，2025-12）。节省来自为每个池 right-sizing；prefill-heavy 工作负载（8K+ 前缀的 RAG）比均衡工作负载受益更多。

### When NOT to disaggregate

- 提示 < 512 token 且输出 < 200 token：传输税占主导。
- 小集群（< 4 GPU）：池多样性不足。
- 团队无法运维两个带按角色扩缩容的 GPU 池：Dynamo 有帮助但非 trivial。
- 无 RDMA  fabric：TCP 传输税更重。

### The router integrates with Phase 17 · 11

Disaggregated routers 是 KV-cache-aware（Phase 17 · 11）。请求落到持有其前缀的 decode pool——若无匹配，则走 prefill → decode。Hit rate 和 disaggregation 叠加——cache-aware router 决定是否甚至需要新的 prefill。

### MoE on Blackwell is where the real numbers are

GB300 NVL72 + Dynamo 显示相比 Hopper 基线 50 倍 MoE 吞吐量。MoE expert routing 在 prefill 上是计算密集型，在 decode 上是内存密集型（expert caches），因此 disaggregation 是双重胜利。2026 年前沿模型 serving 是 MoE-dominant（DeepSeek-V3、未来 GPT-5 变体）。

### Numbers you should remember

Benchmark numbers 会漂移——NVIDIA 和 inference stack 每季度更新结果。引用前请重新核实。

- DeepSeek-R1 on GB200 NVL72 + Dynamo：相比基线中等延迟区间约 6 倍吞吐量（developer.nvidia.com，2025-06）；社区"高达 30 倍"的全栈 Blackwell + Dynamo 说法是方向性汇总，无单一原始来源。
- GB300 NVL72 + Dynamo：相比 Hopper 高达 50 倍 MoE 吞吐量（developer.nvidia.com，无日期）。
- 节省参考（内部合成，非单一案例）：$200 万年支出中节省 $60–80 万/年，保持相同 SLA。
- Disaggregation 阈值：提示 >512 token + 输出 >200 token。
- KV transfer via NIXL：70B FP8 下 4K 提示的 KV 传输 20-80 ms。

## Use It

`code/main.py` 模拟 colocated vs disaggregated serving。报告 throughput、cost per request 和 prompt-length crossover。

## Ship It

本课产出 `outputs/skill-disaggregation-decider.md`。根据工作负载和集群，决定是否 disaggregate。

## Exercises

1. 运行 `code/main.py`。disaggregation 在何种提示长度下击败 colocation？
2. 为 RAG 服务设计 prefill pool 和 decode pool，P99 前缀长度 8K，输出 300。
3. Dynamo vs llm-d：为纯 Kubernetes 团队、无 Python runtime 偏好选择其一。
4. 计算 KV transfer 成本：70B FP8 下 4K prefill = ~500 MB KV。RDMA 100 GB/s 时传输 = 5 ms。TCP 10 GB/s = 50 ms。对你的 SLA 哪个重要？
5. MoE expert routing 改变 KV 访问模式。disaggregation 在 MoE 每 token 激活不同 expert 时如何表现？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Disaggregated serving | "split prefill/decode" | Separate GPU pools for each phase |
| NIXL | "NVIDIA transport" | Dynamo's inter-node KV transfer（RDMA/TCP） |
| NVIDIA Dynamo | "the orchestrator" | Stack-above coordinator for vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernetes native" | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | "Dynamo auto-config" | Measures workload, configures pool ratios |
| SLA Planner | "Dynamo policy" | Auto-rate-matches prefill:decode to meet SLOs |
| `packDomain: rack` | "llm-d topology" | Pack prefill+decode on same rack for fast KV |
| UCCL | "unified collective" | llm-d 0.5 networking layer for scale-to-zero |
| MoE expert routing | "expert per token" | DeepSeek-V3 pattern; disaggregation helps |

## Further Reading

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
