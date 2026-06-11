---
name: disaggregation-decider
description: 针对给定工作负载和集群，决定是否采用 disaggregated prefill/decode（Dynamo 或 llm-d）。量化 prefill:decode 比例、KV transfer 成本和预期节省。
version: 1.0.0
phase: 17
lesson: 17
tags: [disaggregated-serving, dynamo, llm-d, nixl, kv-transfer, prefill-decode]
---

给定工作负载画像（提示/输出长度分布、模型、并发）、集群拓扑（GPU、fabric、RDMA 可用性）和当前 serving 成本，产出 disaggregation 决策。

产出：

1. 是否分离？是/否，附编号理由。基线：提示 > 512 且输出 > 200。Fabric：有 RDMA 有帮助；仅 TCP 会推高盈亏平衡点。
2. 栈选择。NVIDIA Dynamo（vLLM/SGLang/TRT-LLM 之上的托管编排器）或 llm-d（Kubernetes-native Services）。匹配运维场景。
3. Prefill:decode 比例。使用 Dynamo Planner Profiler 读数，或从工作负载形状计算（prefill TFLOPS vs decode bytes/sec）。示例：RAG-heavy 为 2 prefill : 1 decode；输出-heavy 为 1:2。
4. KV transfer 计划。命名传输层（NIXL over InfiniBand / RDMA / TCP fallback）。按提示 P99 计算每次请求的 transfer tax。
5. Router 集成。Cache-aware router（Phase 17 · 11）必须在前面——无 prefix matching 的分离式部署会失去缓存收益。
6. 预期节省。与 colocated 基线对比计算；引用公开案例（相同 SLA 下 30-40%）。

硬拒绝：
- 对短提示工作负载（<512 token）做分离式部署。拒绝——transfer tax 占主导。
- 无 cache-aware router 的部署。拒绝——盲路由抵消了 KV locality。
- 忽略拓扑（机架打包）。拒绝——跨机架多跳的 KV transfer 成本高于同机架 RDMA。

拒绝规则：
- 如果集群 < 4 GPU，拒绝——池多样性不足，分离式部署不划算。
- 如果无 RDMA/InfiniBand 且无计划，说明 TCP 将盈亏平衡点推高到提示 >2K；重新评估。
- 如果团队无法运维两个带按角色扩缩容的 GPU 池，拒绝 llm-d 并要求 Dynamo 作为托管替代方案。

输出：一页决策文档，含是否分离、栈选择、比例、传输、路由、预期节省。结尾附单一验证指标：KV transfer P99 latency；超过计划阈值时 gate。
