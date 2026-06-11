---
name: vllm-stack-decider
description: 根据工作负载和集群规模，决定 vLLM 部署布局——production-stack Helm chart、KV offload（native CPU 或 LMCache）、router/observability 集成。
version: 1.0.0
phase: 17
lesson: 18
tags: [vllm, production-stack, lmcache, kv-offload, connector-api]
---

给定工作负载（提示形状、并发、前缀复用模式）、集群（引擎、GPU 类型）和运维场景（Kubernetes-native、多租户、预算），产出 vLLM 栈计划。

产出：

1. 栈。使用 vLLM production-stack Helm chart（新部署推荐）或自建。说明适用哪些 operators/CRD。
2. KV offload。选择：
   - None（短提示、低并发——overhead 超过收益）。
   - Native vLLM CPU offload（单引擎 HBM 压力，简单）。
   - LMCache connector（多引擎前缀复用、preemption-heavy 或多租户共享提示）。
3. HBM 利用率监控。设置 `--gpu-memory-utilization` 并留 headroom；持续 92%+ 时告警，作为 pre-preemption 信号。
4. Router 集成。Cache-aware router（Phase 17 · 11）。确认 KV-event 通道已配置。
5. Observability。每引擎 Prometheus scrape、OTel GenAI attributes（Phase 17 · 13）、production-stack 的 Grafana dashboard 模板。
6. 预期影响。量化预期吞吐量增益 vs 当前——引用 16x H100 基准形状（KV footprint 超过 HBM 时 LMCache 有帮助）。

硬拒绝：
- 无共享前缀或 preemption 时部署 LMCache。拒绝——overhead，无收益。
- 无 HBM 压力监控时运行 vLLM。拒绝——第一次 preemption 将是意外。
- Helm chart 已覆盖用例时自建 production-stack。拒绝——重复造轮子。

拒绝规则：
- 如果集群 <2 引擎，拒绝 LMCache——跨引擎复用是重点；单引擎用 native。
- 如果工作负载提示 < 1K token 且并发 < 100，拒绝任何 offload——HBM headroom 足够。
- 如果团队无 K8s 能力，拒绝 production-stack——从单引擎 vLLM + 简单代理开始。

输出：一页计划，含栈、KV offload 选择、HBM 监控、路由集成、observability、预期影响。结尾附单一 gate：过去 24 小时 HBM utilization P99。
