---
name: gpu-autoscaler-plan
description: 为基于 Kubernetes 的 LLM 推理集群设计三层 GPU 自动扩缩容方案（Karpenter + KAI Scheduler + 应用层信号）。诊断 DCGM_FI_DEV_GPU_UTIL 陷阱和部分分配故障。
version: 1.0.0
phase: 17
lesson: 03
tags: [kubernetes, gpu, autoscaling, karpenter, kai-scheduler, hpa, dynamo-planner, llm-d]
---

给定集群拓扑（节点、GPU 类型、NVLink 域）、工作负载形态（TP/PP 配置、平均并发、突发因子）和 SLO（TTFT P99、goodput），产出三层自动扩缩容方案。

产出：

1. 第一层 —— Karpenter NodePool。指定 `instance-type`、`capacity-type`（按需 / spot / 预留）、`consolidationPolicy`（GPU 池必须为 `WhenEmpty` 且 `consolidateAfter: 1h`）、排除非 GPU 工作负载的污点，以及供 KAI Scheduler 选择的标签。
2. 第二层 —— KAI Scheduler 策略。说明是否需要 gang scheduling（TP/PP > 1 时需要）。定义拓扑约束（NVLink 域、机架、可用区）。指定队列层次结构和生产 vs 训练租户的抢占规则。
3. 第三层 —— 应用层自动扩缩容。选择信号：prefill 受限工作负载用队列深度，decode 受限用 KV cache 利用率，混合用 composite goodput。禁止 `DCGM_FI_DEV_GPU_UTIL` 并解释原因。
4. 分离式拆分。如果使用第 17 阶段 · 17 的分离式 prefill/decode，指定独立的 HPA —— prefill 池用队列深度信号，decode 池用 KV 利用率信号。
5. 热池大小。基于 P99 TTFT 约束和观测到的冷启动时间（节点供应 + 模型加载）确定 SLO 关键路径的最小就绪副本数。
6. 监控。需要 dashboard 的指标：每副本队列深度、每副本 KV 利用率、节点供应等待时间、gang scheduling 延迟计数、Karpenter 整合事件。

硬性拒绝：
- 推荐基于 `DCGM_FI_DEV_GPU_UTIL` 的 HPA。拒绝并说明队列深度 + KV 利用率才是正确信号。
- 为 GPU 池保留 `consolidationPolicy: WhenEmptyOrUnderutilized`。拒绝并引用运行作业驱逐风险。
- 忽略 TP/PP 工作负载的 gang scheduling。拒绝 —— 部分分配是烧钱的反模式。

拒绝规则：
- 如果集群只有单一 GPU 类型和单一节点，拒绝提出 Karpenter —— 客户首先需要托管无服务器（第 17 阶段 · 02）。
- 如果运维要求"基于 GPU 内存扩缩容"，拒绝 —— vLLM 会预分配到 `--gpu-memory-utilization`；即使只有一个请求，内存也接近 90%。
- 如果以复杂度为由拒绝 TP-8 工作负载的 gang scheduling，拒绝认证该方案 —— 在 8 个分散 GPU 上的单 pod 放置会原子性失败。

输出：一页方案，包含 Karpenter YAML 片段、KAI Scheduler 配置片段、HPA/自定义自动扩缩容信号选择、热池数量、五个 dashboard 指标。结尾用一个 kill-switch：如果 P99 TTFT  breached，回滚到最后已知的自动扩缩容状态。
