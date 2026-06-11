---
name: ab-plan
description: 设计 LLM A/B 测试——选择平台（Statsig 或 GrowthBook）、primary metric、guardrails、含 LLM 噪声缓冲的样本量、CUPED、序贯停止和多重比较校正。
version: 1.0.0
phase: 17
lesson: 21
tags: [ab-testing, statsig, growthbook, cuped, sequential, benjamini-hochberg, srm]
---

给定功能变更（prompt / model / generation parameter）、基线指标、预期提升和团队姿态（数仓原生 OSS vs 打包 SaaS），产出 A/B 计划。

产出：

1. 平台。Statsig（打包 SaaS，OpenAI 拥有）或 GrowthBook（MIT OSS，数仓原生）。说明理由。
2. Primary metric + guardrails。Primary 是你想推动的指标；guardrails 是绝不能回退的（cost/request、latency P99、refusal rate）。
3. 样本量。经典 power calculation × 1.4（LLM 非确定性缓冲）。
4. 设计。Fixed-horizon 或 sequential。预期强信号时选 sequential；变化微妙时选 fixed。
5. CUPED。若 primary metric 存在前周期数据则启用；指定回归量。
6. 校正。测试数量少时用 Bonferroni；大量相关测试时用 Benjamini-Hochberg。
7. SRM。每个实验都要求 SRM check；标记时停止并调试。

硬拒绝：
- 凭感觉上线。拒绝——要求 A/B 或记录的无 A/B 例外。
- 同一 primary metric 上运行 >5 个实验而不做 BH/Bonferroni。拒绝——false discovery 必然发生。
- 跳过 SRM check。拒绝——分配 bug 很常见。

拒绝规则：
- 如果功能流量 < 1000 用户/周，拒绝 fixed A/B——要求 shadow + canary（Phase 17 · 20）替代。
- 如果 primary metric 是主观的（例如"质量"）且无客观代理，要求并行人工评估。
- 如果提升假设小于 LLM 噪声底，拒绝——实验无法以现实样本量检测到它。

输出：一页计划，含平台、primary + guardrails、样本量、设计、CUPED、校正、SRM 策略。结尾附决策规则：primary 显著 + 所有 guardrails 非显著负向 → 上线；任何 guardrail 违规 → 无论 primary 如何都不上线。
