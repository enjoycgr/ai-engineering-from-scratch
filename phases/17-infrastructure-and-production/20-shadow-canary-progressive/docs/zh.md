# Shadow Traffic（影子流量）、Canary Rollout（金丝雀发布）与 Progressive Deployment（渐进式部署）for LLMs

> LLM 发布结合了软件部署中最困难的部分：没有单元测试、弥散的故障模式、延迟的信号。正确顺序是：(1) shadow mode（影子模式）——将生产请求复制到候选模型，记录日志，与生产对比，对用户零影响；能捕获明显的分布问题但不是质量保证；(2) canary rollout（金丝雀发布）——渐进式流量切换 10% → 25% → 50% → 75% → 100%，每步设 gates（关卡）；跟踪 latency percentiles（延迟分位数）、cost/request（单次请求成本）、error/refusal rate（错误/拒绝率）、output length distribution（输出长度分布）、user-feedback rate（用户反馈率）；(3) 稳定性确认后对明显不同的替代方案进行 A/B testing（A/B 测试）。Non-determinism（非确定性）不可消除——由于 GPU FP non-associativity（浮点非结合性）加 batch-size variance，相同输入跨运行可达 15% 的 accuracy variation。Cost 是变量而非常量——一个提升 20% 的模型每次调用可能贵 3 倍。Rollback（回滚）速度是决定性的：如果回滚需要 redeploy（重新部署），你就太慢了。Policy 存在于 config/flags 中；模型存在于 registry 中并带有 pinned digests（固定摘要）；回滚 = 翻转 policy + 恢复阈值 + 秒级内固定旧模型。

**Type:** Learn
**Languages:** Python（stdlib，toy canary-progression simulator）
**Prerequisites:** Phase 17 · 13（Observability）, Phase 17 · 21（A/B Testing）
**Time:** ~60 minutes

## Learning Objectives

- 区分 shadow mode（零影响对比）、canary（真实流量渐进式）和 A/B（稳定性确认后对比）。
- 列举五个 LLM 特有的 canary metrics（延迟、单次请求成本、错误/拒绝、输出长度分布、用户反馈）。
- 解释为何 LLM non-determinism（高达 15%）改变了发布中"稳定"的含义。
- 设计一条秒级回滚路径（policy flip），而非小时级（redeploy）。

## The Problem

你发布了一个新模型。Offline evals（离线评估）显示准确率提升 3%。你在生产环境直接切换。24 小时内，成本上升 40%，用户 thumbs-down（点踩）上升 8%，三个客户工单报告"答案奇怪"。你回滚。Redeploy 耗时 3 小时。你的周末毁了。

每一步都可避免。Shadow mode 本可在任何用户看到之前捕获 40% 的成本飙升。Canary 本可在 thumbs-down 异动时停在 10%。Policy-flag 回滚本可在 30 秒内完成。这种纪律填补了"离线评估看起来不错"与"真实用户满意"之间的鸿沟。

## The Concept

### Shadow mode

候选模型接收与生产相同的请求；输出被记录但不返回给用户。对用户零影响。记录：

- 输出内容（与生产对比 diff）。
- Token 数量（成本增量）。
- Latency。
- Refusal 和 error。

能捕获：成本暴涨、长度回退、明显的 refusal 变化、硬性错误。不能捕获：用户能感知到的质量差异。Shadow 是 smoke test（冒烟测试），不是 quality test。

### Canary rollout

渐进式流量切换并设 gates。典型推进：1% → 10% → 25% → 50% → 75% → 100%。每步检查 5 个指标：

1. **Latency percentiles** — P50、P95、P99。违规：canary 的 P99 > 1.5 倍基线。
2. **Cost per request** — 混合成本。违规：> 基线 20%。
3. **Error / refusal rate** — 5xx 加明确拒绝。违规：2 倍基线。
4. **Output length distribution** — mean + P99。违规：分布偏移。
5. **User-feedback rate** — thumbs-down / 工单提交。违规：1.5 倍基线。

### Non-determinism is the new variance

相同输入产生不同输出。原因：

- GPU FP non-associativity（浮点约简顺序因 batch 而异）。
- Batch-size variance（同一提示在 batch 128 与 batch 16 中）。
- Sampling（temperature > 0）。

实测：相同评估集上运行间准确率差异可达 15%。发布中的"stable"指指标在预期 variance 内，而非与基线完全相同。Gates 应设在 noise floor 之上。

### Cost is a variable

一个提升 20% 的模型每次调用可能贵 3 倍。Cost/request 是五个 gates 之一。发布一个"更好"但破坏 unit economics（单位经济）的模型是回滚场景。

### Rollback is the weapon

- Policy flag（feature flag 系统）：在 config 中翻转百分比；秒级完成。
- Model pinning（registry digest）：固定模型不会自动升级。
- 回滚 = 恢复 flag + 将 pinned digest 设为上一版本。秒级，而非小时级。

如果你的堆栈回滚需要 redeploy，在发布前先修复。

### Tooling

**Argo Rollouts** / **Flagger** — Kubernetes progressive delivery 控制器。集成 Istio/Linkerd weighted routing。

**Istio weighted routing** — service-mesh 级流量拆分。

**KServe / Seldon Core** — 内置 canary 的模型 serving。

**Feature flags** — LaunchDarkly、Flagsmith、Unleash。Policy 级翻转，无需 redeploy。

### Metrics cadence

Canary gates 每 5-15 分钟检查一次，取决于流量。1% 流量且 10 req/min 时，每窗口 50-150 个数据点——对 latency 足够但对用户反馈很嘈杂。10% 时约多 10 倍。每步应暂停足够长时间以积累足够样本。

### The A/B step is optional

如果新模型明显不同（不同行为、不同成本曲线、不同语气），canary 通过后以 50% 进行 A/B test。如果只是改进版本，canary gates 通过后直接跳到 100%。

### Numbers you should remember

- Canary progression：1% → 10% → 25% → 50% → 75% → 100%。
- Non-determinism 上限：相同输入运行间 variance 可达 15%。
- 五个 canary metrics：latency、cost、error/refusal、output length、user feedback。
- Cost gate：> 基线 20% 即违规。
- 回滚：秒级，而非小时级。

## Use It

`code/main.py` 模拟带有注入回退的 canary rollout。报告发布停在哪个阶段以及哪个 gate 触发。

## Ship It

本课产出 `outputs/skill-rollout-runbook.md`。根据候选模型、基线和风险容忍度，设计 shadow→canary→100% 计划。

## Exercises

1. 运行 `code/main.py`。注入 25% 成本回退。Canary 停在哪个阶段？
2. 你的新模型离线准确率提升 3%，但 cost/request 增加 18%。是否上线？取决于策略——写出两种路径。
3. 设计一条端到端 60 秒内的回滚路径。列出所需基础设施。
4. Non-determinism 在评估中显示 ±7%。设置 canary gates 以避免误报。使用什么乘数？
5. Shadow mode 在 canary 之前捕获 40% 成本飙升。写出在 shadow 中触发的 alert rule。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Shadow mode | "duplicate to new" | Zero-impact send-to-candidate for logging |
| Canary | "progressive traffic" | Gradual user-exposed rollout with gates |
| Gates | "rollout checks" | Metric thresholds that block progression |
| Non-determinism | "LLM variance" | Irreducible run-to-run differences |
| Policy flag | "flag flip rollback" | Config-level rollback, seconds not hours |
| Model pin | "registry digest" | Immutable reference to a model version |
| Argo Rollouts | "K8s progressive" | Kubernetes-native canary/rollback controller |
| KServe | "inference K8s" | Model serving with canary primitives |
| Istio weighted | "mesh split" | Service-mesh traffic splitter |

## Further Reading

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
