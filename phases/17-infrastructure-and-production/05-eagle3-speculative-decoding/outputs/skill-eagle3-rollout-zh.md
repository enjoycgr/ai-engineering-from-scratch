---
name: eagle3-rollout
description: 制定分阶段 EAGLE-3 投机解码上线计划，在上线前先在真实流量上测量接受率 alpha。
version: 1.0.0
phase: 17
lesson: 05
tags: [speculative-decoding, eagle-3, vllm, alpha, production-rollout]
---

给定目标模型、硬件（GPU 类型和数量）、流量描述（通用对话 / 代码 / 专业领域）、并发目标，以及当前基线指标（TTFT、ITL、吞吐量），产出分阶段 EAGLE-3 上线计划。

产出：

1. 基线测量计划。哪个基准（LLMPerf、GenAI-Perf 或生产影子流量）、哪个 prompt 分布、哪个并发点、记录哪些指标（TTFT 平均/P99、ITL 平均/P99、吞吐量、并发）。
2. 草稿头选择。通用对话用 ShareGPT 训练的 EAGLE-3。专业流量（代码、医疗、法律）用领域训练 EAGLE-3，或决定上线前先训练一个。
3. 配置。精确的 vLLM `speculative_config` 字段（method、model、num_speculative_tokens）。注意 v0.18.0 兼容性：草稿模型投机不能与 `--enable-chunked-prefill` 组合；V1 中的 N-gram GPU 投机解码是例外。
4. Alpha 门槛。目标 alpha >= 0.55（生产并发下）。测量流程：影子流量 24 小时，记录 vLLM `spec_decode_metrics`，接受 token 除以请求草稿长度。任何 1 小时窗口内 alpha 低于 0.45 则触发 kill switch。
5. 尾部观察。绘制 P99 ITL 增量（投机开启 - 投机关闭）。如果增量为正，被拒绝草稿的两阶段模式在咬人。降低 K 或在此工作负载上禁用。
6. 盈亏平衡检查。在报告的并发下，计算当前验证开销的盈亏平衡 alpha。仅当测量 alpha 超过盈亏平衡至少 0.1 时才上线。

硬性拒绝：
- 未在生产流量上测量 alpha 就上线。拒绝并要求 24 小时影子测量。
- 不说明测量 alpha 就声称 2-3x 加速。
- 为延迟不是约束的离线 batch 作业启用投机解码。
- 在 vLLM v0.18.0 上将草稿模型投机与 chunked prefill 组合。硬不兼容。

拒绝规则：
- 如果流量主要是极短输出（平均低于 50 token），拒绝。草稿开销占主导；部署纯目标模型。
- 如果硬件是消费级（RTX 4090 / 5090）且 batch size 低于 8，推荐纯目标模型——验证开销的 batch 摊销需要硬件无法提供的并发。
- 如果用户想要没有测量循环的 K 自动调优，拒绝。K 从测量的 alpha 加验证开销中选择；没有自动调优能替代测量。

输出：一页分阶段上线计划，列出基线 → 配置 → alpha 门槛 → 尾部观察 → 盈亏平衡确认。结尾用"下一步测量什么"段落，根据诊断命名领域特定 EAGLE-3 训练、降低 K 或回退到纯目标模型之一。
