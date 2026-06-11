---
name: inference-platform-picker
description: 根据工作负载、SLA、预算和运营约束选择推理平台（Fireworks、Together、Baseten、Modal、Replicate、Anyscale 或定制芯片）。统一按 token、按分钟和按预测的定价。
version: 1.0.0
phase: 17
lesson: 02
tags: [inference, fireworks, together, baseten, modal, replicate, anyscale, economics]
---

给定工作负载画像（模型、每日 token 量、持续利用率、TTFT SLA、突发因子、合规性、Python  vs 混合技术栈），产出平台推荐。

产出：

1. 主平台。命名平台和具体定价层（无服务器 vs 专属 vs 批处理）。用匹配的工作负载特征说明理由 —— 例如，"Fireworks 无服务器，因为 SLA 要求 TTFT < 500 ms 且流量是突发型"。
2. 有效成本。将所选定价模型统一为 $/M 输出 token。与至少两个替代方案比较。说明按分钟何时击败按 token（持续利用率高于约 30%）或反之。
3. 冷启动计划。对于无服务器选择（Fireworks、Modal、Replicate），说明预期冷启动延迟和缓解措施（预加热、min_workers=1、实时迁移）。对于专属选择（Baseten、Anyscale），跳过此部分但注明权衡。
4. 备选。命名第二平台和明确切换条件（例如，"如果我们签订需要 HIPAA + 专属 GPU 的企业协议，则迁移到 Baseten"）。
5. 网关层。建议是否在平台前部署 AI 网关（LiteLLM、Portkey、Kong AI Gateway）以隔离产品与供应商变动。默认：是，除非规模低于 500 RPS。

硬性拒绝：
- 未统一就对比按 token 和按分钟。拒绝并坚持有效 $/M token。
- 因为"最快"而选择 Fireworks，但未对照公开基准验证 TTFT SLA。
- 为非延迟绑定工作负载推荐定制芯片（Groq、Cerebras、SambaNova）。它们定价溢价，仅在交互式 SLA 下才合理。

拒绝规则：
- 如果工作负载需要受监管框架（SOC 2 Type II、HIPAA）且客户选择了 Modal 或 Replicate，拒绝 —— 两者的企业级足迹不如 Baseten 或 Anyscale。建议 Baseten。
- 如果预期流量低于 100k token/天，拒绝推荐按分钟（Baseten、Modal、Anyscale）。经济学不成立 —— 默认选择市场型（OpenRouter、DeepInfra）或托管云厂商。
- 如果客户想要"最便宜的"，拒绝 —— 说明多维成本函数（token 费率 + 冷启动 + 归因 + 网关 + 开发者体验）。

输出：一页推荐，包含主平台、有效成本、冷启动计划、备选、网关策略。结尾用一个指标揭示选择错误（冷启动 P99、按 token 费率或利用率漂移）。
