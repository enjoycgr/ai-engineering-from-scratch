---
name: managed-platform-picker
description: 根据工作负载、SLA 和合规要求选择托管 LLM 平台（Bedrock、Azure OpenAI、Vertex AI）及冗余备用平台，然后制定 FinOps 插桩计划。
version: 1.0.0
phase: 17
lesson: 01
tags: [bedrock, azure-openai, vertex-ai, ptu, finops, managed-platforms]
---

给定工作负载画像（所需模型、每月 token 量、P50/P99 TTFT SLA、合规约束、现有云基础），产出平台推荐。

产出：

1. 主平台。命名平台、其覆盖的具体模型，以及根据利用率判断按需实例还是 Provisioned Throughput Units (PTUs)（预置吞吐单元）/ Provisioned Throughput 更合适。引用盈亏平衡数学（PTU 大约在 40-60% 持续利用率时划算）。
2. 备用平台。命名最少双供应商的备用。证明配对的合理性 —— 冗余必须覆盖模型重叠（Bedrock 上的 Claude + Azure OpenAI 上的 GPT 是常见组合）和区域重叠。
3. FinOps 插桩。指定第一天要启用的内容：Bedrock Application Inference Profiles、Azure 作用域 + PTU 预留作为成本对象、Vertex 按项目分团队 + BigQuery Billing Export。命名归因维度 —— 按用户、按任务、按租户。
4. SLA 检查。将目标 TTFT P99 与公开基准比较（Azure OpenAI PTU ≈ 50 ms P50；Bedrock 按需 ≈ 75 ms P50）。如果 SLA 比按需实例能提供的更严格，则要求 PTU。
5. 合规检查。根据需要验证 BAA、SOC 2 Type II、HIPAA、欧盟数据驻留。注意三者都满足基线，但保留策略和滥用监控（abuse monitoring）退出方式不同。
6. 迁移路径。命名团队本周可以采取的一个可逆步骤（例如，通过抽象供应商的 AI 网关部署；插桩归因头）和一个长期步骤（PTU 承诺；跨区域故障转移）。

硬性拒绝：
- 推荐没有命名备用的单一平台。拒绝并坚持最少双供应商。
- 没有利用率估算就推荐 PTU。拒绝并要求持续利用率数据。
- 在归因被列为需求时忽略 Bedrock Application Inference Profiles —— 它们是最干净的原生界面。

拒绝规则：
- 如果工作负载需要 Claude、Gemini 和 GPT 都是 P0，说明三平台现实（Bedrock + Vertex + Azure OpenAI 在网关后），而不是假装一个平台能服务所有三个。
- 如果 SLA 是 TTFT P99 < 100 ms 且预期预算无法支持 PTU，拒绝承诺 SLA —— 解释按需实例的方差上限。
- 如果客户要求"使用最便宜的供应商"，拒绝 —— 价格是多维的（token 费率 + 专属容量 + 归因开销 + 锁定成本）。

输出：一页决策，包含主平台、备用平台、PTU vs 按需、插桩列表、SLA/合规验证、两个迁移步骤。结尾用一个指标捕捉计划偏离（持续利用率、PTU 浪费或归因覆盖率）。
