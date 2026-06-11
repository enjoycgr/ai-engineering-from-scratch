---
name: chaos-plan
description: 设计 LLM 混沌工程计划 —— 验证先决条件、构建四个平面、选择工具、从三个安全实验开始、强制执行安全平面关卡。
version: 1.0.0
phase: 17
lesson: 24
tags: [chaos-engineering, litmuschaos, chaosmesh, harness, llm-chaos, game-day]
---

给定技术栈（Kubernetes / VM / 托管）、SLI/SLO 成熟度、可观测性质量和团队值班成熟度，产出混沌计划。

产出：

1. 先决条件检查。验证 SLI/SLO 已定义、可观测性已接入、回滚已自动化、运维手册已结构化、值班轮班已建立。如有缺失，拒绝在生产环境运行混沌。
2. 四个平面。为每个平面命名工具（control、target、safety、observability）。可观测性指向 Phase 17 · 13。
3. 三个初始实验。从 pod kill 开始。然后是 provider 429。然后是内存过载。每个都包含 blast-radius 上限、持续时间、成功标准。
4. 安全关卡。Burn-rate（> 预期 2 倍）、blast-radius（< 集群 30%）、trace-ID 标签、抑制窗口。
5. 节奏。每周小规模金丝雀。每月 game day（跨团队）。每季度韧性审计。
6. 工具。LitmusChaos（开源，CNCF 毕业项目）、Chaos Mesh（开源，CNCF 沙箱项目）、Harness Chaos（商业 AI 辅助）、AWS FIS / Azure Chaos Studio（托管云原生）。

硬性拒绝：
- 缺少五个先决条件就在生产环境运行混沌。拒绝 —— 会变成真实事件。
- 没有 blast-radius 上限的实验。拒绝。
- 没有 trace-ID 标签的实验。拒绝 —— 无法对告警去重。

拒绝规则：
- 如果团队从未在 staging 成功运行过一个实验，拒绝生产混沌，直到 staging 有一个绿色通过。
- 如果事件量已经很高（>2/周），拒绝新增混沌 —— 先稳定。
- 如果团队没有 SLO，要求在任何实验前先建立 SLO。

产出：一页计划，包含先决条件检查、四个平面工具、三个初始实验、安全关卡、节奏。最后附季度依赖映射更新承诺。
