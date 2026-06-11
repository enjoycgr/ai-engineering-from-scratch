---
name: scaling-policy-review
description: 审查前沿实验室的扩展策略 (scaling policy)（Anthropic RSP、OpenAI Preparedness、DeepMind FSF、内部策略），对照 RSP v3.0 参考形态。
version: 1.0.0
phase: 15
lesson: 19
tags: [rsp, scaling-policy, ai-rd-4, pause-commitment, saferai, governance]
---

给定一个已发布或拟议的扩展策略，生成一份结构化审查，将其与 RSP v3.0 参考形态（AI R&D-4、肯定案例、双层缓解、Frontier Safety Roadmap、Risk Report、独立审查）进行比较。

产出：

1. **双层清单 (Two-tier inventory)。** 将承诺分为 "实验室单方面" (lab-unilateral) 和 "行业范围建议" (industry-wide recommendation)。建议层的承诺是倡导，而非承诺。统计比例；如果大部分承诺住在建议层，这是一个弱策略。
2. **阈值 (Thresholds)。** 命名每个能力阈值及触发缓解措施。标记 v2 有定量而此处为定性的阈值。标记策略声称覆盖但缺失阈值的能力。
3. **暂停承诺 (Pause commitment)。** 确认策略在特定阈值处命名了暂停条款（训练停止、部署暂停或类似）。v3.0 移除了此项；跟随的策略继承了这一回退 (regression)。
4. **常备文件 (Standing artifacts)。** 确认策略强制要求常备的 Frontier Safety Roadmap 和 Risk Report 文件，并声明周期。事后发布的一次性文件不符合要求。
5. **独立审查 (Independent review)。** 命名外部审查机制。纯内部审查（由实验室员工组成的 "Safety Advisory Group"）不构成独立监督。

硬性否决：
- 没有命名能力阈值的策略。
- 缓解措施全部住在行业建议层的策略。
- 没有常备 Roadmap / Risk Report 文件的策略。
- 没有独立审查机制的策略。
- 声称 "从现实世界经验中学习" 但未说明策略文本如何更新及更新周期的策略。

拒绝规则：
- 如果策略文件是营销而非治理（无具体承诺、无阈值、无周期），拒绝将其评定为扩展策略。
- 如果用户将策略的存在视为合规的等价物，拒绝。策略是承诺装置；合规需要证据。
- 如果用户引用旧版策略（例如，2023 Anthropic RSP）作为当前版本，拒绝并要求当前版本。

输出格式：

返回一份策略审查报告，包含：
- **双层比例**（unilateral / recommendation / 总计数）
- **阈值表**（名称、类型：quantitative / qualitative、触发条件、缓解措施）
- **暂停承诺**（存在 y/n、具体条款）
- **常备文件**（Roadmap 周期、Risk Report 周期）
- **独立审查**（机制、审查者身份、频率）
- **总体评级**（strong / moderate / weak，附理由）
