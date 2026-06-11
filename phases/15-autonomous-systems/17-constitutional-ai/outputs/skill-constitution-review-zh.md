---
name: constitution-review
description: 审计部署的宪法层 (constitutional layer) —— 硬编码禁令 (hardcoded prohibitions)、软编码默认值 (soft-coded defaults)、操作者可调边界 (operator-adjustable bounds) 以及四层层级解析 (four-tier hierarchy resolution)。
version: 1.0.0
phase: 15
lesson: 17
tags: [constitutional-ai, rule-override, hierarchy, cai, rlaif, hardcoded-prohibition]
---

给定部署的宪法层（system prompt、operator config、声明原则），对照 Claude Constitution 参考标准进行审计，标记缺失的硬编码禁令、模糊原则或层级顺序错误。

产出：

1. **硬编码禁令清单 (Hardcoded prohibition inventory)。** 列出无论操作者或用户指令如何都不可弯曲的每项禁令。最低底线：生物武器 / CBRN 增强、CSAM、关键基础设施攻击策划、被要求时的虚假身份。新增项取决于部署场景（例如，金融服务添加特定的欺诈禁令）。
2. **软编码默认值 (Soft-coded defaults)。** 列出操作者可调整的每种行为。对每种，声明其已声明的边界 (bound)。一个 "可调整" 但无边界设置的后门覆盖 (back-door override)。
3. **层级排序 (Tier ordering)。** 确认解析顺序为：safety > ethics > guidelines > helpfulness。如果在实现的解析器中 helpfulness 胜过 ethics，标记为部署中断 (deployment break)。
4. **原则模糊性标记 (Principle ambiguity flags)。** 识别任何文本留下实质性不同解释空间的原则。模糊性会在训练周期中累积（原则漂移，principle drift）。
5. **层级完整性 (Layer completeness)。** 确认运行时层控制（第 10、13、14 课）与宪法层同时存在。仅靠宪法层不足够；仅靠运行时层也不足够。

硬性否决：
- 没有任何硬编码禁令层的部署。
- 声称可覆盖硬编码禁令的操作者配置（即使通过重命名）。
- 将 helpfulness 置于 ethics 之上的层级顺序。
- 文本过于笼统以至于无法评估的原则（"做个好人"）。
- 将 Constitutional AI 视为运行时控制的替代品。

拒绝规则：
- 如果用户说出了一个硬编码禁令但无法指向其运行时层后备 (runtime-layer backstop)，将部署标记为单层 (single-layer) 并拒绝生产上线。
- 如果操作者配置包含一个可调整的 "safety" 设置但没有声明边界，拒绝。
- 如果用户将 2023 年的参与式宪法 (participatory-constitution) 发现视为在当前部署中可执行，请检查：2026 年宪法并未纳入它们，因此 "民主继承" 是部署无法支持的主张。

输出格式：

返回一份宪法审计报告，包含：
- **硬编码底线**（禁令、执行层：weights / inference / both）
- **软编码默认值**（设置、操作者边界、用户可见 y/n）
- **层级顺序**（列出；确认 safety > ethics > guidelines > helpfulness）
- **模糊性标记**（原则、具体模糊点、建议收紧）
- **层级完整性**（宪法层 y/n、运行时控制 y/n、两者都需）
- **就绪度**（production / staging / research-only）
