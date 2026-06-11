---
name: cross-policy-diff
description: 针对特定能力，使用 OpenAI Preparedness Framework v2、Anthropic RSP v3.0 和 DeepMind FSF v3 作为参考生成交叉策略对比。
version: 1.0.0
phase: 15
lesson: 20
tags: [preparedness-framework, fsf, rsp, cross-policy, scaling-policy]
---

给定一个特定的前沿能力（例如，"长程自主性"、"自主复制与适应"、"R&D 自动化"），生成交叉策略差异 (cross-policy diff)，展示三个框架如何分类该能力以及触发什么缓解措施。

产出：

1. **OpenAI PF v2 分类。** Tracked 或 Research。如果是 Tracked，命名 Capabilities + Safeguards Report 触发条件。如果是 Research，注意策略语言是 "潜在" 缓解措施。
2. **Anthropic RSP v3.0 分类。** 哪个阈值（ASL-3、AI R&D-4、hardcoded prohibition）？哪种缓解措施（肯定案例、security + deployment）？确认承诺住在 Anthropic 单方面层级还是行业建议层级。
3. **DeepMind FSF v3 分类。** 哪个领域（Cyber、Bio、ML R&D、CBRN）？哪个 CCL 或 Tracked Capability Level？是否调用了欺骗对齐监控 (deceptive alignment monitoring)？
4. **趋同总结 (Convergence summary)。** 三个策略在能力严重性上是否一致，还是存在有意义的分歧？哪种分类最严格，哪种最不严格？
5. **测量依赖 (Measurement dependency)。** 每个分类都依赖能力测量。命名该能力如何测量以及哪个评估提供者（METR、Apollo、internal、third-party）拥有该测量。

硬性否决：
- 基于公告语言相似性而非文档级证据声称跨策略对齐。
- 任何无法指向源文档中具体条款的分类。
- 将 "Research Category"（OpenAI）等同于 "Tracked Category" —— 它们有不同的操作后果。

拒绝规则：
- 如果用户无法为每个分类提供源文档段落，拒绝并要求先提供引用。
- 如果用户将策略存在视为缓解措施实际执行的证据，拒绝并要求具体缓解措施已触发的证据。
- 如果能力被声称被某个框架 "覆盖" 但该词未出现在文档中，拒绝并要求具体条款引用。

输出格式：

返回一份差异文档，包含：
- **能力定义**（一句话）
- **OpenAI PF v2 行**（分类、触发条件、源条款）
- **Anthropic RSP v3.0 行**（分类、触发条件、unilateral-vs-recommendation）
- **DeepMind FSF v3 行**（领域、CCL / TCL、deceptive-alignment 参与）
- **趋同总结**（一致 + 有意义的分歧）
- **测量归属**（评估提供者、评估周期）
- **读者推荐**（最严格、最不严格、附理由）
