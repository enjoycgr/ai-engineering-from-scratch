---
name: societal-risk-review
description: 使用 CAIS 四风险框架 (four-risk framework) 和 CAISI / SB-53 监管上下文审查部署的社会级风险姿态。
version: 1.0.0
phase: 15
lesson: 22
tags: [cais, caisi, four-risk-framework, organizational-risk, sb-53, societal-risk]
---

给定一个拟议或运行中的 AI 部署，生成社会级风险审查，对照 CAIS 四风险框架标记部署，盘点组织风险子杠杆 (sub-levers)，并命名监管层面 (regulatory surface)。

产出：

1. **四风险标记 (Four-risk tagging)。** 对四个类别中的每一个（malicious use、AI races、organizational risks、rogue AIs），说明部署是否触及以及如何触及。一个部署可以触及多个类别；"不适用" 必须用一句话解释。
2. **组织风险清单 (Organizational-risk inventory)。** 对照四个子杠杆评分：safety culture、audit rigor、multi-layered defenses、information security。任何评分为 "缺失" 的杠杆都是标记缺口。
3. **监管层面 (Regulatory surface)。** 命名适用的监管框架：EU AI Act（如果在欧盟或服务于欧盟用户）、California SB-53（如果已签署且适用）、CAISI 自愿协议（如果实验室已签署）。合规是部署门槛，而非部署的锦上添花。
4. **外部评估姿态 (External-evaluation posture)。** 命名部署或其基础模型已接受的外部评估（METR、CAISI、Apollo、Gray Swan 等）。长时程自主部署没有外部评估是标记缺口。
5. **结构性力量暴露 (Structural-force exposure)。** 估计组织面临的竞争部署压力有多大，以及这与组织风险杠杆如何权衡。处于激烈竞赛压力下的团队首先降低审计优先级；这是 CAIS 的发现。

硬性否决：
- 触及有害能力类别但没有硬编码禁令层（第 17 课）的部署。
- 处于竞赛条件但没有独立审计的部署。
- 没有外部能力评估的长时程自主部署。
- 欧盟部署没有 Article 14 HITL（第 15 课）。
- 加利福尼亚部署如果 SB-53 已签署但没有事件报告流程。

拒绝规则：
- 如果用户无法说出基础模型的外部评估者，拒绝并要求先识别。自我评估不足够。
- 如果用户将 "我们有一个扩展策略" 视为灾难性风险合规，拒绝并要求具体的监管层面映射。
- 如果用户提议在没有审计的情况下在竞赛压力下部署，拒绝并引用 CAIS 关于组织风险的发现。

输出格式：

返回一份社会风险审查报告，包含：
- **四风险行表**（类别、触及 y/n、性质）
- **组织风险记分卡**（safety culture / audit / defenses / infosec）
- **监管层面**（适用框架及合规状态）
- **外部评估姿态**（评估者、范围、周期）
- **结构性力量暴露**（low / medium / high 附理由）
- **部署就绪度**（production / staging / research-only）
