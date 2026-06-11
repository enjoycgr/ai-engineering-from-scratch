# CAIS、CAISI 与社会规模风险

> Center for AI Safety（CAIS，旧金山，由 Hendrycks 和 Zhang 于 2022 年创立）发布 four-risk framework（四类风险框架）——恶意使用、AI 竞赛、组织风险、rogue AIs（失控 AI）——以及 2023 年 5 月由数百名教授和公司领导者签署的灭绝风险声明。2026 年 CAIS 发布：AI Dashboard（前沿模型评估）、Remote Labor Index（与 Scale AI 合作）、Superintelligence Strategy Paper、AI Frontiers newsletter。一个独立的实体：NIST Center for AI Standards and Innovation（CAISI）——面向美国政府的 voluntary agreements（自愿协议）和 unclassified capability evaluations（非机密能力评估），专注于 cyber、bio 和 chemical-weapons risks（化学武器风险）。CAIS 将组织风险列为四个顶级风险之一：安全文化、严格审计、多层防御和信息安全是基础性的，但经常被与部署速度进行权衡。California SB-53 如果签署，将成为美国首个州级灾难性风险法规。

**类型：** Learn
**语言：** Python（stdlib，four-risk inventory and mitigation matcher，四类风险清单与缓解匹配器）
**前置条件：** Phase 15 · 19（RSP），Phase 15 · 20（PF + FSF）
**时间：** ~45 分钟

## 问题

第 19 和 20 课涵盖了实验室内部扩展政策。第 21 课涵盖了独立能力评估。本课涵盖第三种视角：塑造公众讨论和灾难性 AI 风险监管基线的公民社会和政府组织。

两个不同的实体很重要。CAIS 是一个非营利研究组织，发布 AI 风险思考框架并协调公开声明。CAISI 是 NIST 内的美国政府中心，运行与实验室的自愿协议和非机密能力评估。名字押韵；使命不重叠。从业者应该了解两者。

实用内容：CAIS 的 four-risk framework 是文献中被最广泛引用的社会规模风险 taxonomy。安全文化和组织风险是其中四类之一，这是从业者最能直接控制的一类。SB-53（加利福尼亚）如果签署，将成为美国首个州级灾难性风险法规；该法案的框架很重要，因为在美国科技政策中，州级法规历史上引领了联邦行动。

## 概念

### CAIS — Center for AI Safety

- 成立：2022 年旧金山，由 Dan Hendrycks 和同事创立（"Zhang" 名字指的是早期合作者，不是现任联合创始人；参见 CAIS 网站了解现任领导层）。
- 状态：501(c)(3) 非营利组织。
- 2023 年著名产出：灭绝风险声明，由数百名研究人员和 CEO 共同签署。声明："Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as pandemics and nuclear war."
- 2026 年产出：AI Dashboard 用于前沿模型评估、Remote Labor Index（与 Scale AI 合作）、Superintelligence Strategy Paper、AI Frontiers newsletter。

### 四类风险框架

CAIS 的框架将灾难性 AI 风险分为四个顶级类别：

1. **Malicious use（恶意使用）**：坏行为者使用 AI 造成伤害（生物武器合成、虚假信息、网络攻击）。
2. **AI races（AI 竞赛）**：实验室、公司或国家之间的竞争性压力推动部署越过安全点。
3. **Organizational risks（组织风险）**：内部实验室动态（安全文化失败、审计不足、资源不足的安全）导致不良部署。
4. **Rogue AIs（失控 AI）**：一个足够有能力的 AI 追求与人类福利冲突的目标。

这不是唯一的 taxonomy；它是最常被引用的。类别不是互斥的——一个由在竞赛中为了速度而交易审计的组织生产的 rogue AI 涵盖全部四类。

### 组织风险所在

在四种类别中，组织风险对从业者最可操作。实验室的安全文化、审计严格性、防御分层和信息安全决定他们的模型是否真正落实了第 10–18 课的控制，还是这些控制只是没人验证的清单项。

具体的组织风险杠杆：

- **Safety culture**：团队成员是否能够在没有职业成本的情况下升级关切？CAIS 调查发现这是其他杠杆的强预测因子。
- **Rigorous audits**：外部和内部。仅内部审计产生乐观报告。
- **Multi-layered defenses**：没有单层是充分的（Phase 15 的持续主题）。
- **Information security**：模型权重泄漏、评估数据泄漏、监控绕过技术泄漏。第 19 课中的 RAND SL-4 是具体标准。

### CAISI — Center for AI Standards and Innovation

- 在 NIST 内运作。
- 与前沿实验室运行自愿协议。
- 发布专注于 cyber、bio 和 chemical-weapons risks 的非机密能力评估。
- 与 CAIS 不同；首字母缩写冲突；检查 URL（nist.gov）以确认你在阅读哪个。

CAISI 的角色是 METR 私人实验室合作（第 21 课）的公共、政府面向的对应物。CAISI 报告是非机密的；METR 报告经常是 NDA 限制的。从业者同时阅读两者获得更完整的图景。

### California SB-53

加利福尼亚参议院法案（2025–2026 会期）解决前沿模型的灾难性风险。草案的关键条款：

- 触发州级义务的特定能力阈值。
- AI 实验室员工的 whistleblower protections（举报人保护）。
- 灾难性失败的事件报告要求。

如果签署，它将成为美国首个州级灾难性风险法规。无论签署状态如何，该法案的框架塑造了其他州立法机构如何处理该问题。加利福尼亚的从业者应跟踪法案状态；其他地方的从业者应阅读它以了解美国州级法规可能的样子。

### 社会规模风险不是单层问题

Phase 15 的持续主题——defense in depth（纵深防御）——也适用于社会层。没有单一组织、法规或框架能关闭灾难性风险。生态系统只有在以下情况下才能运作：

- 实验室发布扩展政策（第 19、20 课）。
- 外部评估者产生测量（第 21 课）。
- 公民社会跟踪并公开（CAIS）。
- 政府运行自愿项目和基线监管（CAISI、SB-53）。
- 从业者构建多层控制（第 10–18 课）。

这是本 phase 的最终综合：每一课都是栈中的一层，其完整性比任何单层的强度更重要。

## 使用

`code/main.py` 实现了一个小型风险清单工具。给定一个提议的部署，它根据 four-risk categories 标记部署并返回缓解检查清单。它是框架的阅读辅助，不能替代人类判断。

## 交付

`outputs/skill-societal-risk-review.md` 审查部署的社会规模风险姿态：它触及四类中的哪些、有什么缓解措施、组织风险暴露是什么。

## 练习

1. 运行 `code/main.py`。输入三个不同规模的合成部署。确认 four-risk tags 与你的预期匹配；识别一个工具 under-tag 或 over-tag 的案例。

2. 全文阅读 CAIS four-risk 论文。选择一个风险类别，写两段关于你认为该类别中 2026 年最重要的发展。

3. 阅读 California SB-53 的当前草案。识别一个你认为加强灾难性风险姿态的条款和一个你认为削弱它的条款。为两者 justify。

4. 选择一个你了解的生产 AI 部署（你的或已发布的）。根据组织风险子杠杆评分：安全文化、审计严格性、多层防御、信息安全。哪个最弱？将其提升到同等水平需要多少成本？

5. 勾画一个 2028 年版本的 four-risk framework，反映一年的额外能力和一年的额外部署经验。你会添加、移除或重新分组什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| CAIS | "Center for AI Safety" | 非营利组织；four-risk framework；2023 年灭绝声明 |
| CAISI | "US government AI safety" | NIST 中心；自愿协议；非机密评估 |
| Four-risk framework | "CAIS 的 taxonomy" | 恶意使用、AI 竞赛、组织风险、失控 AI |
| Malicious use | "坏行为者使用 AI" | 生物武器、虚假信息、网络攻击 |
| AI races | "竞争性压力" | 实验室/公司/国家推动部署越过安全点 |
| Organizational risk | "实验室内部失败" | 安全文化、审计、防御、信息安全 |
| Rogue AI | "不对齐 agent" | 追求与人类福利冲突目标的有能力 AI |
| California SB-53 | "州级法规" | 2025–2026 年法案；如果签署，美国首个州级灾难性风险法规 |

## 延伸阅读

- [Center for AI Safety](https://safe.ai/) — four-risk framework 的机构主页。
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — four-risk 论文。
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — 简短联合声明。
- [NIST CAISI](https://www.nist.gov/caisi) — 政府面向的 AI 标准和创新中心。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将实验室级承诺连接到社会规模框架。
