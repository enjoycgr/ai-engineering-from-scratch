# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，替代了 2023 年政策。Two-tier mitigation（双层缓解）：Anthropic 将单方面做什么 vs 作为 industry-wide recommendation（全行业建议）框架的内容（包括 RAND SL-4 安全标准）。新增 Frontier Safety Roadmaps 和 Risk Reports 作为 standing documents（常备文件）而非一次性交付物。删除了 2023 年的 pause commitment（暂停承诺）。引入 AI R&D-4 threshold：一旦越过，Anthropic 必须发布 affirmative case（正面论证），识别 misalignment risks（不对齐风险）和缓解措施。Claude Opus 4.6 未越过该阈值。Anthropic 在 v3.0 公告中声明" confidently ruling this out is becoming difficult"。SaferAI 对 2023 RSP 的评级为 2.2；他们将 v3.0 降级至 1.9，将 Anthropic 放入"weak"（弱）RSP 类别，与 OpenAI 和 DeepMind 并列。Qualitative thresholds（定性阈值）替代了 2023 年的 quantitative commitments（定量承诺）；移除 pause clause 是最明显的 regression（退步）。

**类型：** Learn
**语言：** Python（stdlib，RSP threshold decision engine，阈值决策引擎）
**前置条件：** Phase 15 · 06（AAR），Phase 15 · 07（RSI）
**时间：** ~45 分钟

## 问题

前沿实验室发布的 scaling policies（扩展政策）部分是技术文档，部分是治理文档，部分是向监管机构发出的信号。RSP v3.0 是当前 Anthropic 的文档。仔细阅读它很重要，不是因为遵守它具有约束力（事实并非如此），而是因为其框架塑造了实验室如何构想灾难性风险以及如何向公众沟通权衡。

v3.0 vs v2.0 的差异是有用的分析单元。新增：Frontier Safety Roadmaps、Risk Reports、AI R&D-4 threshold。移除：2023 年 pause commitment。重构：two-tier mitigation schedule，分为 Anthropic-unilateral 和 industry-recommendation。外部审查——SaferAI——将评分从 2.2（v2）降至 1.9（v3.0）。这就是 scaling policy 如何在看起来更精致的同时变得不那么严格。

## 概念

### 双层缓解时间表

- **Anthropic unilateral actions**：无论其他实验室做什么，Anthropic 都会做的事。超过阈值时停止训练，特定安全措施，特定部署门控。
- **Industry-wide recommendations**：Anthropic 认为行业应该集体做什么。包括 RAND SL-4 安全标准。这些不是 Anthropic 的承诺；它们是政策倡导。

双层结构不在 v2 中。这意味着读者需要查看每个承诺所在的列。"industry-wide recommendation" 列中的安全措施不是 Anthropic 的承诺；它是 Anthropic 的希望。

### AI R&D-4 threshold

这是 RSP v3.0 命名的重要下一个阈值。具体而言：一个能够以 competitive cost（有竞争力的成本）自动化 substantial fraction（大部分）AI 研究的模型。一旦 Anthropic 认为模型越过它，他们必须在继续扩展之前发布 affirmative case，识别 misalignment risks 和缓解措施。

Claude Opus 4.6 按照 v3.0 公告未越过它。文档补充说："confidently ruling this out is becoming difficult." 这种措辞很重要；它承认阈值足够接近，是一个现实关切，而非推测性限制。

第 6 课（Automated Alignment Research）和第 7 课（Recursive Self-Improvement）直接流入此阈值。自动化对齐研究者越过研究质量门槛是 AI R&D-4 threshold 正在接近的证据。

### Frontier Safety Roadmaps 和 Risk Reports

v3.0 将两种 artifact 提升为 standing documents：

- **Frontier Safety Roadmap**：前瞻性文档，描述计划的安全工作、能力预期和缓解研究。
- **Risk Report**：模型发布后的回顾性文档，描述观察到的能力和残余风险。

两者都是公开的。两者都在声明的 cadence（节奏）上更新。用途是：读者可以追踪 Anthropic 在 Roadmap 中说他们会做什么，与 Risk Report 中报告的内容相比如何。

### 移除 pause clause

2023 RSP 包含明确的 pause commitment：如果模型越过特定能力阈值，训练将暂停，直到缓解措施到位。v3.0 将明确的暂停替换为更软的表述（发布 affirmative case，如果缓解措施足够则继续）。SaferAI 和其他分析师直接指出这是新文档中最强的 regression。

政策变更的论据：2023 年的定量阈值到 2026 年的能力 benchmark 证明是不可达到的，因为 benchmark 本身被重新调整了。反驳论据：scaling policy 中的 pause clause 是一种 commitment device（承诺机制）；移除它削弱了政策的可信度。

### SaferAI 的降级

SaferAI 是一个独立组织，评级 RSP 风格文档。他们的公开评级：2023 Anthropic RSP 得分 2.2（满分 4.0 是当前最佳 RSP，1.0 是名义上的）。v3.0 得分 1.9。这将 Anthropic 从"moderate"（中等）移至"weak"（弱），与 OpenAI 和 DeepMind 同属弱类别。

SaferAI 降级因素：
- 定性阈值替代了定量阈值。
- Pause commitment 被移除。
- AI R&D-4 threshold 的缓解措施被描述为"affirmative case"而非具体措施。
- 审查机制依赖 Anthropic 的 Safety Advisory Group，独立监督有限。

### 本课不是什么

这不是合规课。RSP v3.0 不是法规；没有什么强制 Anthropic 遵守它。本课是带着应有的具体性和怀疑性阅读文档。Scaling policies 是前沿实验室关于灾难风险姿态的主要公开信号。好好阅读它们是一项实用技能，适用于任何工作依赖前沿能力的人。

## 使用

`code/main.py` 实现了一个小型 decision engine，镜像 RSP threshold-evaluation 的形状：给定一个候选模型和一组能力测量，返回 AI R&D-4 threshold 是否被越过、所需的 affirmative-case 部分以及是否可以部署。它故意简单；目的是让文档的逻辑显式化。

## 交付

`outputs/skill-scaling-policy-review.md` 根据 v3.0 参考审查 scaling policy（Anthropic、OpenAI、DeepMind 或内部）：two-tier 结构、thresholds、pause commitments、independent review。

## 练习

1. 运行 `code/main.py`。输入三个不同能力水平的合成模型。确认 threshold evaluator 行为符合预期并产生正确的 affirmative-case 模板。

2. 全文阅读 RSP v3.0（32 页）。识别每一个位于"industry-wide recommendation" tier 的承诺。哪些承诺在 v2 中会是"Anthropic unilateral"？

3. 阅读 SaferAI 的 RSP 评级方法。将他们的 rubric（评分标准）应用于文档，复现他们对 v3.0 的 1.9 分。哪一行 rubric 最推动了降级？

4. 2023 年 pause commitment 被移除。提出一个替代承诺，在保留政策可信度的同时承认 2026 年 benchmark-rescaling 问题。

5. 比较 RSP v3.0 和 OpenAI Preparedness Framework v2（第 20 课）。选择一个 v3.0 更强的领域。选择一个 Preparedness Framework 更强的领域。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| RSP | "Anthropic 的扩展政策" | Responsible Scaling Policy；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | "研究自动化阈值" | 以有竞争力的成本自动化大部分 AI 研究的能力 |
| Affirmative case | "安全论证" | 公开发布的论点，说明风险已被识别且缓解措施充分 |
| Frontier Safety Roadmap | "前瞻计划" | 关于计划安全工作和预期能力的常备文件 |
| Risk Report | "模型回顾" | 模型发布后关于观察能力和残余风险的常备文件 |
| Two-tier mitigation | "单方面 vs 行业" | Anthropic 承诺 vs 行业建议，分开列出 |
| Pause commitment | "2023 条款" | 明确承诺暂停训练；在 v3.0 中被移除 |
| SaferAI rating | "独立 RSP 评级" | 第三方 rubric；v3.0 得分 1.9（v2 为 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页政策。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — v2 变更摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — 从 RSP v3.0 链接的常备文件。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前前沿模型的回顾。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 连接到测量的自主性。
