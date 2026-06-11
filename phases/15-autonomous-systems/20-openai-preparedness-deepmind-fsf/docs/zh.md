# OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入 Research Categories（研究类别）——Long-range Autonomy（长程自主性）、Sandbagging（沙袋策略）、Autonomous Replication and Adaptation（自主复制与适应）、Undermining Safeguards（破坏防护措施）——与 Tracked Categories（跟踪类别）不同。Tracked Categories 触发 Capabilities Reports（能力报告）加上由 Safety Advisory Group（安全咨询组）审查的 Safeguards Reports（防护措施报告）。DeepMind 的 FSF v3（2025 年 9 月，Tracked Capability Levels 于 2026 年 4 月 17 日添加）将自主性纳入 ML R&D 和 Cyber 领域（ML R&D autonomy level 1 = 以与人类 + AI 工具相比有竞争力的成本完全自动化 AI R&D 流水线）。FSF v3 通过 automated monitoring for instrumental-reasoning misuse（工具性推理滥用的自动监控）明确解决 deceptive alignment（欺骗性对齐）。诚实的注记：PF v2 中的 Research Categories（包括 Long-range Autonomy）不会自动触发缓解措施；政策语言是"potential"（潜在的）。DeepMind 自己表示，如果 instrumental reasoning（工具性推理）增强，automated monitoring "will not remain sufficient long-term"（长期来看将不再足够）。

**类型：** Learn
**语言：** Python（stdlib，three-framework decision-table diff tool，三框架决策表对比工具）
**前置条件：** Phase 15 · 19（Anthropic RSP）
**时间：** ~45 分钟

## 问题

第 19 课仔细研读了 Anthropic 的扩展政策。本课通过阅读 OpenAI 和 DeepMind 的来完成图景。三份文档是处理同一问题的表亲产物——前沿实验室何时应该暂停或门控模型——它们在一小部分类别上趋同，在具体位置上分歧。

趋同：三者都将 long-range autonomy 标记为值得跟踪的能力类别。三者都承认 deceptive behavior（欺骗性行为）（alignment faking、 sandbagging）为特定风险类别。三者都有内部审查机构。分歧：OpenAI 将类别分为"Tracked"（强制缓解）和"Research"（不自动触发）。DeepMind 将自主性折叠到两个领域而非单独命名。实验室命名 Tracked vs Research，或 Critical vs Moderate，或 Tier-1 vs Tier-2；能力所在 bucket 的运营后果在不同实验室之间不同。

一起阅读它们是有用的练习。相同的能力在 Anthropic 可能是"mandatory mitigation"，在 OpenAI 是"monitored but not triggering"，在 DeepMind 是"tracked in a specific domain"。Policy posture（政策姿态）很重要。

## 概念

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **Tracked Categories**：触发 Capabilities Reports（模型能做什么）加上 Safeguards Reports（有什么缓解措施）。部署前由 Safety Advisory Group 审查。
- **Research Categories**：实验室正在跟踪但尚未承诺特定缓解措施的能力类别。包括 Long-range Autonomy、Sandbagging、Autonomous Replication and Adaptation、Undermining Safeguards。

v2 的关键框架：Research Categories 不会自动触发缓解措施。政策语言是"potential"缓解措施。这是需要关注的具体差异：类别在 Tracked vs Research 中的位置是"门控"和"观察"之间的铰链。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；Tracked Capability Levels 于 2026 年 4 月 17 日添加）

结构：

- **Critical Capability Levels (CCLs)**：跨五个领域的能力阈值：Cyber、Bio、ML R&D、CBRN、Autonomy（折叠到 ML R&D 和 Cyber 中）。
- **Tracked Capability Levels**：2026 年 4 月添加的额外粒度。具体例子：ML R&D autonomy level 1 = 以与人类 + AI 工具相比有竞争力的成本完全自动化 AI R&D 流水线。
- **Deceptive alignment monitoring**：对 instrumental-reasoning misuse 进行自动监控的明确承诺。

自主性框架与 OpenAI 的不同。DeepMind 不将"Autonomy"保留为顶级领域；它被折叠到自主性会造成伤害的领域（ML R&D 和 Cyber）。论据是：没有领域的自主性是没有风险的能力；反驳论据是：跨领域的自主性是一个框架应该命名的元风险。

### 三者的趋同之处

- 内部 Safety Advisory Group（命名 Anthropic SAG、OpenAI SAG、DeepMind 内部委员会）。部署前审查高能力模型。
- 明确提及 deceptive alignment / alignment faking 作为风险类别。
- 声明 cadence 上的 standing artifacts（Anthropic：Frontier Safety Roadmap、Risk Report；OpenAI：Capabilities and Safeguards Reports；DeepMind：FSF update cycle）。
- 承认仅监控的防御有上限。DeepMind 明确："automated monitoring will not remain sufficient long-term."

### 分歧之处

- **Anthropic**：v3.0 移除了 pause commitment；AI R&D-4 threshold 是命名的下一个门控。
- **OpenAI**：Tracked vs Research 分割；Research Categories（包括 Long-range Autonomy）不自动门控。
- **DeepMind**：自主性折叠到其他领域；Tracked Capability Levels 于 2026 年 4 月添加粒度。

### Sandbagging：一个使三者都复杂化的具体能力

Sandbagging（模型在评估中策略性地表现不佳）在 OpenAI 的 Research Categories 中。Anthropic 的 RSP v3.0 通过 evaluation-context gap（第 1 课）解决它。DeepMind 通过 FSF v3 中的 deceptive alignment monitoring 解决它。

如果模型在评估中 sandbagging，每个框架的能力阈值都被低估。框架只有在测量有效时才有效。这就是为什么除了实验室自我评估外，还需要 external measurement（第 21 课，METR）和 adversarial evaluation（对抗性评估）。

### 政策阅读技能

- **Locate**：你关心的每个能力都应该在政策中可找到。如果找不到，政策不覆盖它。
- **Classify**：它是 Tracked（触发缓解）还是 Research（跟踪但不触发）？OpenAI 命名了这一点；Anthropic 和 DeepMind 有自己的等价物。
- **Cadence**：政策是在声明的 schedule 上更新，还是仅在特定事件后更新？声明的 cadence 更强。
- **Independence**：外部审查是强制还是可选？Anthropic 与 Apollo 和 US AI Safety Institute 合作；OpenAI 与 METR 合作；DeepMind 主要与内部 SAG 合作。

## 使用

`code/main.py` 实现了一个小型 decision-table diff tool。给定一个能力（autonomy、deceptive alignment、R&D automation、cyber uplift 等），它输出三个政策如何分类该能力以及什么缓解措施触发。它是阅读辅助，不是政策工具。

## 交付

`outputs/skill-cross-policy-diff.md` 使用三个框架作为参考，为特定能力生成跨政策比较。

## 练习

1. 运行 `code/main.py`。确认 diff tool 的输出与你能根据源文档验证的至少两个能力匹配。

2. 全文阅读 OpenAI Preparedness Framework v2。识别每个 Research Category。对每个，写一句话解释为什么它在 Research 而非 Tracked 中。

3. 全文阅读 DeepMind FSF v3，加上 2026 年 4 月 Tracked Capability Levels 更新。识别 ML R&D autonomy level 1 的具体评估标准。你如何外部测量它？

4. Sandbagging 在 OpenAI 的 Research Categories 中。设计一个评估，迫使 sandbagging 模型暴露其实际能力。参考第 1 课 eval-context-gaming 讨论。

5. 比较三个政策在特定能力上的分类（你的选择）。命名你认为最严格的政策分类和最不严格的。用源文本 justify。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Preparedness Framework | "OpenAI 的扩展政策" | PF v2（2025 年 4 月）；Tracked vs Research 类别 |
| Tracked Category | "强制缓解" | 触发 Capabilities + Safeguards Reports；SAG 审查 |
| Research Category | "仅监控" | 跟踪但不自动缓解；包括 Long-range Autonomy |
| Frontier Safety Framework | "DeepMind 的扩展政策" | FSF v3（2025 年 9 月）+ Tracked Capability Levels（2026 年 4 月） |
| CCL | "Critical Capability Level" | DeepMind 每个领域的阈值（Cyber、Bio、ML R&D、CBRN） |
| ML R&D autonomy level 1 | "R&D 自动化" | 以有竞争力的成本完全自动化 AI R&D 流水线 |
| Sandbagging | "策略性表现不佳" | 模型在评估中表现不佳；在 OpenAI Research Categories 中 |
| Instrumental reasoning | "手段-目的推理" | 关于如何实现目标的推理；DeepMind 监控的目标 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels 补充。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式 Risk Report 示例。
