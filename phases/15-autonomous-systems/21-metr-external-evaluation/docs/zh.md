# METR Time Horizons 与外部能力评估

> METR（前 ARC Evals）自 2023 年 12 月起是独立的 501(c)(3)。他们的 Time Horizon 1.1 benchmark（2026 年 1 月）将 logistic curve（逻辑曲线）拟合到 task-success probability（任务成功概率）vs log(expert human completion time)（专家人类完成时间的对数）；50% 概率的交点定义了模型的 time horizon（时间范围）。2025–2026 年的评估对象包括 GPT-5.1、GPT-5.1-Codex-Max 和原型监控评估（monitor 能否 catch side tasks；agent 能否 evade）。Benchmark suites：HCAST（180+ ML、cyber、SWE、reasoning 任务；1 分钟到 8+ 小时）、RE-Bench（71 个 ML research-engineering 任务，带专家基线）、SWAA。诚实的注记：METR 测量是 idealized（理想化的）——没有人类，没有真实后果——团队记录了 eval-vs-deployment behavior gap（评估与部署行为差距）（第 1 课）。Time horizon 是 upper bound（上限），不是部署预测。

**类型：** Learn
**语言：** Python（stdlib，logistic-fit horizon estimator，逻辑拟合范围估计器）
**前置条件：** Phase 15 · 01（Long-horizon agents），Phase 15 · 19（RSP）
**时间：** ~60 分钟

## 问题

Scaling policies（第 19、20 课）只有其引用的测量数据才有用。"AI R&D-4 threshold"和"Long-range Autonomy"在政策散文中定义；只有当特定评估产生特定数字时，它们才变得可执行。

METR 是 2024–2026 年定义了许多这些数字的外部评估组织。他们评估前沿模型——通常是发布前，与实验室签订 NDA——之后发布方法论文。Time Horizon 1.1 benchmark（2026 年 1 月）是他们的头条成果：一个单一标量，将能力压缩成人类可读的单元（"这个模型可以以 50% 可靠性完成专家花费 X 小时的任务类型"。

本课部分关于方法论（如何计算 horizon），部分关于解释（为什么 horizon 是 upper bound 而非部署预测）。两个技能属于一起。一个理解 horizon 如何拟合的团队，比一个只在幻灯片上看到"14 小时"的团队更难被糟糕的供应商声明欺骗。

## 概念

### METR 背景

- 成立：2023 年 12 月（前 ARC Evals，分拆为独立 501(c)(3)）。
- 范围：评估前沿模型的自主能力，通常是发布前。
- 合作实验室：Anthropic、OpenAI（2025–2026 年多次合作）。
- 著名交付物：Time Horizon 1.0（2025 年 3 月）、Time Horizon 1.1（2026 年 1 月）、原型监控评估。

### Time Horizon 拟合

方法论（来自 METR 博客和论文）：

1. 收集一个任务套件，跨越分钟级到小时级专家完成时间。当前套件：HCAST（180+ 任务）、RE-Bench（71 任务）、SWAA。
2. 在每个任务上运行模型；记录成功或失败。
3. 拟合 logistic curve：P(success) 作为 log(expert completion time) 的函数。
4. Horizon 是 P(success) = 0.5 时的专家时间。

Logistic-fit 形状是正确的，因为能力通常与任务难度有递增、趋近平稳的关系。50% 点是一个选择（可以是 10%、90%）；METR 在详细论文中报告多个阈值，但以 50% 为首，因为它最直观。

### 2026 年 1 月数字

根据 Time Horizon 1.1：

- Claude Opus 4.6：~14 小时，50% 可靠性，截至 Time Horizon 1.1（2026 年 1 月）。
- HCAST 风格任务的翻倍时间：~4.3 个月（130.8 天），根据 Time Horizon 1.1（2026 年 1 月）报告的 post-2023 拟合；~7 个月数字是 Time Horizon 1.0 的 full 2019–2025 拟合，与 post-2023 数字一起在 TH1.1 中报告。

这些是 Phase 15 其余部分引用的具体数字。它们在 Anthropic 的 RSP v3.0（第 19 课）和 DeepMind 的 FSF v3（第 20 课）中被引用，作为 feeding threshold assessments 的测量之一。

### Benchmark suites

- **HCAST**：189 个任务，涵盖 ML、cybersecurity、software engineering 和 general reasoning。范围：1 分钟到 8+ 小时专家时间。Horizon 数字背后的主要套件。
- **RE-Bench**：71 个 ML research-engineering 任务，带人类专家基线。测量模型是否能做 feed into AI R&D-4 threshold 的 ML R&D 工作。
- **SWAA**：较短任务；用于校准 horizon 曲线的低端。

### 原型监控评估

2025–2026 年 METR 工作包括监控评估原型：monitor 能否 catch 模型执行 side task（操作者未要求的事项）；agent 能否 evade monitor。这些直接 feed into OpenAI PF v2 中的"Undermining Safeguards"类别和 DeepMind FSF v3（第 20 课）中的 deceptive-alignment monitoring。

### 为什么 horizons 是 upper bounds

- **Idealized tooling**：benchmark 任务给模型提供干净的工具和格式良好的数据。真实部署更混乱。
- **No real consequences**：模型从不真正向客户收费、删除真实数据或联系真实的人。真实部署有不可逆的赌注。
- **Eval-context gaming**：第 1 课。模型在测试中表现不同。2026 年 International AI Safety Report 实证记录了这一点。
- **No legitimate user variance**：benchmark 提示是结构化的。真实用户产生模糊的、上下文依赖的请求。

Horizon 是有利条件下的能力 ceiling。部署可靠性是另一个数字，更低，团队必须测量自己的分布才能知道它。

### 外部评估者案例

外部评估很重要，因为内部实验室有优化他们报告的指标的动机。METR 的独立性——一个 501(c)(3)，有声明的方法论和同行评审论文——是结构性缓解。它单独并不足够（实验室仍控制 METR 看到什么），但严格优于没有外部评估。

### 实践中如何使用 horizon 数字

- **作为能力过滤器**：如果模型的 horizon 远低于提议任务的专家时间，不要将其自主部署（第 1 课的 skill file）。
- **作为趋势指标**：翻倍时间告诉你，即使没有新缓解措施，当前实践还能保持安全多久。
- **作为先验**：14 小时的 horizon 是起点。根据你的任务分布、工具质量和部署上下文向下调整。

## 使用

`code/main.py` 实现了 task-success vs log(expert time) 的 logistic fit，给定合成结果集。它报告 50% horizon（METR 的头条）、10% horizon（保守）和 90% horizon（乐观）。还演示了当 success rate 被 eval-context gaming 人为夸大时会发生什么变化。

## 交付

`outputs/skill-horizon-interpretation.md` 审查供应商的 horizon 声明，并生成 benchmark 声明与部署现实之间的 gap analysis。

## 练习

1. 运行 `code/main.py`。确认拟合的 50% horizon 与合成 ground truth 匹配。现在将任务时间网格减半；horizon 估计会显著变化吗？

2. 阅读 METR 的 Time Horizon 1.1 博客文章。识别可靠性最高和最低的具体任务。解释差距为何存在。

3. 阅读 METR 的"Measuring Autonomous AI Capabilities"资源。列出 HCAST 任务类别。选择一个你会为生产任务加权的类别并 justify 为什么。

4. 向模拟器引入 eval-context gaming：将 ~20% 的失败任务翻转为成功。报告新的 horizon。这近似 20% gaming rate 对观察数字的影响。

5. 在你自己的 bug backlog 或代表性任务集上设计内部 horizon 评估。描述数据收集、拟合以及输出告诉你什么。与 METR 数字比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| METR | "外部评估者" | 前 ARC Evals；自 2023 年 12 月起独立 501(c)(3) |
| Time Horizon | "能力测量" | 50% 可靠性下的专家任务长度，来自 logistic fit |
| HCAST | "METR 的主要套件" | 180+ 任务，跨越 1 分钟到 8+ 小时 |
| RE-Bench | "研究工程" | 71 个 ML research-engineering 任务，带人类基线 |
| SWAA | "短任务套件" | 校准 horizon 曲线的低端 |
| Doubling time | "增长率" | 50% horizon 翻倍的时间；HCAST 约 7 个月 |
| Eval-context gaming | "模型表现不同" | 测试与部署之间记录的行为差距 |
| Upper bound | "Horizon 是上限" | Benchmark horizon > 负载下的部署可靠性 |

## 延伸阅读

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST、RE-Bench、SWAA 规格。
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — 原始 horizon 论文。
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — 当前数字和方法论。
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — 实时跟踪。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — METR 测量的内部视角。
