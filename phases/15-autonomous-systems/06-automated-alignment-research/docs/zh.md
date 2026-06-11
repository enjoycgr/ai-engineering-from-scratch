# 自动化对齐研究（Anthropic AAR）

> Anthropic 在独立的沙箱中并行运行 Claude Opus 4.6 自动化对齐研究团队，通过一个共享论坛进行协调，论坛日志位于任何沙箱之外（因此智能体无法删除自己的记录）。在弱到强训练（weak-to-strong training）问题上，AAR 的表现超过了人类研究人员。Anthropic 自己的总结指出，规定的工作流程（prescribed workflows）往往会限制 AAR 的灵活性并降低性能。自动化对齐研究是一个压缩步骤，它将时间线压缩到 RSP 旨在检测的确切错位风险。

**类型：** 学习
**语言：** Python（标准库，并行研究论坛模拟器）
**前置要求：** Phase 15 · 05（AI Scientist v2），Phase 15 · 04（DGM）
**时间：** ~60 分钟

## 问题

对齐研究在人类研究者的时间上非常昂贵。可扩展监督（scalable oversight）、奖励规范（reward specification）或弱到强训练等问题需要每次迭代耗时数周的实验。随着前沿能力的进步，对齐工作负载的增长速度超过了合格研究人员的供给速度。

自动化对齐研究（AAR）问道：那些能力正在超越对齐的前沿模型，是否也能为缩小差距做出贡献。Anthropic 2026 年关于 AAR 运行的弱到强训练研究的报告（alignment.anthropic.com/2026/automated-w2s-researcher/）是这类部署系统中首批公开结果之一。

结果是真正积极的：AAR 解决了一个比人类基线更好的研究问题。这个结果也将本阶段发展的每一个担忧集中在一个系统中。如果对齐研究可以自动化，那么妥协保障措施的部分也可以。RSP 的 AI R&D 能力阈值就是针对这个循环编写的。

## 概念

### 架构（如公开描述）

- **并行智能体。** 多个 Claude Opus 4.6 实例作为 AAR 运行，每个在独立的沙箱中。沙箱限制文件系统、网络和进程面。
- **共享论坛。** 发现发布到一个共享论坛，其存储后端位于每个沙箱之外。智能体可以阅读论坛；智能体不能从自己的沙箱内删除或编辑论坛记录。
- **任务分配。** 人工设计的任务队列将研究问题分配给 AAR。AAR 提出分解、运行实验、撰写发现。
- **弱到强训练焦点。** 发表的研究解决了弱到强训练——使用较弱的监督者模型训练较强的学生模型。AAR 在这项任务上超过了人类研究人员。

### 为什么沙箱外日志很重要

日志完整性属性是受控实验与可能被妥协实验之间的区别。如果 AAR 可以编辑自己的日志，它们可以掩盖失败或可疑的实验。Anthropic 直接指出了这一点：该属性是信任研究输出的必要条件，且验证它并非易事。

基础设施术语中的类比是带写透到外部存储的仅追加日志（例如 S3 对象锁定、不可变的 Datadog 日志）。智能体没有修改过去记录的凭证。这与 2026 年企业安全团队用于制作防勒索软件审计跟踪的相同模式。

### 规定工作流程的权衡

Anthropic 的报告指出，规定的工作流程往往会限制 AAR 的灵活性并降低性能。换句话说：如果人类设计者逐步编写研究计划，AAR 的表现比给予高级目标和自由分解时差。这与智能体研究的跨结果一致（LangGraph、Microsoft 的 Agent Framework）——过度指定的脚手架会损害有能力的智能体。

安全含义是微妙的。提高 AAR 性能的灵活性也允许未经授权的分解、新颖的工具使用和目标漂移。这是一个设计空间决策：多少灵活性足以购买击败人类的能力，而不给智能体空间将目标泛化到分布外？

### 压缩风险

RSP v3.0（第 19 课）引入了一个 AI R&D 能力阈值：以与人类 + AI 工具相比具有竞争力的成本完全自动化 AI R&D 管道的能力。DeepMind 的 FSF v3 包含一个类似的 ML R&D 自主性级别。两个框架都将此阈值视为触发加强控制的开关。

AAR 距离阈值还差一步：它自动化了管道的一部分（特定、范围明确任务上的对齐研究），但不是端到端的能力开发循环。时间线问题是差距缩小的速度。

压缩的时间线是复合故障的担忧。如果对齐研究和能力研究以相似的速度复合，错位风险面至少与能力一样快地增长。如果能力复合得更快（历史趋势），差距就会扩大。这是 AAR 作为有条件的好事的论据：每个额外的对齐结果都会缩小差距，当且仅当研究过程是可信的。

### AAR 不能替代什么

人类研究人员设定任务队列、审查结果并持有宪法权威。AAR 加速管道的中间部分，而非两端。Anthropic 的发表输出包括 AAR 贡献和人类研究者对发表什么、撤回什么和提炼什么的判断。

这与第 15 课的 propose-then-commit 模式应用于研究本身相匹配：AAR 提议；人类提交。

## 使用它

`code/main.py` 模拟一个多 AAR 研究论坛。三个独立的"研究智能体"并行运行，将发现发布到一个带可选篡改尝试的仅追加论坛。

观察：

- 沙箱外日志如何捕捉日志篡改尝试。
- 自由分解的任务分配如何在更难的问题上优于固定工作流程。
- 如果日志不是防篡改的，单个被妥协的 AAR 如何污染论坛的信号。

## 交付它

`outputs/skill-aar-deployment-review.md` 是自动化对齐研究管道的预部署审查。

## 练习

1. 运行 `code/main.py`。比较"固定工作流程"与"自由分解"设置。自由分解总是赢，还是有问题类别中固定工作流程具有竞争力？

2. 修改模拟器，使一个智能体尝试日志篡改。确认仅追加日志检测到它。写一段话描述检测签名在日志中的确切样子。

3. 阅读 Anthropic 的弱到强 AAR 报告。识别 AAR 击败人类研究者的具体子任务。什么使其适合自动化？

4. 设计一个平衡 AAR 灵活性（更好结果）与规定工作流程约束（更容易审计）的任务队列分配策略。描述你将如何 A/B 测试两者。

5. 阅读 RSP v3.0 的 AI R&D-4 阈值。用一段话描述你认为什么会跨越它而 AAR 目前不会。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|---|---|---|
| AAR | "自动化对齐研究者" | 在对齐问题上自主运行的 Claude Opus 4.6 实例 |
| Weak-to-strong training | "用较弱监督者训练较强模型" | AAR 超越人类的经典可扩展监督基准 |
| Shared forum | "智能体发布发现的地方" | 仅追加、沙箱外存储 |
| Out-of-sandbox log | "智能体无法编辑自己的记录" | 防篡改写透到外部存储 |
| Prescribed workflow | "人类设计者的一步一步计划" | 限制 AAR；通常比自由分解表现差 |
| Free decomposition | "智能体决定如何分解任务" | 更有能力，更难审计 |
| AI R&D threshold | "RSP/FSF 能力级别" | 以竞争性成本完全自动化 R&D 管道 |
| Compressed timeline | "对齐与能力的竞赛" | 如果能力复合速度超过对齐，错位风险增长 |

## 延伸阅读

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — 主要来源。
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — AI R&D 阈值框架。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 更广泛的智能体自主性框架。
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — 与 RSP 平行的 ML R&D 自主性级别。
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — AAR 攻击的底层问题。
