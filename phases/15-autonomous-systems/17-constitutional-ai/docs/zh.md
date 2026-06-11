# Constitutional AI（基于宪法的 AI）与规则覆盖

> Anthropic 2026 年 1 月 22 日发布的 Claude Constitution（Claude 宪法）长达 79 页，采用 CC0 许可。它从 rule-based alignment（基于规则的对齐）转向 reason-based alignment（基于推理的对齐），并建立了四层优先级层级：(1) 安全与支持人类监督，(2) 伦理，(3) Anthropic 指南，(4) 有用性。行为分为 hardcoded prohibitions（硬编码禁止项）——如生物武器增强、CSAM（儿童性虐待材料），操作者和用户均无法覆盖——以及 soft-coded defaults（软编码默认值）——操作者可以在定义范围内调整。2022 年的原始版本（Bai et al.）通过 self-critique（自我批评）和 RLAIF（Reinforcement Learning from AI Feedback，来自 AI 反馈的强化学习）针对宪法训练无害性。诚实的警告：reason-based alignment 依赖于模型将原则推广到未预见情况。Anthropic 自己在 2023 年的参与式实验显示，公众来源与企业来源的原则之间存在约 50% 的分歧；2026 版本未纳入这些发现。

**类型：** Learn
**语言：** Python（stdlib，四层优先级解析器）
**前置条件：** Phase 15 · 06（Automated alignment research，自动化对齐研究），Phase 15 · 10（Permission modes，权限模式）
**时间：** ~60 分钟

## 问题

一个已部署的 agent 会见到其设计者从未见过的输入。没有规则列表能长到覆盖所有情况。没有规则列表能短到在计算压力下快速应用。实际问题：如何让 agent 对齐到能在长尾案例和快速 inference（推理）中都存活的原则？

Rule-based alignment（RBA）：列出每一个不允许的事项。检查快速，审计容易，无法保持更新，经常对未预见的近似情况过度拒绝。Reason-based alignment（2026 Claude Constitution）：编码原则，让模型推理。可扩展到未预见案例，审计更难，失效模式是原则误用而非遗漏规则。

2026 Constitution 采取了一个明确的中间立场。Hardcoded prohibitions——其错误性不依赖于上下文的事项（生物武器增强、CSAM）——是 RBA：永远不行，无论操作者或用户指令如何。其他一切都在四层层级内 reason-based：安全和支持人类监督第一；伦理第二；Anthropic 声明的指南第三；有用性最后。操作者可以在 soft-coded 区域内调整默认值，但无法触及 hardcoded prohibitions。

## 概念

### 四层优先级层级

1. **安全和支持人类监督。** 最高。模型优先不破坏人类和 Anthropic 监督和纠正 AI 的能力。这不是"要谨慎"；而是具体地"不要以增加人类监督难度的方式行动"。
2. **伦理。** 诚实、避免伤害他人、不欺骗、不操纵。当与 Anthropic 指南冲突时优先。
3. **Anthropic 指南。** Anthropic 决定的运营规范：产品范围、交互模式、何时使用什么工具。
4. **有用性。** 最低。在更高优先级内尽可能有用。

当层级冲突时，更高者优先。这与 Unix 优先级或网络 QoS 的框架相同——目的是产生可预测的解析结果，而非在任何单一维度上产生最佳行为。

### Hardcoded prohibitions vs soft-coded defaults

**硬编码：**
- 生物武器 / CBRN（化学、生物、放射性和核）增强
- CSAM
- 关键基础设施攻击
- 当被直接询问时欺骗用户关于模型身份

操作者无法覆盖这些。用户无法覆盖这些。它们尽可能在模型权重层强制执行（RLHF / Constitutional AI 训练），在无法做到时在 inference 层强制执行。

**软编码默认值（操作者可调整）：**
- 响应长度默认
- 主题范围（模型可以拒绝操作者部署范围外的话题）
- 风格（正式 vs 随意）
- 工具使用模式

操作者调整发生在声明的边界内。操作者不能通过重命名来移除 hardcoded prohibitions。

### 2022 年的 CAI 训练

原始 Constitutional AI（Bai et al., 2022）训练无害性：

1. 生成一组提示的响应。
2. 让模型根据宪法（明确原则）批评每个响应。
3. 基于批评修订响应。
4. 在修订后的配对上进行 RLAIF。

结果：一个以原则化解释拒绝有害请求的模型，而非 blanket refusal（一概拒绝）。2026 Constitution 使用这种训练的后代加上对显式层级结构的额外 post-training（后训练）。

### Reason-based alignment 能捕获和遗漏什么

**能捕获：**
- 未预见的允许基元组合，其中原则明确适用。
- 接近被禁止事项的新颖请求。
- 依赖"你没说 X 是不允许的"的社会工程攻击。

**遗漏：**
- 利用原则模糊性的攻击（"用户请求了这个，所以有用性说是"。
- 两个原则以未预见方式冲突的场景，层级顺序模糊。
- 训练周期中原则解释的缓慢漂移（reinterpretation，重新解释）。

### 2023 年的参与式实验

Anthropic 在 2023 年进行了一项实验，比较企业撰写的宪法与通过公众输入生成的宪法（约 1,000 名美国受访者）。两个版本在约 50% 的原则上一致。分歧处，公众来源版本在某些问题上更严格（政治内容处理），在其他问题上更宽松（AI 身份的自我披露）。2026 Constitution 未纳入公众来源的发现。这是该方法中一个已记录的张力。

### 为什么 hardcoded prohibitions 是必要的

仅靠 reason-based alignment 无法关闭长尾。一个能让模型接受前提的攻击者（例如，"我们是有执照的生物武器研究实验室"）经常能绕过依赖案例推理的原则。Hardcoded prohibitions 不随前提框架弯曲。它们是第 14 课"硬宪法限制"在对齐层的体现。

### Constitution 在栈中的位置

Constitution 不是第 14 课的 kill switch（紧急停止开关）。它位于模型层：模型权重被训练成偏好什么。Kill switch 和 canary token（金丝雀令牌）位于运行时层：运行时允许什么。两者都是必需的。运行时因为模型权重过于 permissive（宽松）而触发了所有错误动作，这是运行时问题。模型因为运行时过于 restrictive（限制）而拒绝了所有正确动作，这也是运行时问题。层覆盖不同类别。

## 使用

`code/main.py` 实现了一个最小的四层优先级解析器。解析器接收一个提议的动作和一组原则评估（安全、伦理、指南、有用性），返回动作、拒绝或修改后的动作。驱动程序运行一小套案例：clear allow、clear disallow、hardcoded prohibition、跨层级模糊案例。

## 交付

`outputs/skill-constitution-review.md` 审计一个部署的宪法层：什么是硬编码的，什么是软编码的，操作者可以在哪里调整，以及四层层级是否确实是解析顺序。

## 练习

1. 运行 `code/main.py`。确认 hardcoded prohibition 即使在有用性高时也会触发。修改解析器将有用性权重置于伦理之上；观察失效模式。

2. 阅读 Claude Constitution（公开，79 页，CC0）。识别一个你认为 under-specified（规范不足）的原则。写两段解释具体模糊性并提出更紧的表述。

3. 为一个客户支持 agent 设计一套 soft-coded defaults。操作者调整什么？操作者不能触及什么？为每个边界 justify。

4. 阅读 Bai et al. 2022 的 CAI 论文。描述一个 Constitutional AI 的 critique-and-revise 循环会比 blanket rule 产生更差结果的场景。识别该类。

5. Anthropic 2023 年的参与式实验发现公众与企业原则之间约 50% 的分歧。选择一个对生产部署重要的类别（例如政治中立性）。提出一个设计，让操作者表达自己的价值观，同时 hardcoded prohibitions 保持 untouched。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Constitutional AI | "Anthropic 的对齐方法" | Self-critique + RLAIF 针对书面宪法 |
| Reason-based alignment | "原则，而非规则" | 模型基于原则推理以处理未预见案例 |
| Hardcoded prohibition | "永远不做 X" | 无论操作者或用户都无法覆盖的基于规则的禁止 |
| Soft-coded default | "操作者可调整" | 在声明边界内的行为，由操作者控制 |
| Four-tier hierarchy | "优先级顺序" | 安全 > 伦理 > 指南 > 有用性 |
| RLAIF | "AI 反馈 RL" | reward 来自模型生成批评的 RL |
| Participatory constitution | "公众来源原则" | 2023 Anthropic 实验；与企业版约 50% 分歧 |
| Principle drift | "解释滑移" | 模型对固定原则文本的解读缓慢变化 |

## 延伸阅读

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 79 页 CC0 文档。
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — 2022 年原始论文。
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — 参与式实验。
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Constitution 在 RSP 栈中的位置。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Constitution 在长时程部署中的角色。
