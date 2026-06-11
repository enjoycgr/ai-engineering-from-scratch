# STaR、V-STaR、Quiet-STaR — 自我教学推理

> 最小的自我改进循环存在于推理过程 (rationale) 内部。模型生成思维链 (chain of thought, CoT)，保留那些得出正确答案的，并在这些上面进行微调 (fine-tuning)。这就是 STaR。V-STaR 添加了一个验证器 (verifier)，使推理时的选择更好。Quiet-STaR 将推理过程下推到每个 token。三种方法都有效。没有一种是魔法——该循环会保留任何碰巧到达正确答案的捷径。

**类型：** 学习
**语言：** Python（标准库，自举循环模拟器）
**前置要求：** Phase 13 · 01-03（推理和 CoT），Phase 15 · 01（长程框架）
**时间：** ~60 分钟

## 问题

教模型推理的直接方法是收集人类编写的推理轨迹。这昂贵、缓慢，且受限于人类愿意撰写的高质量思维链的数量。

STaR（Self-Taught Reasoner, Zelikman 等人，2022）问道：如果模型自己写推理过程并根据已知答案给它们打分呢？循环如下：

1. 采样一个推理轨迹加答案。
2. 如果最终答案正确，保留该轨迹。
3. 在保留的轨迹上微调。
4. 重复。

它有效。GSM8K 和 CommonsenseQA 都在没有新的人工标注的情况下得到了提升。但该循环有一个内置偏差：任何产生正确答案的推理过程都会被保留，无论推理本身是否可靠。V-STaR（Hosseini 等人，2024）用学习到的验证器修补了这一点；Quiet-STaR（Zelikman 等人，2024）将这一想法推广到每个 token 的内部推理。

## 概念

### STaR：在有效的东西上自举

从一个具有某些弱推理能力的基础模型开始。在每个训练问题上，采样一个推理过程加答案。如果答案与标签匹配，保留（问题，推理，答案）三元组。在保留集上微调模型。重复。

一个转折很重要。如果模型永远无法答对某个问题，循环就无法从中学习。STaR 添加了**合理化 (rationalization)**：对于模型失败的问题，注入正确答案作为提示并重新提示模型生成一个引导到它的推理过程。合理化的推理过程被添加到训练集中。

原始论文（Zelikman 等人，2022）的结果：GPT-J 基础模型在通过重复 STaR 轮次加合理化进行训练后，GSM8K 上从 5.8% 提升到 10.7%——约 5 个百分点的绝对提升。在 CommonsenseQA 上，STaR 训练的 GPT-J 6B 达到 72.5%，可与在手工标注推理上微调的 GPT-3 175B（~73%）相媲美——后者模型大约大 30 倍。

### V-STaR：用 DPO 训练验证器

STaR 丢弃不正确的推理过程。Hosseini 等人（2024）观察到这些也是数据：每对（推理过程，"这是否正确"）都可以训练一个验证器。他们使用 Direct Preference Optimization (DPO, 直接偏好优化) 在正确和不正确的解上构建一个排序器。在推理时，采样 N 个推理过程并选择验证器的首选。

报告的提升：在 GSM8K 和 MATH 上比之前自改进基线高 4 到 17 个百分点，大部分增益来自使用验证器进行推理时选择，而非用于额外的生成器微调。

### Quiet-STaR：每个 token 的内部推理

Zelikman 等人（2024）问道：如果模型在每个 token 位置学习生成一个简短的内部推理，而不仅仅是在问题和答案之间呢？Quiet-STaR 训练模型在每个预测 token 之前发出一个隐藏的"思维"，然后通过一个学习到的权重将带思维的预测与基线预测混合。

结果：Mistral 7B 在 GSM8K 上的零样本 (zero-shot) 绝对提升从 5.9% 到 10.9%，CommonsenseQA 从 36.3% 到 47.2%，无需任务特定微调。模型学会了"何时思考"——难的 token 获得更长的内部推理；简单的几乎不需要。

### 为什么三者共享同一个安全问题

三种方法都使用最终答案作为梯度信号 (gradient signal)。一个通过有缺陷的推理到达正确答案的推理过程——利用捷径、猜测或使用非泛化模式——会得到正强化。在分布内 (in-distribution) 问题上捷径有效。在分布外 (out-of-distribution, OOD) 问题上它静默失效。

V-STaR 的验证器通过学习排序推理过程来缓解，但验证器是在同一标签集上训练的。它可能学会偏好格式良好的错误推理而非诚实的不确定性。更安全的设计是将 STaR 风格数据与 (a) 过程监督奖励模型 (process-supervised reward model)（奖励中间步骤，不仅仅是答案）和 (b) 能打破简单捷径的留出 OOD 评估结合起来。

### 对比

| 方法 | 训练信号 | 推理成本 | 数据浪费 | 已知故障模式 |
|---|---|---|---|---|
| STaR | 如果正确则保留（推理，答案） | 1x | 丢弃所有不正确推理 | 捷径推理 |
| STaR + 合理化 | 上述 + 正确答案提示的重试 | 1x | 较少 | 合理化推理可能不合理 |
| V-STaR | STaR + 两类 DPO 验证器 | Nx（best-of-N） | 最小 | 验证器可强化自信的错误 |
| Quiet-STaR | 每个 token 的推理 + 混合权重 | 1.5-3x | 最小 | 仍是答案条件化的梯度 |

### 这在 2026 年技术栈中的位置

STaR 是旧方法。但该模式在 2025-2026 年无处不在。在可验证数学问题上的 RL（DeepSeek-R1、Kimi-k1.5、o1）是 STaR 的答案条件化梯度信号，规模化应用。过程奖励模型（Lightman 等人，2023；OpenAI 的"Let's verify step by step"）是过程监督的替代方案。AlphaEvolve（第 3 课）是用于代码的 STaR，用程序评估器代替标签。Darwin Godel Machine（第 4 课）是用于智能体脚手架 (scaffolding) 本身的 STaR。

理解 STaR 能让这些都变得清晰。它是最小可行的自我改进循环。

## 使用它

`code/main.py` 在一个玩具算术任务上运行模拟的 STaR 循环。你可以观察：

- 准确率如何在自举轮次中攀升。
- 捷径如何潜入：模拟器包含一个"懒惰"推理类别，40% 的时间能得到正确答案但泛化很差。观察 STaR 是否保留它们。
- 验证器（V-STaR 风格）如何在推理时帮助但无法完全剪除训练期间引入的捷径。

## 交付它

`outputs/skill-star-loop-reviewer.md` 帮助你在训练之前审计提议的自我教学推理管道。

## 练习

1. 运行模拟器。将捷径频率设为零，然后设为 0.4。即使两者在训练分布上都达到 >90%，最终准确率有多大差异？

2. 向模拟器添加一个留出的 OOD 测试。从不同分布中抽取问题，并在分布内和 OOD 集上评估自举模型。量化差距。

3. 阅读 Quiet-STaR 论文（arXiv:2403.09629）第 3 节。分别用三句话解释"思维结束"token 和混合权重头 (mixing-weight head)。

4. 将 STaR 的"正确则保留"过滤器与独立奖励每个推理步骤的过程监督替代方案进行比较。识别标注成本差异和可能的质量差异。

5. 设计一个能捕获部署模型中捷径推理的评估。它不必完美——它只需要打破 STaR 循环会强化的最简单捷径。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|---|---|---|
| STaR | "Self-Taught Reasoner" | 在模型生成且 landing 正确答案的推理过程上微调；重复 |
| Rationalization (合理化) | "提示重试" | 注入正确答案并重新提示生成推理，用于基础模型失败的问题 |
| V-STaR | "Verifier STaR" | 在正确和不正确推理上 DPO 训练验证器，用于推理时选择 |
| Quiet-STaR | "每个 token 的推理" | 在每个 token 位置生成隐藏思维；与基线预测混合 |
| Answer-conditioned gradient (答案条件化梯度) | "基于结果的信号" | 训练循环奖励最终答案，而非推理步骤 |
| Process reward model (过程奖励模型) | "步骤级验证器" | 在每一步正确性上训练的奖励模型，而非结果——与 STaR 对比 |
| Shortcut rationale (捷径推理) | "正确答案，错误推理" | 通过非泛化模式到达标签的推理过程；STaR 保留这些 |

## 延伸阅读

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — 原始论文。
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — 添加 DPO 验证器用于推理时选择。
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — 每个 token 的内部推理。
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — 过程奖励模型，替代梯度信号。
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在可验证任务上的 RL，STaR 规模化到前沿训练。
