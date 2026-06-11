---
name: classifier-stack-audit
description: 审计部署的输入/输出分类器栈 (input/output classifier stack)（模型、分类体系、输入护栏、输出护栏、对话护栏），并标记对抗攻击 (adversarial-attack) 缺口。
version: 1.0.0
phase: 15
lesson: 18
tags: [llama-guard, nemo-guardrails, input-rails, output-rails, colang, adversarial-attacks]
---

给定部署的分类器栈（Llama Guard 版本、NeMo Guardrails 配置、自定义分类器、归一化步骤），对照 2026 参考标准进行审计，标记栈未覆盖的攻击面。

产出：

1. **模型清单 (Model inventory)。** 列出正在使用的分类器。Llama Guard 3 (8B / 1B-INT4) 与 Llama Guard 4 (多模态, S1–S14)。NeMo Guardrails 版本。任何自定义分类器。如果部署接受图像，确认分类器是多模态的。
2. **分类体系映射 (Taxonomy mapping)。** 将声明的业务类别映射到分类器的分类体系。操作者关心的每个类别必须映射到一个分类器类别；未映射的类别是未受保护的 (unguarded)。
3. **护栏覆盖 (Rail coverage)。** 确认输入护栏 (input rails) 在模型轮次前触发，输出护栏 (output rails) 在响应发出前触发。对话护栏 (dialog rails, NeMo 中的 Colang) 执行跨轮次约束。单轮分类器无法捕捉多轮攻击。
4. **归一化 (Normalization)。** 确认输入经过 NFKC 归一化、同形异义字 (homoglyph) 映射，并在分类前去除零宽字符 / 变体选择器字符。原始字节分类是 Emoji Smuggling (Huang et al. 2025) 的 100% ASR 目标。
5. **攻击语料覆盖 (Attack-corpus coverage)。** 对每个已记录攻击（emoji smuggling、homoglyph、in-context redirection、semantic paraphrase），命名栈中的具体防御。仅靠分类器防御无法通过审计；必须与 Constitution（第 17 课）和运行时层（第 10、13、14 课）分层。

硬性否决：
- 在多模态输入上使用纯文本分类器的部署。
- 没有归一化步骤的部署。
- 只有输入护栏（敏感类别输出没有输出护栏）的部署。
- 将分类器视为单层安全层的栈。
- 操作者无法在自己的分布上复现的 ASR 声明。

拒绝规则：
- 如果用户声明的类别未映射到分类器的分类体系，拒绝并要求先提供映射。未映射 = 未受保护。
- 如果部署在多模态输入表面上引用 Llama Guard 3 的 ASR 数字，拒绝并要求 Llama Guard 4 或多模态分类器。
- 如果用户在高风险场景中将分类器层视为足够，拒绝。EU AI Act Article 14（第 15 课）期望人在其上监督。

输出格式：

返回一份分类器审计报告，包含：
- **模型清单**（名称、版本、模态）
- **分类体系映射**（操作者类别 → 分类器类别）
- **护栏覆盖**（input / output / dialog；在模型前/后触发）
- **归一化说明**（NFKC y/n、homoglyph y/n、zero-width strip y/n）
- **攻击语料覆盖**（attack → defense）
- **层级完整性**（classifier + constitution + runtime；三者都需）
- **就绪度**（production / staging / research-only）
