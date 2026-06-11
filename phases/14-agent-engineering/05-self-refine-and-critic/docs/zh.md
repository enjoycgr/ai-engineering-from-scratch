# Self-Refine and CRITIC: Iterative Output Improvement（迭代输出改进）

> Self-Refine（Madaan 等人，2023）用一个 LLM 扮演三个角色——generate（生成）、feedback（反馈）、refine（精炼）——在一个循环中。7 个任务平均提升 +20 绝对百分点。CRITIC（Gou 等人，2023）通过将 verification（验证）路由到外部工具来强化反馈步骤。2026 年，这个模式在每个框架中都以 "evaluator-optimizer（评估器-优化器）"（Anthropic）或 guardrail loop（护栏循环）（OpenAI Agents SDK）的形式出现。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 分钟

## Learning Objectives（学习目标）

- 陈述 Self-Refine 的三个 prompt（generate、feedback、refine），并解释为什么 history 对 refine prompt 很重要。
- 解释 CRITIC 的关键洞察：没有外部 grounding 时，LLM 在 self-verification（自我验证）上不可靠。
- 用 stdlib 实现一个带 history 的 Self-Refine 循环，以及可选的外部 verifier（验证器）。
- 将该模式映射到 Anthropic 的 "evaluator-optimizer" 工作流和 OpenAI Agents SDK 的输出 guardrails。

## The Problem（问题）

智能体产生了一个几乎正确的答案。也许一行代码有语法错误。也许一个摘要太长。也许一个计划遗漏了一个边界情况。你想要的是：智能体批判自己的输出，然后修复它。

Self-Refine 表明这可以用单个模型完成，无需训练数据，无需 RL。但有一个陷阱：LLM 在硬事实上的 self-verification 很差。CRITIC 命名了修复方案——将 verify 步骤路由到外部工具（搜索、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认值：generate、verify（尽可能外部验证）、refine，直到 verifier 通过。

## The Concept（概念）

### Self-Refine（Madaan 等人，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 看到完整 history —— 所有先前的输出和 critiques —— 因此不会重复错误。论文做了消融实验：去掉 history，质量急剧下降。

 headline：7 个任务平均 +20 绝对百分点提升（数学、代码、首字母缩略词、对话），包括 GPT-4。无需训练，无需外部工具，单个模型。

### CRITIC（Gou 等人，arXiv:2305.11738，v4 2024年2月）

Self-Refine 的弱点：feedback 步骤是 LLM 给自己打分。对于事实性论断，这不可靠（幻觉通常对产生它的模型来说看起来很有说服力）。CRITIC 将 `feedback(task, output)` 替换为 `verify(task, output, tools)`，其中 `tools` 包括：

- 用于事实性论断的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定验证器（单元测试、类型检查器、linter）。

Verifier 产生基于工具结果的 structured critique（结构化批评）。Refiner 然后以该 critique 为条件进行改写。

 headline：CRITIC 在事实性任务上优于 Self-Refine，因为 critique 有 grounding。在没有外部验证器的任务（创意写作、格式化）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形状：

1. **Verifier passes（验证器通过）。** 外部测试返回成功。有外部验证器时首选（单元测试、类型检查器、guardrail assertion）。
2. **No feedback issued（没有反馈）。** 模型说"输出没问题。"更便宜但不可靠；与最大迭代次数上限配对。

2026 默认值：组合使用。"如果 verifier 通过 OR 模型说没问题 AND 迭代次数 >= 2 OR 迭代次数 >= max_iterations 则停止。"

### Evaluator-Optimizer（Anthropic，2024）

Anthropic 2024 年12月的文章将此命名为五种工作流模式之一。两个角色：

- Evaluator：为输出打分并产生 critique。
- Optimizer：给定 critique 修改输出。

循环直到 evaluator 通过。这是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 添加的关键工程细节：evaluator 和 optimizer prompt 应该显著不同，这样模型就不会只是 rubber-stamp（橡皮图章式通过）。

### OpenAI Agents SDK 输出 guardrails

OpenAI Agents SDK 将此模式以 "output guardrails（输出护栏）" 形式提供。Guardrail 是在智能体最终输出上运行的 validator（验证器）。如果 guardrail 触发（引发 `OutputGuardrailTripwireTriggered`），输出被拒绝，智能体可以重试。Guardrails 可以调用工具（CRITIC 风格）或作为纯函数（Self-Refine 风格）。

### 2026 年陷阱

- **Rubber-stamp loops（橡皮图章循环）。** 同一个模型用相同 prompt 风格做生成和 critique，收敛到"看起来不错。"使用结构上不同的 prompt，或用一个更便宜的小模型做 critique。
- **Over-refinement（过度精炼）。** 每次 refine pass 增加延迟和 token。预算 1-3 轮；超过后，升级到人工审核。
- **CRITIC on trivial tasks（ trivial 任务上的 CRITIC）。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为 stub verifier 付出延迟代价。

## Build It（动手实现）

`code/main.py` 在玩具任务上实现 Self-Refine 和 CRITIC：给定主题生成一个短 bullet list。Verifier 检查格式（3 个 bullet，每个不超过 60 字符）。CRITIC 添加一个外部 "fact verifier"，对已知幻觉进行惩罚。

组件：

- `generate` —— 脚本化生产者。
- `feedback` —— LLM 风格 self-critique。
- `verify_external` —— CRITIC 风格 grounded verifier。
- `refine` —— 给定 history 重写输出。
- Stop condition —— verifier 通过或最多 4 次迭代。

运行：

```
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 运行。CRITIC 捕捉到 Self-Refine 遗漏的事实错误，因为外部 verifier 拥有 self-critic 所没有的 grounding。

## Use It（应用）

Anthropic 的 evaluator-optimizer 是 Claude 友好语言中的这个模式。OpenAI Agents SDK 的 output guardrails 是 CRITIC 形状的（guardrails 可以调用工具）。LangGraph 提供一个读起来像 Self-Refine 的 reflection 节点。Google 的 Gemini 2.5 Computer Use 添加了一个每步 safety evaluator，这是 CRITIC 的变体：每个动作在提交前都经过验证。

## Ship It（交付）

`outputs/skill-refine-loop.md` 根据任务形状、验证器可用性和迭代预算配置 evaluator-optimizer 循环。发出 generator、evaluator/verifier 和 optimizer 的 prompt，以及一个停止策略。

## Exercises（练习）

1. 用 max_iterations=1 运行玩具。CRITIC 还有帮助吗？
2. 将外部验证器替换为 noisy 的（随机 30% 假阳性）。循环做什么？这是 2026 年大多数 guardrail 堆栈的现实。
3. 实现 "generator-critic on different models" 变体：大模型生成，小模型 critique。它打败同模型吗？
4. 阅读 CRITIC 第3节（arXiv:2305.11738 v4）。命名三种 verification-tool 类别并各给一例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的 verifier 角色。SDK 做对了什么，做错了什么？

## Key Terms（关键术语）

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| Self-Refine | "LLM that fixes itself（自我修复的 LLM）" | Generate -> feedback -> refine 循环，同一模型，带 history |
| CRITIC | "Tool-grounded verification（工具化验证）" | 用外部验证器（搜索、代码、计算、测试）替换 feedback |
| Evaluator-Optimizer | "Anthropic workflow pattern（Anthropic 工作流模式）" | 两个角色——evaluator 打分，optimizer 修改——循环到收敛 |
| Output guardrail | "Post-hoc check（事后检查）" | OpenAI Agents SDK 的 validator，在智能体产生输出后运行 |
| Verify step | "Critique phase（批判阶段）" | 承重决策：grounded 还是 self-rated |
| Refine history | "What the model already tried（模型已尝试的内容）" | Prepending 到 refine prompt 的先前输出 + critiques；去掉则质量崩溃 |
| Rubber-stamp loop | "Self-agreement failure（自我认同失败）" | 相同 prompt critique 返回"看起来不错"；用结构上不同的 prompt 修复 |
| Stop condition | "Convergence test（收敛测试）" | Verifier 通过 OR 无 feedback AND 迭代上限；永不用单一条件 |

## Further Reading（延伸阅读）

- [Madaan 等人, Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— 经典论文
- [Gou 等人, CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) —— 工具化验证
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— evaluator-optimizer 工作流模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) —— 作为 CRITIC 形状验证器的 output guardrails
