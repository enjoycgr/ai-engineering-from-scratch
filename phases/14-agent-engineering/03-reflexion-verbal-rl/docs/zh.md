# Reflexion: Verbal Reinforcement Learning（语言强化学习）

> 基于梯度的 RL 需要数千次试验和 GPU 集群才能修复一个失败模式。Reflexion（Shinn 等人，NeurIPS 2023）用自然语言实现：每次失败试验后，智能体写一篇 reflection（反思），存入 episodic memory（情节记忆），并在下一次试验中以该记忆为条件。这是 Letta 的 sleep-time compute（睡眠时计算）、Claude Code 的 CLAUDE.md learnings 和 pro-workflow 的 learn-rule 背后的模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 分钟

## Learning Objectives（学习目标）

- 说出 Reflexion 的三个组成部分（Actor、Evaluator、Self-Reflector）以及 episodic memory（情节记忆）的作用。
- 用 stdlib 实现一个 Reflexion 循环，包含 binary evaluator（二元评估器）、reflection buffer（反思缓冲区）和全新的重试。
- 为给定任务选择 scalar（标量）、heuristic（启发式）和 self-evaluated（自评估）反馈源。
- 解释为什么 verbal reinforcement（语言强化）能捕捉基于梯度的 RL 需要数千次试验才能修复的错误。

## The Problem（问题）

智能体失败了一个任务。在标准 RL 中，你需要再跑数千次试验，计算梯度，更新权重。昂贵、缓慢，而且大多数生产智能体没有为每次失败准备训练预算。

Reflexion（Shinn 等人，arXiv:2303.11366）问了一个不同的问题：如果智能体只是思考为什么失败，然后带着这个想法重试呢？没有权重更新。没有梯度。只是把自然语言存储在试验之间。

结果：在 ALFWorld 上击败 ReAct 和其他非 fine-tune 基线。在 HotpotQA 上超越 ReAct。在代码生成（HumanEval/MBPP）上当时达到 SOTA。全部没有一次梯度步骤。

## The Concept（概念）

### 三个组成部分

```
Actor         : 生成一条轨迹（ReAct 式循环）
Evaluator     : 为轨迹打分——二元、启发式或自评估
Self-Reflector: 就失败写一篇自然语言的 reflection
```

加上一个数据结构：

```
Episodic memory（情节记忆）: 先前 reflection 的列表， prepended 到下一次试验的 prompt
```

一次试验运行 Actor。Evaluator 打分。如果分数低，Self-Reflector 产生一篇 reflection（"我选错了工具，因为我把问题误读成问 X，而实际在问 Y"）。Reflection 进入 episodic memory。下一次试验重新开始，但看到了 reflection。

### 三种评估器类型

1. **Scalar（标量）** —— 外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单，信号最强。
2. **Heuristic（启发式）** —— 预定义失败特征。"如果智能体连续两次产生相同动作，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **Self-evaluated（自评估）** —— LLM 为自己的轨迹打分。在没有 ground truth 时需要。信号较弱；与 tool-grounded verification（Lesson 05 — CRITIC）配合良好。

2026 年的默认是混合：有 scalar 时用 scalar，没有时用 self-eval，heuristic 作为安全护栏。

### 为什么这个模式泛化

Reflexion 与其说是一个新算法，不如说是一个被命名的模式。几乎每个生产级"自愈"智能体都在运行某种变体：

- Letta 的 sleep-time compute（Lesson 08）：一个独立的智能体反思过去的对话并写入 memory blocks。
- Claude Code 的 `CLAUDE.md` / "save memory" 模式：将 reflection 捕获为 learning，prepending 到未来会话。
- pro-workflow 的 `/learn-rule` 命令：将修正捕获为显式规则。
- LangGraph 的 reflection 节点：一个节点为输出打分，并在需要时路由到 refine。

全部源于同一个洞察：自然语言是一种足够丰富的媒介，可以在运行之间承载"我从失败中学到了什么"。

### 何时有效、何时无效

Reflexion 有效当：

- 有清晰的失败信号（测试失败、工具错误、错误答案）。
- 任务类别可复现（可以再次问同一类型的问题）。
- Reflection 有改进轨迹的空间（足够的 action budget）。

Reflexion 无效当：

- 智能体第一次尝试就成功。
- 失败是外部的（网络中断、工具损坏）——对"网络中断了"的 reflection 对未来运行没有帮助。
- Reflection 变成 superstition（迷信）——存储关于一次偶然 flaky run 的叙事。

2026 年陷阱：memory rot（记忆腐烂）。Reflections 不断累积；有些已过时或错误；重试随着 episodic buffer 增长而变慢。缓解：定期 compaction（Lesson 06）、reflection 的 TTL，或单独的 sleep-time cleanup agent（Letta）。

## Build It（动手实现）

`code/main.py` 在玩具谜题上实现 Reflexion：生成一个和为目标值的 3 元素列表。Actor 发出候选列表；Evaluator 检查和；Self-Reflector 写一行关于出了什么问题的诊断。Reflection 进入下一次试验的 episodic memory。

组件：

- `Actor` —— 一个看到 reflection 后会改进的脚本化策略。
- `Evaluator.binary()` —— 对目标和进行 pass/fail 判定。
- `SelfReflector` —— 生成失败的一行诊断。
- `EpisodicMemory` —— 带 TTL 语义的受限列表。

运行：

```
python3 code/main.py
```

轨迹显示三次试验。试验 1 失败，存储 reflection，试验 2 看到 reflection 并改进但仍失败，试验 3 成功。与基线运行（无 reflection）比较——它停留在试验 1 的答案。

## Use It（应用）

LangGraph 将 reflection 作为节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将 episodic buffer 外部化为 markdown 文件。Letta 的 sleep-time compute 在 downtime 运行 Self-Reflector，使主智能体保持 latency-bound（受延迟约束）。OpenAI Agents SDK 不直接提供 Reflexion；你需要用按分数拒绝轨迹的自定义 Guardrail 和跨运行保留的 memory `Session` 来构建它。

## Ship It（交付）

`outputs/skill-reflexion-buffer.md` 创建并维护一个带 reflection 捕获、TTL 和 deduplication（去重）的 episodic buffer。给定一个任务类别和一次失败，它发出一篇真正帮助下一次试验的 reflection（不是泛泛的"be more careful"）。

## Exercises（练习）

1. 从二元评估器切换到返回距离指标（离目标多远）的 scalar evaluator。收敛更快吗？
2. 为 reflection 添加 10 次试验的 TTL。更旧的 reflection 在那之后有害还是有益？
3. 实现启发式评估器：如果相同动作重复，标记试验为卡住。这与 Self-Reflector 如何交互？
4. 用一个对抗性 Actor（忽略 reflection）运行 Reflexion。强制 Actor 注意到它们的最低 reflection prompt engineering 是什么？
5. 阅读 Reflexion 论文第4节关于 AlfWorld。从概念上复现 130% 成功率提升：与 vanilla ReAct 的关键差异是什么？

## Key Terms（关键术语）

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| Reflexion | "Self-correction（自我修正）" | Shinn 等人 2023 —— Actor、Evaluator、Self-Reflector 加 episodic memory |
| Verbal reinforcement | "Learning without gradients（无梯度学习）" | 将自然语言 reflection prepending 到下一次试验的 prompt |
| Episodic memory | "Per-task reflections（每任务反思）" | 一个任务类别的先前 reflection 的受限缓冲区 |
| Scalar evaluator | "Binary success signal（二元成功信号）" | 来自 ground truth 的 pass/fail 或数值分数 |
| Heuristic evaluator | "Pattern-based detector（基于模式的检测器）" | 预定义失败特征（如 stuck-loop、too-many-steps） |
| Self-evaluator | "LLM-as-judge on own trace（LLM 评判自身轨迹）" | 没有 ground truth 时的低信号备选——与 tool-grounded verification 配对 |
| Memory rot | "Stale reflections（陈旧反思）" | Episodic buffer 充满过时条目；用 compaction/TTL 修复 |
| Sleep-time reflection | "Async self-reflection（异步自我反思）" | 在 hot path 之外运行 Self-Reflector，使主智能体保持快速 |

## Further Reading（延伸阅读）

- [Shinn 等人, Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) —— 经典论文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) —— 生产中的异步 reflection
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 将 episodic buffer 作为上下文的一部分管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— reflection 节点模式
