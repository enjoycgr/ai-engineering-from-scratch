# 心智理论与涌现协调

> Li et al.（arXiv:2310.10701）表明，LLM 智能体在合作文本游戏中表现出**涌现的高阶心智理论**（Theory of Mind, ToM）——推理另一个智能体对第三个智能体信念的信念——但由于上下文管理和幻觉，在长期规划（long-horizon planning）中失败。Riedl（arXiv:2510.05174）测量了群体中的高阶协同（higher-order synergy），发现**只有** ToM 提示条件能产生与身份关联的分化（identity-linked differentiation）和目标导向的互补性（goal-directed complementarity）；低容量 LLM 只表现出虚假的涌现。也就是说，协调涌现是提示条件性的且模型依赖的，并非免费。本课实现了一个最小的 ToM 感知智能体，在有无 ToM 提示的情况下运行合作任务，并根据 Riedl 2025 协议测量协调增量。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 分钟

## Problem

多智能体协调（multi-agent coordination）常常看起来神奇：智能体分工、互相预判、避免冗余。通常这种“涌现”是提示工程的产物——有人告诉智能体要“协调”。移除提示，协调就消失。

Riedl 2025 的发现更严格：在受控条件下，只有当智能体被提示去推理**其他智能体的心智**（ToM）时，协调才会涌现。没有 ToM 提示，即使是强模型也显示出无法通过统计检验的协调模式。这对生产环境很重要：团队发布的“多智能体协调”功能是提示依赖且脆弱的。

本课将 ToM 视为一种特定能力（对信念的信念进行推理），构建一个最小的 ToM 感知智能体，并测量真实协调与提示伪装（prompt dressing）之间的区别。

## Concept

### ToM 的含义

发展心理学：3 岁孩子认为任何人的内心世界都和自己一样。5 岁孩子理解他人有不同的信念。7 岁孩子推理关于信念的信念（“她认为我认为球在杯子下面”）。这些分别是零阶、一阶和二阶 ToM。

对于 LLM 智能体，ToM 阶数映射为：

- **零阶（Zeroth-order）：** 没有他人的模型。智能体只基于自己的观察行动。
- **一阶（First-order）：** 智能体有每个其他智能体信念的模型。“Alice 相信 X。”
- **二阶（Second-order）：** 智能体建模递归信念。“Alice 相信 Bob 相信 X。”

Li et al. 2023 发现，一阶和二阶 ToM 在合作游戏中涌现于 LLM 智能体，但随着长期规划和不可靠通信而退化。

### Sally-Anne 测试简述

1985 年的错误信念测试：Sally 把弹珠放在篮子 A 里，离开。Anne 把它移到篮子 B。Sally 回来时会去哪里找？有一阶 ToM 的孩子说篮子 A（Sally 的信念与现实不同）。没有的孩子说篮子 B。

GPT-4 时代的 LLM 在直白提出时能通过 Sally-Anne 风格的测试。当叙事很长、场景多次变化、或问题间接提出时，它们会失败。这是 2026 年生产 LLM 中 ToM 的实际状态。

### Riedl 的协调测量

Riedl（arXiv:2510.05174）构建了一个群体规模测试：N 个智能体，一个合作目标，可变的提示条件。测量：

1. **身份关联分化（Identity-linked differentiation）。** 智能体是否随时间发展出稳定的角色区分？
2. **目标导向互补性（Goal-directed complementarity）。** 智能体的行动是否互补（不同子任务）而非重复？
3. **高阶协同（Higher-order synergy）。** 一个统计度量，衡量群体是否实现了任何子集都无法实现的目标。

结果：只有在 ToM 提示条件下，三个指标才产生高于基线的信号。没有 ToM 提示，中等容量模型的指标在机会水平附近徘徊。大型模型在没有显式 ToM 提示时表现出一些协调，但效果比显式提示时小。

### 协调幻觉

没有统计控制，“演示中的涌现协调”往往反映：

- 烘焙了协调的提示工程（系统提示词说“一起工作”）。
- 观察者偏差（我们看到我们期望的模式）。
- 对成功运行的事后选择。

没有可测量信号就宣传“涌现协调”的生产系统应被视为营销。先测量，再宣称。

### 最小 ToM 感知智能体

结构：

```
agent state:
  own_beliefs:    {智能体相信的事实}
  other_models:   {other_agent_id -> {智能体归因给它们的信念}}
  actions_last_N: [其他人行动的历史]

observation update:
  - 从直接观察更新 own_beliefs
  - 从它们的行动 + 先验信念更新 other_models[agent_id]

action selection:
  - 枚举候选行动
  - 对每个行动，预测在给定它们被建模的信念下，每个其他智能体下一步会做什么
  - 选择在这些预测下最大化联合结果的行动
```

`other_models` 属性就是 ToM 状态。一阶 ToM 只保留一层。二阶增加 `other_models[i][other_models_of_j]` —— 我认为智能体 i 认为智能体 j 相信什么。

### 为什么长期规划有害

Li et al. 记录：上下文限制导致智能体忘记哪个信念属于谁。幻觉向其他智能体模型添加错误信念。两者都产生“我以为他认为 X”的错误，随时间复合。

论文和 2024-2026 年后续工作中记录的缓解措施：

- **提示词中的显式 ToM 状态。** 结构化格式：`{agent_id: belief_list}`。强制检索以保持身份-信念绑定。
- **更短的推理链。** 每轮更少的 ToM 更新减少复合幻觉。
- **外部 ToM 存储。** 在 LLM 上下文之外维护模型；每轮只注入相关部分。

### ToM 在生产环境中何时失败

- **对抗性环境。** 具有良好 ToM 的智能体更容易被操纵（你可以建模它们对你的建模，然后利用）。
- **异构团队。** 当模型不同时，适用于一个对手的 ToM 模型无法泛化。
- **依赖 ground truth 的任务。** ToM 是关于信念的；如果正确性依赖事实，ToM 可能是一种干扰。

### 你实际可以测量的协调

三个实用信号表明团队的协调是真实的而非提示伪装的：

1. **随时间的互补性。** 在多轮任务中，智能体的行动是否覆盖不相交的子任务？
2. **预判。** 智能体 A 在 T+1 轮的行动是否依赖于对 B 在 T+2 轮行动的正确预测？
3. **修正。** 当 A 在 T 轮误读 B 的信念时，A 是否在 T+2 轮前修正？

这些可以在记录的多智能体系统中测量。它们是“协调”叙事的实质版本。

## Build It

`code/main.py` 实现了：

- `ToMAgent` —— 追踪自身信念和每个其他智能体的信念模型。
- 一个合作任务：三个智能体必须从三个盒子中收集三个 token；每个盒子只能放一个 token。智能体不能通信；它们从彼此的行动中推断意图。
- 两种配置：`zeroth_order`（无 ToM）和 `first_order`（带一层信念模型的 ToM）。
- 在 200 次随机试验上的测量：完成率、重复率（两个智能体瞄准同一个盒子）、平均完成轮数。

运行：

```
python3 code/main.py
```

预期输出：零阶智能体以约 35% 的比率重复努力，并在 10 轮内完成约 60% 的试验。一阶 ToM 智能体以约 5% 的比率重复，并完成约 95%。这个增量就是可测量的协调效果。

## Use It

`outputs/skill-tom-auditor.md` 是一个技能文件，它审计多智能体系统对“涌现协调”的宣称。检查提示伪装、与对照组的统计显著性，以及测量的互补性。

## Ship It

协调宣称检查清单：

- **对照条件。** 你的系统的一个没有协调提示词的版本。两者都测量。
- **统计检验。** 系统和对照之间的差异在你的指标上是否显著于 `p < 0.05`？
- **互补性测量。** 随时间的行动不相交性，而不仅仅是最终成功。
- **失败案例日志。** 当智能体协调失败时，ToM 状态看起来如何？
- **模型容量披露。** 如果效果在较小模型上消失，请说明。

## Exercises

1. 运行 `code/main.py`。确认一阶 ToM 将重复率降低约 7 倍。当你扩展到 5 个智能体和 5 个盒子时，差距是否持续？
2. 实现二阶 ToM（智能体 A 建模 B 对 C 的想法）。它比一阶有改进吗？在什么任务上？
3. 向 ToM 状态注入**幻觉**：每轮随机翻转一个信念。这会使一阶性能退化多少？
4. 阅读 Li et al.（arXiv:2310.10701）。复现“长期退化”发现：随着轮次从 10 增长到 30，你的一阶 ToM 性能如何变化？
5. 阅读 Riedl 2025（arXiv:2510.05174）。在你的仿真日志上实现高阶协同统计量。没有 ToM 提示条件时，效果是否存在？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Theory of Mind | "Understanding others' minds" | 建模另一个智能体信念的能力。按阶数分级（0, 1, 2+）。 |
| Sally-Anne test | "The false-belief test" | 1985 年发展心理学；LLM 通过直白版本，失败复杂版本。 |
| First-order ToM | "A believes X" | 建模另一个智能体对事实的信念。 |
| Second-order ToM | "A believes B believes X" | 递归建模再深一层。 |
| Identity-linked differentiation | "Stable roles over time" | Riedl 的指标：角色持续存在，而非随机。 |
| Goal-directed complementarity | "Disjoint actions" | 智能体瞄准不同子任务，而非同一个。 |
| Higher-order synergy | "Group exceeds any subset" | Riedl 对真实协调的统计度量。 |
| Coordination illusion | "It looks coordinated" | 没有可测量信号的协调的提示伪装外观。 |

## Further Reading

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) —— 合作游戏中的涌现 ToM；长期失败模式
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) —— 群体规模测量；ToM 提示是承重条件
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) —— 1978 年 ToM 概念的起源
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) —— Sally-Anne 论文（1985）
