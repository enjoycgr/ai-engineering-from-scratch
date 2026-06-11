# Benchmarks: SWE-bench, GAIA, AgentBench

> 2026 年，三大 benchmark (基准测试) 锚定了 agent (智能体) 评估。SWE-bench 测试代码 patching (补丁)。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的组成、它们的 contamination (数据污染) 故事，以及它们不测量什么。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 分钟

## Learning Objectives

- 说出 SWE-bench 的 test harness (测试工具链)（FAIL_TO_PASS）并解释为什么它 gate (门禁) 在 unit tests (单元测试) 上。
- 解释为什么 SWE-bench Verified（OpenAI，500 个任务）存在以及它移除了什么。
- 描述 GAIA 的设计：对人类简单，对 AI 困难；三个难度等级。
- 说出 AgentBench 的八个环境和它对开源 LLM (大语言模型) 的主要 blocker (阻碍)。
- 总结 SWE-bench+ 的 contamination (数据污染) 发现及其影响。

## The Problem

排行榜告诉你哪个模型在一个 benchmark (基准测试) 上获胜。它们不会告诉你：

- 该 benchmark (基准测试) 是否被 contaminated (污染)（训练数据中有解决方案、测试泄露）。
- 该 benchmark (基准测试) 是否测量了你关心的内容（代码 vs 浏览 vs 通用）。
- 该 evaluator (评估器) 是否 robust (稳健)（AST matching (AST 匹配)、state checks (状态检查)、human review (人工审查)）。

在引用数字之前，了解这三个锚定 benchmark (基准测试) 及其 failure mode (失效模式)。

## The Concept

### SWE-bench（Jimenez et al., ICLR 2024 oral）

- 来自 12 个热门 Python 仓库的 2,294 个真实 GitHub issue (问题)。
- Agent (智能体) 获得：pre-fix commit (修复前提交) 的代码库 + 自然语言 issue description (问题描述)。
- Agent (智能体) 产出：一个 patch (补丁)。
- Evaluator (评估器)：应用 patch (补丁)，运行仓库的 test suite (测试套件)。Patch (补丁) 必须 flip FAIL_TO_PASS tests（之前失败，现在通过）而不破坏 PASS_TO_PASS tests。

SWE-agent（Yang et al., 2024）在发布时通过强调 agent-computer interfaces (agent-计算机接口)（模型理解的 file editor commands (文件编辑命令)、search syntax (搜索语法)）达到了 12.5%。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工精选的 500 任务子集。移除 ambiguous issues (模糊问题)、unreliable tests (不可靠测试) 和修复不清晰的任务。"你的 agent (智能体) 是否交付真实 patch (补丁)？"的主要 benchmark (基准测试)。

### Contamination (数据污染)

- 超过 94% 的 SWE-bench issues 早于大多数模型的 cutoff (截止日期)。
- **SWE-bench+** 发现 32.67% 的成功 patch (补丁) 在 issue text (问题文本) 中泄露了解决方案（模型在描述中看到了修复），31.08% 由于 weak test coverage (弱测试覆盖) 而可疑。
- Verified 更干净但并非完全没有 contamination (污染)。

实际影响：在 SWE-bench 上得分 50% 的模型在 SWE-bench+ 上可能只得分 35%。如果你声称 SWE-bench 性能，始终同时报告两者。

### GAIA（Mialon et al., 2023 年 11 月）

- 466 个问题；300 个保留给 huggingface.co/gaia-benchmark 的 private leaderboard (私有排行榜)。
- 设计理念："conceptually simple for humans (92%) but hard for AI (GPT-4 with plugins: 15%)。"（对人类概念上简单（92%），但对 AI 困难（带插件的 GPT-4：15%）。）
- 测试推理、multi-modality (多模态)、web (网页)、tool use (工具使用)。
- 三个难度等级；Level 3 需要跨 modalities (模态) 的长工具链。

GAIA 是你用来测量"generalist capability (通用能力)"的。不要将它与代码专用 benchmark (基准测试) 混淆。

### AgentBench（Liu et al., ICLR 2024）

- 8 个环境跨越 code (代码)（Bash、DB、KG）、games (游戏)（Alfworld、LTP）、web (网页)（WebShop、Mind2Web）和 open-ended generation (开放式生成)。
- Multi-turn (多轮)，每个 split (分割) 约 4k-13k turns (轮次)。
- 主要发现：long-term reasoning (长期推理)、decision-making (决策) 和 instruction following (指令遵循) 是开源 LLM (大语言模型) 追赶商业模型的 blocker (阻碍)。

### 这些不测量什么

- 真实世界的 operational cost (运维成本)（tokens (令牌)、wall-clock (挂钟时间)）。
- 对抗条件下的 safety behavior (安全行为)。
- 你领域中的性能（使用你自己的 evals (评估)，Lesson 30）。
- Tail failures (尾部故障)（benchmarks (基准测试) 取平均；生产运维人员关心最差的 1%）。

### Benchmarking (基准测试) 何时出错

- **Single-number fixation ( fixation 在单一数字上)。** SWE-bench 50% 告诉你的信息少于 P50/P75/P95 cost + step distribution (步骤分布)。
- **Contaminated claims (被污染的主张)。** 报告 SWE-bench 而不提及 Verified 或 SWE-bench+ 是误导性的。
- **Benchmark-as-development-target (以 benchmark 为开发目标)。** 为 benchmark (基准测试) 优化会偏离生产实用性。

## Build It

`code/main.py` 实现了一个玩具级的 SWE-bench-like harness (类 SWE-bench 工具链)：

- 合成 bug-fix tasks (缺陷修复任务)（3 个任务）。
- 一个脚本化的"agent (智能体)"提出 patch (补丁)。
- 检查 FAIL_TO_PASS（bug (缺陷) 现在已修复）和 PASS_TO_PASS（没有破坏任何东西）的 test runner (测试运行器)。
- 基于 question decomposition depth (问题分解深度) 的 GAIA-style (类 GAIA) 难度分类器。

运行方式：

```
python3 code/main.py
```

输出显示每个任务 + 每个难度的 resolution rate (解决率)，并使 evaluator (评估器) 规则具体化。

## Use It

- 对于代码 agent (智能体)，使用 **SWE-bench Verified**。始终报告 Verified 分数。
- 对于通用 agent (智能体)，使用 **GAIA**。使用 private leaderboard split (私有排行榜分割)。
- 对于多环境对比，使用 **AgentBench**。
- 对于你产品的实际形态，使用 **Custom evals (自定义评估)**（Lesson 30）。

## Ship It

`outputs/skill-benchmark-harness.md` 为任何 codebase-task pair (代码库-任务对) 构建了一个 SWE-bench-style harness (类 SWE-bench 工具链)，带有 FAIL_TO_PASS / PASS_TO_PASS gating (门禁)。

## Exercises

1. 将玩具 harness (工具链) 移植到运行在一个真实仓库上（选一个你自己的）。为已知 bug (缺陷) 编写 3 个 FAIL_TO_PASS 测试。
2. 添加一个 step-count metric (步骤计数指标)。在你的 3 个任务上，每个 resolution (解决) 需要多少 agent (智能体) 步骤？
3. 阅读 SWE-bench+ 论文。实现一个 solution-leakage check (解决方案泄露检查)（将 issue text (问题文本) 与 diff (差异) 进行模式匹配）。
4. 从 public split (公开分割) 下载一个 GAIA 问题。追踪 GPT-4 级别的 agent (智能体) 会做什么。它需要哪些工具？
5. 阅读 AgentBench 的 per-environment breakdown (按环境细分)。哪个环境镜像了你的产品面？那里的"SOTA (最先进)"是什么样？

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|----------|---------|
| SWE-bench | "Code agent benchmark (代码智能体基准测试)" | 2,294 个 GitHub issues (问题)；patch (补丁) 必须 flip FAIL_TO_PASS tests |
| SWE-bench Verified | "Clean SWE-bench (干净 SWE-bench)" | 500 个人工精选任务，OpenAI |
| FAIL_TO_PASS | "Fix gate (修复门禁)" | 之前失败、patch (补丁) 后必须通过 tests (测试) |
| PASS_TO_PASS | "No-regression gate (无回归门禁)" | 之前通过的 tests (测试)，patch (补丁) 后仍必须通过 |
| GAIA | "Generalist benchmark (通用基准测试)" | 466 个人类简单 / AI 困难的多工具问题 |
| AgentBench | "Multi-env benchmark (多环境基准测试)" | 8 个环境；长期多轮 |
| Contamination (数据污染) | "Training-set leak (训练集泄露)" | Benchmark tasks (基准测试任务) 出现在模型训练中 |
| SWE-bench+ | "Contamination audit (污染审计)" | 在成功 SWE-bench patch (补丁) 中发现 32.67% 解决方案泄露 |

## Further Reading

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — 原始 benchmark (基准测试)
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 精选子集
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — 通用 benchmark (基准测试)
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — 多环境套件
