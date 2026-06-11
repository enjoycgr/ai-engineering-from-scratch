# 自主编程智能体全景（2026）

> SWE-bench Verified 在不到三年内从 4% 提升到 80.9%。同一个 Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2%，在 Cline 自主模式下得分 59.8%——模型周围的脚手架现在与模型本身一样重要。OpenHands（原 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 动作而非 JSON 工具调用。标题数字隐藏了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个只需要 1-2 行变更，而 SWE-bench Pro（10+ 行任务）对相同的前沿模型 sit 在 23-59%。

**类型：** 学习
**语言：** Python（标准库，CodeAct 与 JSON 工具调用比较）
**前置要求：** Phase 14 · 07（工具使用），Phase 15 · 01（长程智能体）
**时间：** ~45 分钟

## 问题

"哪个编程智能体最好"是错误的问题。正确的问题是：在与我的工作匹配的任务分布上，用我将在生产中运行的脚手架，我得到什么端到端可靠性？

2022 到 2026 年间，该领域学到脚手架——检索层、规划器、沙箱、编辑验证循环、反馈格式——是 load-bearing 的。Claude Sonnet 4.5 在 SWE-agent v1 上得分 43.2% 在 SWE-bench Verified 上；同一模型在 Cline 的自主脚手架内得分 59.8%。16.6 个绝对百分点差异，相同权重。基础模型是一个组件；循环是产品。

伴随问题是基准饱和隐藏回归。SWE-bench Verified 接近饱和，且简单任务尾部（161/500 任务需要 ≤2 行）拉高顶级分数。真实世界质量在 SWE-bench Pro（10+ 行变更）等分布上测量更好，相同领先者仍 sit 在 23-59%。

## 概念

### SWE-bench，一段说明

SWE-bench（Jimenez 等人）取真实 GitHub 问题和 ground-truth 补丁，要求智能体产生使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是人工策划的 500 任务子集，移除了模糊和破损任务。SWE-bench Pro 是更难的后继——需要 10+ 行变更的任务，当前前沿智能体 sit 在 23-59%。

### 2022 → 2026 曲线实际展示的内容

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025**：Aider 和 SWE-agent 内的 Claude 3.5/3.7 Sonnet 推进到 40-55% 范围。
- **2026**：Claude Sonnet 4.5 和前沿竞争者在 SWE-bench Verified 上 70-80%+。Epoch AI 的排行榜实时跟踪此数据。

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反思、验证器循环）和更好的基准（Verified 移除噪声）。

### CodeAct 与 JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，原 OpenDevin）做了一个具体的架构赌注：模型不是发出主机解码并执行的 JSON 工具调用，而是发出 Python 代码，Jupyter 风格内核在沙箱中运行它。智能体可以在一个动作内循环文件、链工具并捕获自己的异常。

权衡：

- **JSON 工具调用**：每个动作是一轮；易于审计；有限的组合性；默认安全因为每个调用通过显式验证器。
- **CodeAct**：一个动作可以是整个程序；可组合的；需要加固的沙箱（OpenHands 使用 Docker 隔离）；故障模式包括沙箱运行时允许的任何内容。

两种架构都在生产中。CodeAct 在开放平台中占主导（OpenHands、smolagents）。JSON 工具调用在托管服务中保持主导（Anthropic Managed Agents、OpenAI Assistants），提供者控制执行器。

### 2026 全景中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands (OpenDevin) | MIT | Docker 中的 CodeAct | 最活跃的开源平台；事件流可重放 |
| SWE-agent | MIT | 智能体-计算机接口 (ACI) | 首个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库中的 diff 编辑 | 最小脚手架，强回归稳定性 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上最高分的开源脚手架 |
| Devin (Cognition) | 专有 | 托管 VM + 规划器 | 首个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例行程序 | 第 10 课详细覆盖智能体循环 |

### 为什么脚手架占主导

编程运行是一个长程轨迹（第 1 课）。可靠性跨步骤复合。脚手架购买分数的三个地方：

1. **检索**：找到正确的文件来读取是静默瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库地图都攻击这一点。
2. **验证器循环**：运行测试、读取堆栈跟踪和重试是 SWE-bench 上 10+ 分的增量。
3. **失败遏制**：在错误时回滚的沙箱防止复合损坏。有和没有验证器循环的同一模型看起来像两个不同的产品。

### 基准饱和与真实分布

OpenHands 作者和 Epoch AI 都标记 SWE-bench Verified 有一个简单尾部：500 个任务中有 161 个只需要 1-2 行变更。高分数部分由该尾部驱动。SWE-bench Pro 限制为 10+ 行变更，即使对前沿系统也返回 23-59% 的分数。你的生产分布几乎肯定更接近 Pro 而非 Verified。

选择智能体的含义：在你的错误积压上运行一个 Pro 般的子集。重要的分数是你对代表性任务 shipped 的分数。

## 使用它

`code/main.py` 在固定的微型任务分布上比较两个玩具智能体脚手架：

1. 每轮一个动作的 **JSON 工具调用** 脚手架。
2. 每动作可以发出小段 Python 代码的 **CodeAct** 脚手架。

两者使用存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离。输出显示 CodeAct 脚手架在更少轮次中解决更多任务，代价是更大的每动作爆炸半径。

## 交付它

`outputs/skill-scaffold-audit.md` 帮助你在采用前审计提议的编程智能体脚手架：检索质量、验证器存在、沙箱隔离和基准到分布契合度。

## 练习

1. 运行 `code/main.py`。每个脚手架在同一任务集上需要多少轮？每个的每动作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文认为 CodeAct 在复杂任务上击败 JSON 工具调用。识别论文承认的一个故障模式，并写一句话关于该模式何时会在生产中占主导。

3. 从你的错误积压中挑选一个需要跨两个文件进行 10+ 行变更的任务。估算前沿模型在 (a) JSON 工具调用和 (b) CodeAct 下的端到端成功概率。证明差距。

4. SWE-bench Verified 有 161 个单文件、1-2 行任务。构造一个排除它们的分数。排行榜如何变化？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除模糊任务的具体方法论，并命名一个策划会遗漏的类别。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|---|---|---|
| SWE-bench | "编程基准" | 真实 GitHub 问题和 ground-truth 补丁及测试套件 |
| SWE-bench Verified | "清理子集" | 500 个人工策划任务，存在简单尾部 |
| SWE-bench Pro | "更难子集" | 10+ 行变更；前沿 sit 在 23-59% |
| CodeAct | "代码即动作" | 智能体发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON tool call | "函数调用" | 每个动作是在执行前验证的结构化 JSON payload |
| Scaffold | "智能体框架" | 围绕基础模型的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI (Agent-Computer Interface) | "SWE-agent 的格式" | 为 LLM 人机工程学设计的命令集，非人类 shell |
| Verifier loop | "测试并重试" | 运行测试、读取输出、修订补丁；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策划子集如何构建。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构和事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时跟踪分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长程编程智能体可靠性框架。
