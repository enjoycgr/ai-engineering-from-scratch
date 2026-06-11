# Eval-Driven Agent Development（评估驱动的智能体开发）

> Anthropic 的指导："从简单提示开始，通过全面评估优化它们，仅在需要时才添加多步智能体系统。"评估不是最后一步。它是驱动 Phase 14 中所有其他选择的外层循环。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** All of Phase 14.
**Time:** ~60 分钟

## Learning Objectives

- 说出三个评估层——static benchmarks（静态基准）、custom offline（自定义离线）、online production（在线生产）——以及每层的作用。
- 解释 evaluator-optimizer（评估器-优化器）紧密循环。
- 描述 2026 年最佳实践：评估与代码共存，在 CI 中运行，把关 PR。
- 将 Phase 14 的每一课连接到它生成的评估用例。

## The Problem

智能体通过演示。它们以演示无法预测的方式在生产中失败。基准测试回答"这个模型是否广泛有能？"而不是"这个智能体是否为我产品提交了正确的补丁？"答案是：三层评估，持续运行，每个护栏和学到的规则都映射到一个评估用例。

## The Concept

### 三层评估

1. **Static benchmarks（静态基准）**——SWE-bench Verified 用于代码（Lesson 19）、WebArena/OSWorld 用于浏览/桌面（Lesson 20）、GAIA 用于通才（Lesson 19）、BFCL V4 用于工具使用（Lesson 06）。用于跨模型比较和回归把关。污染是真实的：SWE-bench+ 发现 32.67% 的解决方案泄漏。始终报告 Verified / +-审计分数。

2. **Custom offline evals（自定义离线评估）**——你产品的形态：
   - LLM-as-judge（Langfuse、Phoenix、Opik —— Lesson 24）。
   - Execution-based（运行补丁，检查测试）。
   - Trajectory-based（将动作序列与黄金路径比较；OSWorld-Human 显示顶级智能体比黄金路径高 1.4-2.7 倍）。

3. **Online evals（在线评估）**——生产环境：
   - 会话回放（Langfuse）。
   - 护栏触发的告警（Lesson 16, 21）。
   - 每步成本 / 延迟追踪（Lesson 23 OTel spans）。

### Evaluator-optimizer（Anthropic）

紧密循环：

1. Proposer 生成输出。
2. Evaluator 评判。
3. 精炼直到 evaluator 通过。

这是 Self-Refine（Lesson 05）的泛化。任何你在乎的智能体流程都可以包装在 evaluator-optimizer 中以提高可靠性。

### 2026 年最佳实践

- 评估与代码共存。
- 在每个 PR 的 CI 中运行。
- 根据评估分数把关合并（例如"与 main 相比无 >5% 回归"）。
- 每个护栏映射到一个评估用例。
- 每个学到的规则（Reflexion、pro-workflow learn-rule）映射到一个失败用例。

### 将 Phase 14 串联起来

Phase 14 的每一课都生成评估用例：

| 课程 | 它生成的评估用例 |
|------|----------------|
| 01 Agent Loop | 预算耗尽、无限循环防护 |
| 02 ReWOO | 工具失败时规划器正确重规划 |
| 03 Reflexion | 学到的反思在重试时生效 |
| 05 Self-Refine/CRITIC | Judge 通过精炼输出 |
| 06 Tool Use | 参数强制转换有效；未知工具被拒绝 |
| 07-10 Memory | 检索引用匹配来源；陈旧事实失效 |
| 12 Workflow Patterns | 每种模式产生正确输出 |
| 13 LangGraph | 恢复精确复现状态 |
| 14 AutoGen Actors | DLQ 捕获崩溃的处理器 |
| 16 OpenAI Agents SDK | 护栏在正确输入上触发 |
| 17 Claude Agent SDK | 子智能体结果返回编排器 |
| 19-20 Benchmarks | SWE-bench Verified 分数、WebArena 成功率、OSWorld 效率 |
| 21 Computer Use | 每步安全捕获注入的 DOM |
| 23 OTel | Spans 发出所需属性 |
| 26 Failure Modes | 检测器标记已知故障 |
| 27 Prompt Injection | PVE 拒绝中毒检索 |
| 28 Orchestration | Supervisor 路由到正确专家 |
| 29 Runtime Shapes | DLQ 处理 N% 故障 |

如果你的评估套件对每个都有用例，你就覆盖了 Phase 14。

### 评估驱动开发的常见失效点

- **无基线。** 没有 last-known-good 的评估无法阅读。存储基线。
- **无依据的 LLM-judge。** Judge 也会幻觉。CRITIC 模式（Lesson 05）——judge 依据外部工具。
- **对评估过拟合。** 为评估优化偏离了生产实用性。轮换用例。
- **不稳定的评估。** 非确定性用例导致误报。固定种子，快照状态。

## Build It

`code/main.py` 是一个 stdlib 评估载体：

- 带类别（benchmark、custom、online）的用例注册表。
- 一个接受测试的脚本化智能体。
- Evaluator-optimizer 循环：提出、评判、精炼直到通过或达到最大轮数。
- CI 门：聚合通过率 + 与基线的回归。

运行方式：

```bash
python3 code/main.py
```

输出：每用例通过/失败、回归标志、CI 门裁决。

## Use It

- 在与智能体代码相同的仓库中编写评估用例。
- 在每个 PR 的 CI 中运行它们。
- 在回归上失败构建。
- 随时间追踪通过率。
- 将每个生产故障绑定到一个新用例。

## Ship It

`outputs/skill-eval-suite.md` 为一个带 CI 门和回归追踪的智能体产品构建三层评估套件。

## Exercises

1. 取一个你的生产故障。编写一个复现它的评估用例。你的智能体现在能通过吗？
2. 为你的领域构建一个 LLM-judge 评分标准，三个维度（事实、语气、范围）。评分 50 个会话。
3. 将评估套件接入 CI。>=5% 回归时失败构建。
4. 添加轨迹效率指标：智能体花费了多少步 vs 黄金轨迹？
5. 将 Phase 14 的每一课映射到你套件中的一个评估用例。有缺失吗？那是需要弥补的缺口。

## Key Terms

| 术语 | 行业说法 | 实际含义 |
|------|----------|----------|
| Static benchmark | "现成评估" | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | "领域评估" | 针对你产品形态的 LLM-as-judge / 执行 / 轨迹 |
| Online eval | "生产评估" | 会话回放、护栏告警、成本/延迟追踪 |
| Evaluator-optimizer | "提出-评判-精炼" | 迭代直到评判通过 |
| CI gate | "合并阻断器" | 评估回归时失败构建 |
| Baseline | "上次已知良好" | 检测回归的参考分数 |
| Trajectory efficiency | "步数超黄金路径" | 智能体步数除以人类专家最小步数 |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "从简单开始，用评估优化"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) —  curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — 工具使用基准
- [Langfuse docs](https://langfuse.com/) — 实践中的评估 + 会话回放
