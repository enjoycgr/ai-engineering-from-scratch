# 使用 HTN 与进化搜索进行规划

> Symbolic planning (符号规划) 处理计划可被证明正确的场景。Evolutionary code search (进化代码搜索) 处理适应度函数可被机器检验的场景。ChatHTN (2025) 与 AlphaEvolve (2025) 展示了各自与 LLM (大语言模型) 结合时的能力边界。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 分钟

## 学习目标

- 解释 Hierarchical Task Networks (HTN，层次任务网络)：tasks (任务)、methods (方法)、operators (操作符)、preconditions (前置条件)、effects (效果)。
- 描述 ChatHTN 的混合循环 —— symbolic search (符号搜索) 配合 LLM fallback decomposition (LLM 回退分解)。
- 解释 AlphaEvolve 的 evolutionary loop (进化循环) 以及它为何只适用于拥有 programmatic evaluator (程序化评估器) 的场景。
- 用标准库实现一个玩具级 HTN planner (HTN 规划器) 和一个玩具级 evolutionary search (进化搜索)。

## 问题背景

ReWOO (Lesson 02)、Plan-and-Execute 和 ReAct 覆盖了大部分 agent planning (智能体规划)。但有两类场景它们处理得不好：

1. **需要可证明正确性的计划。** 调度、飞行路径规划、合规工作流 —— 计划必须在构造上就是 sound (可靠的)。一个流畅的 LLM 计划偶尔 hallucinate (幻觉) 出某一步，这是不可接受的。
2. **拥有机器可检验 fitness function (适应度函数) 的优化问题。** 矩阵乘法、调度启发式、编译器优化 pass —— 目标不是"一个正确的计划"，而是"最好的计划"。

HTN planning (HTN 规划) 和 AlphaEvolve 分别解决这两类不同的问题。它们都把 LLM 当作 amplifier (放大器)，而非替代品。

## 核心概念

### Hierarchical Task Networks (HTN)

HTN 包含：

- **Tasks (任务)** —— compound tasks (复合任务，需要被分解) 和 primitive tasks (原子任务，可直接执行)。
- **Methods (方法)** —— 将 compound task 分解为 subtasks (子任务) 的方式，带有 preconditions (前置条件)。
- **Operators (操作符)** —— 具有 preconditions 和 effects (效果) 的 primitive actions (原子动作)。
- **State (状态)** —— 一组 facts (事实)。

Planning (规划)：给定一个 goal task (目标任务) 和 initial state (初始状态)，找到一种 decomposition (分解) 为 primitive operators 的方案，使得它们的 preconditions 按顺序被满足。

HTN 比 LLM 更古老，至今仍是 provably-correct plans (可证明正确计划) 的参考标准。

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN (arXiv:2505.11814) 将 symbolic HTN 与 LLM 查询交织在一起：

1. 尝试用现有 methods 分解当前的 compound task。
2. 如果没有 method 适用，询问 LLM："在状态 `s` 下，你会如何分解 `task`？"
3. 将 LLM 的回答转换为 candidate subtasks (候选子任务)。
4. 对照 operator schema (操作符模式) 进行验证；拒绝无效的 decomposition。
5. 递归执行。

该论文的核心主张：每个生成的计划都是 provably sound (可证明可靠) 的，因为 LLM 的建议只作为 candidate decompositions 进入系统，从不直接编辑计划。symbolic layer (符号层) 拥有 correctness (正确性)；LLM 扩展 method library (方法库)。

Online method learning (OpenReview `gwYEDY9j2x`，2025 后续研究) 增加了一个 learner (学习器)，通过 regression (回归) 泛化 LLM 生成的 decomposition —— 可将 LLM 查询频率降低最多 75%。

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve (arXiv:2506.13131，DeepMind，2025 年 6 月) 则是另一种范式：由 Gemini 2.0 Flash/Pro ensemble (模型组合) 编排的 evolutionary code search (进化代码搜索)。

Loop (循环)：

1. 从一个 seed program (种子程序) + programmatic evaluator (返回 fitness score 的评估器) 开始。
2. LLM ensemble 提出 mutations (变异)。
3. 通过 evaluator 运行 mutations。
4. 保留最好的；再次变异。

已发表的成果：

- 56 年来首次超越 Strassen 的 4x4 复数矩阵乘法（48 次标量乘法）。
- 通过 Borg scheduling heuristic 恢复 0.7% 的 Google 计算资源。
- 在前沿工作负载上实现 32% 的 FlashAttention 加速。

Hard constraint (硬性约束)：fitness function 必须是 machine-checkable (机器可检验的)。对 prose answers (散文式回答) 进行 evolutionary search 不会收敛。

### 何时使用哪种方法

| 问题类型 | 使用 | 原因 |
|---|---|---|
| 带硬约束的调度 | HTN + ChatHTN | 可证明的可靠性 |
| 编译器优化 | AlphaEvolve | 机器可检验的 fitness |
| 多步骤任务执行 | ReAct / ReWOO | LLM 在循环中，无形式化保证 |
| 带测试的代码改进 | AlphaEvolve | 测试本身就是 evaluator |
| 策略驱动的自动化 | HTN | Preconditions 编码策略 |

### 这种模式的失效场景

- **没有 operators 的 HTN。** 没有 precondition/effect schema，soundness claim (可靠性声明) 就会崩溃。ChatHTN 的 "LLM 建议分解" 需要 schema 来拒绝无效动作。
- **没有真实 evaluator 的 AlphaEvolve。** "问 LLM 代码是否更好" 不是 fitness function。Evaluator 必须是 deterministic (确定性的) 且 fast (快速的)。
- **过度工程化。** 大多数 agent 任务都不需要这两者。先尝试 ReAct 或 ReWOO。

## 动手实现

`code/main.py` 实现了两个玩具示例：

- 一个 stdlib HTN planner，带有 operators、methods、preconditions、effects，以及当没有 method 匹配 compound task 时触发的 `LLMFallback`。这里的 "LLM" 是一个 scripted decomposer (脚本化分解器)，因此 planner 可以离线运行。
- 一个 stdlib evolutionary search，在 arithmetic programs (算术程序) 上进行搜索：生成表达式，使其输出在测试集上最小化 `|f(x) - target|`。Evaluator 是确定性的。

运行方式：

```
python3 code/main.py
```

输出 trace 展示了 HTN planner 分解 compound task 的过程（包括 mid-plan LLM fallback），以及 evolutionary loop 收敛到目标表达式的过程。

## 如何使用

- **HTN planners** —— `pyhop`、`SHOP3`，或为 domain-specific policy enforcement (领域特定策略执行) 构建自己的 planner。
- **ChatHTN** —— 研究代码；symbolic + LLM fallback 的模式可以干净地移植到任何 HTN planner。
- **AlphaEvolve** —— DeepMind 论文；ensemble + evaluator 的模式是可复现的。OpenEvolve 和类似的 open-source forks 正在出现。
- **Agent frameworks** —— 目前还没有框架原生支持 HTN 或 AlphaEvolve。将其构建为 subagent (子智能体) 或 background worker (后台工作者)。

## 输出产物

`outputs/skill-hybrid-planner.md` 生成一个 hybrid planner scaffold (混合规划器脚手架)（HTN 或 evolutionary），并明确限定 LLM 的角色范围。

## 练习题

1. 为 HTN planner 增加 backtracking (回溯)：当 operator 的 postcondition 在运行时失败，回滚并尝试下一个 method。
2. 为 ChatHTN 增加 LLM-method cache：当 LLM 在状态模式 `P` 下分解任务 `T` 时，存储结果。下次调用时先检查 method library。
3. 将 evolutionary search 的 evaluator 换成真正的 test suite。进化出一个能通过 20 个测试用例的 sort function；报告收敛所需的 generations (代数)。
4. 阅读 AlphaEvolve 的 evaluator 设计笔记。为你关心的领域设计一个 evaluator（SQL 查询优化、测试套件最小化、部署 YAML）。
5. 组合使用：用 HTN 将 compound task 分解为 subtasks，然后在每个 subtask 的 primitive operator 上使用 evolutionary search。它在哪里表现出色，在哪里过度工程化？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| HTN | "层次规划器" | 带有 operators、preconditions、effects 的 task decomposition (任务分解) |
| Method | "分解规则" | 将 compound task 分解为 subtasks 的方式 |
| Operator | "原子动作" | 具有 precondition 和 effect 的具体步骤 |
| ChatHTN | "LLM + HTN" | 当没有 method 匹配时，symbolic planner 向 LLM 求助 |
| AlphaEvolve | "进化代码搜索" | Ensemble LLM 变异代码；deterministic evaluator 进行选择 |
| Fitness function | "评估器" | 对输出进行确定性、机器可检验的评分的函数 |
| Online method learning | "缓存的 LLM 分解" | 存储并泛化 LLM 计划，以降低查询成本 |

## 延伸阅读

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) —— symbolic + LLM 混合规划器
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) —— 使用 LLM 变异的进化代码搜索
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— 何时使用规划器 vs 简单循环
