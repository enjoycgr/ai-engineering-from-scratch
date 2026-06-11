# Tree of Thoughts and LATS: Deliberate Search（审慎搜索）

> 单条 chain-of-thought 轨迹没有回溯空间。ToT（Yao 等人，2023）将推理变成一棵树，每个节点都有 self-evaluation（自我评估）。LATS（Zhou 等人，2024）将 ToT、ReAct 和 Reflexion 统一在 Monte Carlo Tree Search（蒙特卡洛树搜索）下。Game of 24 从 CoT 的 4% 提升到 ToT 的 74%；LATS 在 HumanEval 上达到 92.7% pass@1。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 分钟

## Learning Objectives（学习目标）

- 将推理框架为搜索：节点是"thoughts（思考）"，边是"expansions（扩展）"，value 是"promising（前景如何）"。
- 用 stdlib 实现一个 ToT 风格的 BFS 树搜索，带 self-evaluation 打分。
- 扩展到一个玩具 LATS MCTS 循环，包含 select / expand / simulate / backpropagate。
- 决定搜索何时值得付出 token 倍数代价（Game of 24、代码生成），以及何时单条轨迹足够（简单问答）。

## The Problem（问题）

Chain-of-thought 是线性行走。如果第一步错了，每一步后续步骤都在错误前提下工作。在 Game of 24（用四个数字和 + − × ÷ 凑出 24）上，GPT-4 CoT 只有 4% 准确率。模型早期选错子表达式，无法恢复。

推理需要的是：提出多个候选、评估它们、挑选有前景的、在死胡同时回溯。这就是搜索。Tree of Thoughts 和 LATS 是两种经典表述。

## The Concept（概念）

### Tree of Thoughts（Yao 等人，NeurIPS 2023）

每个节点是一个连贯的中间步骤（"一个 thought"）。每个节点可以扩展为 K 个子 thought。LLM 用 scoring prompt 对每个节点做 self-evaluation。搜索探索这棵树——BFS、DFS 或 beam。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Self-evaluation 是承重的部分。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值打分、候选间投票。三种都大幅超越 CoT 在 Game of 24 上的表现（GPT-4 从 4% 提升到 74%）。

### LATS（Zhou 等人，ICML 2024）

LATS 将 ToT、ReAct 和 Reflexion 统一在 MCTS 下。LLM 扮演三个角色：

- **Policy（策略）**：提出候选下一步动作（ReAct 风格）。
- **Value function（价值函数）**：为部分轨迹打分（ToT 风格的 self-eval）。
- **Self-reflector（自我反思者）**：失败时写一篇自然语言 reflection（Reflexion 风格），并用它为未来的 rollout 重新播种。

Environment feedback（环境反馈，即 observation）混入 value function，因此搜索由真实工具结果而非仅仅是模型意见来指导。论文时的结果：GPT-4 在 HumanEval pass@1 上达到 92.7%（SOTA），WebShop 平均 75.9（GPT-3.5，接近基于梯度的 fine-tuning）。

### MCTS，极简版

每次迭代四个阶段：

1. **Select（选择）** —— 使用 UCT（upper confidence bound for trees，树的上置信界）从根走到叶子。
2. **Expand（扩展）** —— 通过 policy 生成 K 个子节点。
3. **Simulate（模拟）** —— 从子节点使用 policy rollout，用 value function（或环境奖励）为叶子打分。
4. **Backpropagate（反向传播）** —— 更新路径上的访问次数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是 exploitation（利用）；第二项是 exploration（探索）。按任务调整 `c`。

### 成本现实

搜索爆炸式消耗 token。ToT 在 Game of 24 上使用 CoT 的 100–1000 倍 token。LATS 类似。这不是免费的；将搜索保留给：

- 单条轨迹明显不足的任务（Game of 24、复杂代码）。
-  Wall-clock 不如 correctness 重要的任务。
- 有廉价可靠 value function 的任务（代码的单元测试、数学的显式目标）。

如果你的任务有单一正确答案和一个 noisy evaluator，搜索通常会让情况更糟——它会找到一个"打分高但错误"的答案。

### 2026 定位

大多数生产智能体不运行 LATS。它们运行带 tool-grounded verification 的 ReAct（CRITIC，Lesson 05）。搜索出现在专门领域：

- 用测试作为 value function 的 coding agent（HumanEval 风格）。
- 探索多条查询路径的 deep-research agent。
- LangGraph 子图内的 planning-heavy 工作流。

AlphaEvolve（Lesson 11）是 2025 年的极端：代码的进化搜索、机器可检查的 fitness、frontier 收益（56 年来首次 4x4 matmul 改进）。

## Build It（动手实现）

`code/main.py` 实现：

- 一个风格化 "pick arithmetic ops" 任务上的微型 ToT BFS。
- 同一任务上的玩具 LATS MCTS 循环（Select / Expand / Simulate / Backpropagate），带 UCT 选择。
- 一个组合 symbolic score 和 self-eval score 的 value function。

运行：

```
python3 code/main.py
```

轨迹显示 ToT 用 BFS 每节点扩展三个候选，与 LATS 通过 MCTS 收敛到最佳 rollout 的比较。两者都打印 token 数量。

## Use It（应用）

LangGraph 将 ToT 风格探索作为子图模式提供；LangChain 团队 2024 年5月的 LATS 博客是参考教程。LlamaIndex 提供一个 `TreeOfThoughts` agent。对大多数 2026 生产智能体，这个模式存在于 `if task_complexity > threshold: use_search()` 门控后面——参见 Lesson 05 的 evaluator-optimizer 模式。

## Ship It（交付）

`outputs/skill-search-policy.md` 根据任务形状、预算和评估器保真度，在线性 ReAct、ToT、LATS 和进化搜索之间做选择。

## Exercises（练习）

1. 用 UCT c=0.1 与 c=2.0 运行玩具 LATS。轨迹中有什么变化？
2. 将 value function 换成更 noisy 的 scorer（添加随机抖动）。MCTS 仍然找到最佳叶子吗？它能容忍的最小信噪比是多少？
3. 实现 beam-search ToT（每层保留 top-k）并与 BFS 比较。在紧 token 预算下哪个更好？
4. 阅读 LATS 第5.1节。复现 HumanEval 轨迹计数：需要多少次 rollout 才能达到报告的 pass@1？
5. 阅读 LATS 论文关于"LATS 帮助较少时"的讨论。写一个单段落决策规则，将任务形状映射到搜索策略。

## Key Terms（关键术语）

| 术语 | 人们常说 | 实际含义 |
|------|---------|---------|
| Tree of Thoughts | "Branching CoT（分支式思维链）" | Yao 等人 —— 带 self-evaluation 的 thought 节点树 |
| LATS | "MCTS for LLMs（LLM 的 MCTS）" | Zhou 等人 —— 将 ToT + ReAct + Reflexion 统一在 MCTS 下 |
| UCT | "Upper confidence bound（上置信界）" | Select 公式，平衡 exploitation（Q）和 exploration（ln N / n） |
| Value function | "How good is this state（这个状态多好）" | Prompted LLM score 或环境奖励；反馈给 backprop |
| Policy | "Action proposer（动作提议者）" | ReAct 风格生成器；发出候选下一步 thought/action |
| Rollout | "Simulated trajectory（模拟轨迹）" | 从节点走到叶子，使用 policy，用 value 打分 |
| Backpropagate | "Update ancestors（更新祖先）" | 将叶子的奖励推上路径，更新访问次数和 Q |
| Search cost | "Token explosion（Token 爆炸）" | Game of 24 上是 CoT 的 100-1000 倍；采用前先算好预算 |

## Further Reading（延伸阅读）

- [Yao 等人, Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) —— 经典论文
- [Zhou 等人, LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) —— 带 Reflexion 反馈的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) —— 搜索的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) —— 带程序化评估器的进化搜索
