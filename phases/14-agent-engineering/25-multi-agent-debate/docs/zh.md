# Multi-Agent Debate and Collaboration

> Du et al.（ICML 2024，"Society of Minds"）运行 N 个模型实例，各自独立提出答案，然后在 R 轮中迭代地相互批评以收敛。提高事实性、规则遵守和推理能力。稀疏拓扑（sparse topology）在 token 成本上击败全连接。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 分钟

## Learning Objectives

- 解释 debate protocol（辩论协议）：N 个 proposers（提议者），R 轮，收敛到共享答案。
- 描述 debate 为何提高 factuality（事实性）、rule-following（规则遵守）和 reasoning（推理）。
- 解释 sparse topology（稀疏拓扑）：并非每个 debater 都需要看到所有其他人。
- 实现一个基于 stdlib 的 scripted LLM（脚本化 LLM）辩论，包含 full-mesh（全连接）和 sparse（稀疏）变体；测量 token cost（token 成本）vs accuracy（准确率）。

## The Problem

Self-Refine（Lesson 05）是一个模型批评自己——存在 groupthink（群体思维）风险。CRITIC（Lesson 05）将批评基于 external tools（外部工具）——并非总是可用。Debate（辩论）引入了第三种模式：多个实例、交叉批评、通过分歧收敛。

## The Concept

### Society of Minds（Du et al.，ICML 2024）

- N 个模型实例对同一个问题独立提出答案。
- 在 R 轮中，每个模型阅读其他人的 proposals（提议）并批评它们。
- 模型基于批评更新自己的答案。
- R 轮后，返回 convergent answer（收敛答案）。

原始实验使用 N=3、R=2，出于成本考虑。在难题上（MMLU、GSM8K、Chess Move Validity、biography generation），准确率随更多 agents 和更多轮次提高。

Cross-model combinations（跨模型组合）击败 single-model debates（单模型辩论）：ChatGPT + Bard 一起 > 各自单独。

### Sparse topology（稀疏拓扑）

"Improving Multi-Agent Debate with Sparse Communication Topology"（arXiv:2406.11776，2024-2025）表明 full-mesh debate（全连接辩论）并非总是最优。稀疏拓扑（star 星型、ring 环形、hub-and-spoke 轮毂-辐条）可以在更低 token 成本下匹配准确率。每个 debater 仅看到一部分 peers（同伴）。

含义：

- Full mesh N=5, R=3 = 5 × 3 = 15 proposals，每个阅读 4 个 peers = 60 critique ops。
- Star N=5, R=3（一个 hub + 4 个 spokes）= 15 proposals，spokes 仅阅读 hub = 12 critique ops。

### Debate 何时有帮助

- **Factuality（事实性）。** N 个独立 proposals，cross-check（交叉检查）减少 hallucination（幻觉）。
- **Rule-following（规则遵守）。** Chess move validity（国际象棋走法有效性）——一个模型漏掉规则，其他模型抓住它。
- **Open-ended reasoning（开放式推理）。** Multiple framings（多种框架）缩小到正确答案。

### Debate 何时有害

- **Latency-sensitive UX（延迟敏感的用户体验）。** N × R 轮串行延迟你可能承受不起。
- **Cost-sensitive scale（成本敏感的规模）。** N × R tokens per question。
- **Simple factual lookups（简单事实查找）。** 一次查找比五次辩论更便宜。

### 2026 年实际实例

- **Anthropic orchestrator-workers**（Lesson 12）—— debate 的一种变体，带有 synthesis step（综合步骤）。
- **LangGraph supervisor**（Lesson 13）—— 中心路由器 + 专家 agents 可以实现 debate 作为一个节点。
- **OpenAI Agents SDK**（Lesson 16）—— agents 来回 handoff（交接）进行 iterative critique（迭代批评）。
- **Multi-agent evals** —— 将 debate + evaluator-optimizer 配对用于 eval signal（评估信号）。

### 该模式在何处出错

- **Convergence collapse（收敛崩溃）。** 所有 agents 收敛到第一个错误答案。通过 required disagreement rounds（强制分歧轮次）缓解。
- **Hub failure（中心失败）。** 在 star topology（星型拓扑）中，一个坏的 hub 会腐蚀所有人。轮换或使用多个 hubs。
- **Prompt homogenization（提示词同质化）。** 所有 agents 使用相同的 prompt；它们产生相同的答案。使用 diverse prompts（多样化提示词）和/或 models（模型）。

## Build It

`code/main.py` 实现 stdlib debate：

- `Debater` class（带有 per-debater opinion drift 每个辩论者意见漂移的脚本化 LLM）。
- `FullMeshDebate` 和 `SparseDebate` runners。
- 三个问题：一个 factual（事实性）、一个 rule-based（基于规则）、一个 reasoning（推理性）。
- Metrics（指标）：convergent answer（收敛答案）、rounds to convergence（收敛轮次）、total critique ops（总批评操作数）。

运行方式：

```
python3 code/main.py
```

输出：per-protocol accuracy and cost（每种协议的准确率和成本）；sparse 在 2/3 问题上以更低成本匹配 full mesh。

## Use It

- **Anthropic orchestrator-workers** 用于简单 2-3-worker debates。
- **LangGraph** 用于带 checkpointing（检查点）的有状态多轮 debate。
- **Custom** 用于研究或专门的 correctness guarantees（正确性保证）。

## Ship It

`outputs/skill-debate.md` 搭建一个 multi-agent debate（多智能体辩论），包含 configurable topology（可配置拓扑）、N、R 和 convergence rule（收敛规则）。

## Exercises

1. 实现 "forced disagreement"（强制分歧）规则：在第 1 轮，每个 debater 必须产生 distinct proposal（不同的提议）。测量对 convergence speed（收敛速度）的影响。
2. 添加 confidence-weighted aggregation（置信度加权聚合）：debaters 返回 (answer, confidence)；aggregator 按 confidence 加权。有帮助吗？
3. 将一个 "agent" 替换为具有不同 opinions（意见）的不同 scripted LLM。Heterogeneity（异质性）提高准确率吗？
4. 测量你的 3 个问题上 full mesh vs sparse 的 token cost。绘制 cost vs accuracy。
5. 阅读 Society of Minds 论文。将你的玩具示例移植到 N=5, R=3。什么会损坏？什么会变好？

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Debate | "Multi-agent critique（多智能体批评）" | N proposers（提议者），R rounds（轮次）of cross-critique（交叉批评），converge（收敛） |
| Full mesh | "Everyone reads everyone（每个人阅读每个人）" | 每轮每个 debater 阅读每个 peer（同伴） |
| Sparse topology | "Limited peer view（有限同伴视图）" | Debaters 仅阅读 peers 的一个子集 |
| Hub-and-spoke | "Star topology（星型拓扑）" | 一个 central debater（中心辩论者），N-1 spokes 仅阅读 hub |
| Convergence | "Agreement（一致）" | Debaters 收敛到共享答案 |
| Society of Minds | "Du et al. debate paper（Du 等人辩论论文）" | ICML 2024 multi-agent debate method（多智能体辩论方法） |

## Further Reading

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) —— 标准 multi-agent debate（多智能体辩论）
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) —— 稀疏拓扑结果
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) —— orchestrator-workers 作为 debate 变体
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— 单模型 self-critique（自我批评）对应物
