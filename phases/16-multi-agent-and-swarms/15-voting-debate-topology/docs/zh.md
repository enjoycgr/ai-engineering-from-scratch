# 投票、自洽性与辩论拓扑结构

> 最廉价的聚合方式：采样 N 个独立智能体，多数投票。Wang et al. 2022 的自洽性（self-consistency）方法使用同一个模型采样 N 次来实现这一点。多智能体（multi-agent）系统将其扩展为**异构**（heterogeneous）智能体，以摆脱单一文化（monoculture）——不同的模型、不同的提示词、不同的温度参数、不同的上下文。在多数投票之外，辩论拓扑结构（debate topology）也很重要：MultiAgentBench（arXiv:2503.01935，ACL 2025）评估了星型（star）、链式（chain）、树形（tree）和图状（graph）协调结构，发现**图状结构最适合研究类任务**，但在约 4 个智能体之后会出现“协调税”（coordination tax）。AgentVerse（ICLR 2024）记录了两种涌现模式——自愿行为（volunteer behaviors）和从众行为（conformity behaviors）——从众既是特性（达成共识）也是风险（群体思维，见第 24 课）。本课将梳理拓扑结构空间，构建每种变体，并测量协调税。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 分钟

## Problem

辩论可以提高准确率（Du et al., arXiv:2305.14325）。它也可能降低准确率。辩论是否有帮助取决于四个结构性选择：

1. 谁与谁对话（拓扑结构）。
2. 多少轮次（Du 2023：轮次和智能体数量独立起作用）。
3. 智能体是否为异构（不同的基础模型打破单一文化）。
4. 是否存在对抗性声音（steel-manning vs. straw-manning）。

那些把“运行 5 个智能体然后投票”简单套用到任务上的团队，往往比单个智能体表现更差。这些失败并非随机，而是与拓扑结构和异构性相关。本课就是这张拓扑地图。

## Concept

### 自洽性，单模型基线

Wang et al. 2022（"Self-Consistency Improves Chain of Thought Reasoning"）在温度 > 0 的情况下对同一模型采样 N 次，并对推理路径的答案进行多数投票。在 GSM8K 上的结果：N=40 次采样相比单次贪心解码有显著提升。自洽性是单智能体（single-agent）向多智能体投票演进的前身。

局限：自洽性使用单一基础模型。错误在构造上就是相关的。如果模型存在系统性偏差，所有 N 次采样都会共享它。

### 多智能体投票，异构扩展

将 N 次采样替换为 N 个*不同*的智能体。不同的基础模型（Claude、GPT、Llama）、不同的提示词、不同的工具访问权限。好处：错误互不相关。代价：不同智能体的成本不同；协调它们会增加开销。

异构辩论在 2026 年的规范名称是 **A-HMAD** —— Adversarial Heterogeneous Multi-Agent Debate（对抗性异构多智能体辩论）。虽未 universally adopted，但论文用该术语指代“不同模型辩论，从而减少单一文化崩溃带来的相关错误”。

### 四种拓扑结构

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

星型（Star）：一个中心节点，其他所有节点只与中心通信。等价于没有后通道的监督者-工作者模式。
链式（Chain）：线性结构，每个智能体只看到前一个的输出。类似流水线。
树形（Tree）：层级结构，用于分层智能体系统（第 06 课）。
图状（Graph）：任意对任意。包括全连接团（fully-connected clique）和任意有向无环图（DAGs）。

### 协调税（MultiAgentBench）

MultiAgentBench（MARBLE，ACL 2025，arXiv:2503.01935）在包含研究、编码和规划的任务套件上基准测试了星型、链式、树形和图状结构。关键测量结果：

- **图状**拓扑在研究任务上表现最佳。信息任意流动；智能体可以互相批评。
- **星型**在快速回答事实性任务上表现最佳。中心节点过滤并整合信息。
- **链式**在分步流水线（分阶段细化）上表现最佳。
- **协调税**在图状拓扑约 4 个智能体之后出现。挂钟时间和 token 成本增长快于质量提升。

4 智能体上限是经验性的，并非根本性限制。它反映了 2026 年 LLM 的上下文容量：每个智能体的上下文被同伴的输出填满，一旦每个人都能看见每个人，增加第 N+1 个智能体的边际价值就会下降。

### 多智能体辩论策略（"Should we be going MAD?"）

arXiv:2311.17371 是 2023 年 MAD 策略的综述。被其他研究复现的关键发现：在结构上与自洽性相似（独立采样 + 聚合）的 MAD 变体，在相同预算下往往跑输自洽性。MAD 在智能体真正异构且辩论具有对抗性结构（一个智能体提出反对意见）时帮助最大。

### AgentVerse 涌现模式

AgentVerse（ICLR 2024，https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）记录了两种即使在没有显式设计的情况下也会从多智能体辩论中涌现的行为：

- **自愿（Volunteer）。** 一个智能体主动提供帮助（"我可以负责下一步"），无需提示。有用之处在于：它将工作分配给最适合该子任务的智能体。
- **从众（Conformity）。** 一个智能体调整自己的立场以匹配批评者，即使批评者是错的。这是辩论中等价于谄媚（sycophancy，第 14 课）的现象。

从众是“辩论直到达成一致”会奖励霸凌者的原因。限制轮次并设置独立的裁判可以缓解。

### 异构性：真正推动准确率的旋钮

2024-2026 年实用文献中的一个模式：将你的 N 个智能体中的一个换成不同的基础模型，带来的准确率提升比将 N 增加 1 更大。直觉是单一文化——每一个新的独立错误来源都比一个额外的相关样本更有价值。

在极限情况下，异构性（heterogeneity）胜过数量（numerosity）。在大多数有明确正确答案的任务上，三个不同的模型胜过五个同一模型的副本。

### 陪审团方法

Sibyl 框架（在 Minsky-LLM 文献中被引用）将“陪审团”形式化——一小群专业智能体通过在每个阶段投票来精炼答案。与简单的多数投票不同，陪审团有角色：一个智能体交叉质询，一个提供上下文，一个评分合理性。陪审团方法是简单投票（便宜，但易受单一文化影响）和完整 MAD（昂贵，易从众）之间的中间点。

### 投票+辩论何时占优

- 问题有明确答案（事实、数学、代码行为）。投票收敛有意义。
- 智能体可以访问不同的来源或工具（异构性可用）。
- 轮次有界（通常 2-3 轮）且有独立的裁判或验证者。
- 预算允许 3-5 个智能体。超过 5-7 个智能体使用图状拓扑，协调税将占主导。

### 投票+辩论何时有害

- 问题是观点导向的。智能体收敛到看起来最自信的答案，而非最正确的答案。
- 所有智能体共享同一基础模型。单一文化使共识失去意义。
- 轮次无界。从众每次都会赢。
- 任务很简单。单个智能体使用 N=5 的自洽性更便宜且同样准确。

## Build It

`code/main.py` 实现了：

- `run_star(agents, hub, question)` —— 中心节点轮询每个工作者，聚合结果。
- `run_chain(agents, question)` —— 顺序细化。
- `run_tree(root, children, question)` —— 层级结构，深度为 2 的聚合。
- `run_graph(agents, question, rounds)` —— 全对全辩论，有界轮次。
- 一个脚本化的异构性调节器：每个智能体有一个 `error_bias`，表示其系统性错误倾向。
- 一个测量工具，在 N=3、5、7 下运行每种拓扑结构，报告（准确率、总 token 数、模拟挂钟时间）。

运行：

```
python3 code/main.py
```

预期输出：一张拓扑结构 × N →（准确率、token 数、延迟）的表格。图状结构在研究风格任务上于 N=3-5 时获胜；星型在快速事实性任务上获胜；图状结构在 N=7 时显示出协调税（延迟膨胀快于准确率提升）。

## Use It

`outputs/skill-topology-picker.md` 是一个技能文件，它读取任务描述并推荐一种拓扑结构（星型/链式/树形/图状）、N（智能体数量）、异构性配置（使用的基础模型）和轮次上限。

## Ship It

对于任何集成系统：

- 从使用一个强基础模型的 **N=5 自洽性**开始。这是便宜的基线。
- 如果准确率重要，升级到 **N=3 的异构投票**。测量增量。
- 只有当任务有结构（研究、多步骤）且可行限制轮次时，才升级到**辩论拓扑结构**。
- 始终记录少数派集群。当少数派持续正确时，你就获得了一个多样性信号。
- 同时基准测试挂钟时间和 token 数与准确率。“准确率更好但成本 10 倍”是一个商业决策。

## Exercises

1. 运行 `code/main.py`。绘制图状拓扑的协调税曲线：准确率 vs N，token 数 vs N。曲线在何处拐点？
2. 实现 A-HMAD：三个具有故意不同偏差的智能体。在来自第 14 课的单一文化攻击下，全同偏差基线与 A-HMAD 相比如何？
3. 为图状拓扑添加一个“裁判”角色，该角色不投票，只评分最终共识。这会改变涌现的从众行为吗？
4. 阅读 AgentVerse 论文（ICLR 2024）。识别你的实现最强烈地表现出哪种涌现行为。你能通过提示词改变来诱发相反的行为吗？
5. 阅读 MultiAgentBench（arXiv:2503.01935）第 4 节（拓扑实验）。使用你的工具在论文的一个任务上复现“图状结构赢得研究任务”的结果。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Self-consistency | "Sample N times, vote" | Wang 2022. 单一模型，N 次温度>0 采样，对推理路径进行多数投票。 |
| Heterogeneity | "Different models" | 不同基础模型或提示词家族的集成。打破单一文化。 |
| MAD | "Multi-agent debate" | 智能体在多轮中交换批评的通用术语。见 Du 2023。 |
| A-HMAD | "Adversarial Heterogeneous MAD" | 强调不同模型 + 对抗性结构的 MAD 变体。 |
| Topology | "Who talks to whom" | 星型、链式、树形、图状。决定信息流。 |
| Coordination tax | "Diminishing returns" | 图状结构超过约 4 个智能体后，成本增长快于质量提升。 |
| Volunteer behavior | "Unprompted help" | AgentVerse 涌现模式：一个智能体主动提出承担某一步。 |
| Conformity behavior | "Agreement under pressure" | AgentVerse 涌现模式：一个智能体与批评者保持一致。 |
| Jury | "Small specialized panel" | Sibyl 风格的集成，带有角色（质询者、上下文提供者、评分者）。 |

## Further Reading

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) —— 单模型基线
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) —— 智能体数量和轮次都独立起作用
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) —— 拓扑基准测试，显示图状结构最适合研究，链式适合流水线
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) —— MAD 策略综述；发现 MAD 在同等预算下往往输给自洽性
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) —— 自愿和从众涌现模式
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) —— 参考基准测试实现
