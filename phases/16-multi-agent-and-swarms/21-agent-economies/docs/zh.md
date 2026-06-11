# Agent Economies, Token Incentives, Reputation

> 长期自主运行的 agent (智能体)（METR 的 1 小时到 8 小时工作曲线）需要经济主体能力。正在形成的 **5 层栈** 是：**DePIN** (物理计算) → **Identity** (W3C DIDs + 信誉资本) → **Cognition** (RAG + MCP) → **Settlement** (账户抽象) → **Governance** (Agentic DAOs)。生产级 agent 激励网络包括 **Bittensor** (TAO 子网奖励任务专用模型)、**Fetch.ai / ASI Alliance** (ASI-1 Mini LLM + FET token) 和 **Gonka** (基于 transformer 的 PoW，将计算重新分配给生产性 AI 任务)。学术方面：AAMAS 2025 的去中心化 LaMAS 使用 **Shapley-value credit attribution (Shapley 值信用归因)** 来公平奖励贡献 agent；Google Research "Mechanism design for large language models" 提出在 monotone aggregation (单调聚合) 下的 **token auctions (代币拍卖)** 并采用 second-price payment (二价支付)。本课构建一个最小化的 agent marketplace (智能体市场)，将 Shapley-value credit attribution 应用于 multi-agent (多智能体) 流水线，并运行一个 second-price token auction (二价代币拍卖)，让博弈论机制具体落地。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 minutes

## Problem

当 agent 共同创造价值但需要单独奖励时，multi-agent systems (多智能体系统) 会变得复杂。经典机制——均分、最后贡献者全拿——要么不公平，要么可被操纵。通过 Shapley values (Shapley 值) 进行基于联盟的奖励在构造上是公平的，但计算成本高昂。2025–2026 年的文献推动了有用的近似方法：Shapley sampling (Shapley 采样)、monotone aggregation auctions (单调聚合拍卖) 以及基于链上确认的声誉积累。

除了信用归因，该领域已经转向实际的经济 agent：Bittensor TAO 奖励用于微调子网专用模型的挖矿计算，Fetch.ai/ASI 奖励使用 FET token 的 ASI-1 Mini LLM 推理，Gonka 将 transformer proof-of-work (工作量证明) 重新分配给生产性 AI 任务。自主交易的 agent 今天已经存在；问题是如何对齐激励。

本课将 agent economies (智能体经济) 视为一个特定的问题族——信用归因、机制设计和声誉——并用最小化的数学构建每一个，以便让概念深入人心。

## Concept

### 5 层 agent-economy stack (智能体经济栈)

1. **DePIN (physical compute) (物理计算)。** 去中心化基础设施，出租 GPU、存储、带宽。Bittensor 子网、Render Network、Akash。不特定于 agent；agent 使用它。
2. **Identity (身份)。** W3C Decentralized Identifiers (DIDs) (去中心化标识符) 为每个 agent 提供一个独立于任何平台的持久 ID。声誉积累到 DID 上。Agent Network Protocol (ANP) 使用 DID 作为发现层。
3. **Cognition (认知)。** Agent 的推理循环：LLM + RAG + MCP。这是其他阶段所构建的内容。
4. **Settlement (结算)。** Account abstraction (ERC-4337) (账户抽象) 让 agent 从自己的余额支付 gas，无需持有 ETH。Agent 可以为服务、其他 agent 或计算付费。
5. **Governance (治理)。** Agentic DAOs (智能体 DAO)：人类 *和* agent 对协议变更进行投票的治理结构，投票权与声誉挂钩。

并非每个生产系统都使用全部五层。Bittensor 使用 1、2、部分 3、部分 4，不使用 5。OpenAI agent 只使用 3。该栈是一个参考地图，而非要求。

### Bittensor、Fetch.ai、Gonka —— 实际运行的系统

**Bittensor (TAO)。** 子网是专业化任务（语言建模、图像生成、预测）。矿工提交模型输出。验证者对其进行排名；基于质押权重的评分分配 TAO 奖励。每个子网有自己的评估。经济学教训：为任务特定的输出质量付费，而非使用的计算量。

**Fetch.ai / ASI Alliance。** ASI-1 Mini LLM 运行在 Fetch.ai 网络上；用户用 FET token 支付推理费用。这里的 agent-as-peers (agent 即对等节点) 叙事更强：Fetch 上的一个 agent 可以调用另一个 agent 执行任务并用 FET 付费。

**Gonka。** Transformer proof-of-work (工作量证明)："工作"是 transformer 的前向传播。矿工通过运行具有已知正确输出（来自训练数据）的推理任务来赚取收益。资源生产性 PoW，而非基于哈希的 PoW。

截至 2026 年 4 月，这三个都是生产级的。收益分配不同。Bittensor 奖励相对于子网验证者的质量；Fetch 奖励按付费用户衡量的效用；Gonka 奖励可验证的推理工作。

### Shapley-value credit attribution (Shapley 值信用归因)

三个 agent 协作完成一项任务。输出评分为 0.8。各自贡献了多少？

Shapley value (Shapley 值)：满足四个公理（效率、对称性、线性、零 agent）的唯一信用分配。对于 agent `i`：

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

其中 `S_i_O` 是在排序 `O` 中位于 `i` 之前的 agent 集合。实践中：枚举所有排列，记录每个 agent 在每个排列中的边际贡献，取平均。

对于 N=3 个 agent，有 6 种排列。对于 N=10，有 360 万种——所以实践中你采样排序而非枚举。

### Second-price auction for aggregation (聚合的二价拍卖)

Google Research（"Mechanism design for large language models"）提出用于聚合 LLM 输出的 second-price token auctions (二价代币拍卖)。设定：N 个 agent 各提出一个补全；每个 agent 对被选中有一个私有估值。拍卖者选择最高估值的提案并支付 *第二高* 的估值。在 monotone aggregation (单调聚合) 下（价值取决于选择哪个提案，而非有多少个出价），这是 truthfulness (真实激励) 的——agent 会报出真实估值。

这对 LLM 系统为何重要：你可以将补全任务外包给多个定价不同的 agent；拍卖选出最好的 + 公平支付，且 agent 没有虚报的动机。

### Reputation capital (信誉资本)

一个绑定到 DID 的声誉分数从已确认的贡献中积累。简单的更新规则：

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

衰减因子 `alpha` 接近 1。声誉：

- 读取成本低，可用于路由决策（"将困难任务发送给高声誉 agent"）。
- 伪造成本高（随时间积累，绑定到 DID）。
- 可以被 slash (削减)：未通过验证的贡献会扣分。

### AAMAS 2025 decentralized LaMAS

LaMAS 提案（AAMAS 2025）结合了：DID 身份、Shapley-value credit attribution 和简单拍卖机制。核心主张：将信用归因步骤去中心化，使系统可审计并免疫单点操纵。

### 经济学在何处失效

- **Price oracle manipulation (价格预言机操纵)。** 如果信用函数可被操纵，agent 就会操纵它。每个机制都需要对抗性测试。
- **Sybil attacks (女巫攻击)。** 一个运营商启动 N 个假 agent 来膨胀自己的贡献。DIDs 减缓但无法阻止；声誉的伪造成本是缓解措施。
- **Verification cost (验证成本)。** 信用归因的公平性取决于验证者。如果验证便宜（小 LLM），它可被操纵；如果昂贵（人工 panel），系统无法扩展。
- **Regulatory overhang (监管阴影)。** Agent economies 与金融监管相交。截至 2026 年，Bittensor、Fetch 和 Gonka 在某些司法管辖区处于法律灰色地带。

### Agent economies 何时有意义

- **具有异构运营商的开放网络。** 没有单一团队控制所有 agent。
- **可验证的输出。** 没有验证，信用归因就是猜测。
- **长期工作流。** 一次性任务无法从声誉积累中受益。
- **Tokenized payments (代币化支付) 在你的司法管辖区合法。**

在封闭的企业系统中，经济学让位于更简单的分配（经理分配工作，指标是内部的）。经济学文献主要适用于开放网络。

## Build It

`code/main.py` 实现了：

- `shapley(value_fn, agents)` —— 对小 N 通过枚举进行精确 Shapley 计算。
- `second_price_auction(bids)` —— truthfulness (真实激励) 机制；获胜者支付第二高价。
- `Reputation` —— 绑定 DID 的声誉，具有指数衰减和 slashing (削减)。
- Demo 1：三个 agent 协作，精确 Shapley 归因信用。
- Demo 2：五个 agent 竞标任务槽；second-price auction (二价拍卖) 选出获胜者 + 支付。
- Demo 3：100 轮任务分配给具有异构声誉的 agent；在 warmup 后，reputation-weighted routing (声誉加权路由) 优于随机分配。

运行：

```
python3 code/main.py
```

预期输出：每个 agent 的 Shapley 值；显示 truthfulness-bid equilibrium (真实出价均衡) 的拍卖结果；在 warmup 后 reputation-weighted routing 比随机分配质量提升 10–20%。

## Use It

`outputs/skill-economy-designer.md` 设计了一个最小化的 agent economy (智能体经济)：身份层选择、信用归因机制、支付机制、声誉规则。

## Ship It

2026 年运行 agent economy (智能体经济)：

- **从声誉开始，而非代币。** 声誉实现成本低且本身就有价值；代币增加法律和经济复杂性。
- **先验证，后奖励。** 永远不要在没有独立验证步骤的情况下分配信用。自我报告的质量会积累 sybil games (女巫游戏)。
- **Shapley-sample，而非 Shapley-exact。** 采样 100–1000 个排序；精确枚举无法扩展。
- **限制衰减因子和声誉下限。** 无界衰减会抹除合法贡献者；衰减太慢会奖励过时的旧高声誉 agent。
- **对抗性地审计机制。** 在开放网络前运行 red-team (红队) 场景。每个机制都有博弈论；你要找到漏洞，而非攻击者。

## Exercises

1. 运行 `code/main.py`。确认 Shapley 值之和等于总价值（效率公理）。更改价值函数；Shapley 分配是否按预期方向变化？
2. 实现 Shapley *sampling* (采样)（对 K 个排序进行蒙特卡洛）。K 如何影响近似精度？与 N=4 时的精确值比较。
3. 在拍卖前实现一个 coalition-forming (联盟形成) 步骤：agent 可以合并为团队并作为单位出价。哪些联盟会形成？结果是否比个人出价 Pareto-better (帕累托更优)？
4. 阅读 Google Research 的机制设计文章。找出一个如果被违反就会破坏 truthfulness (真实性) 的假设。在 LLM 设置中，这种 failure mode (故障模式) 是什么样的？
5. 阅读 AAMAS 2025 去中心化 LaMAS 论文。在合成任务上为 10 个 agent 实现他们的 Shapley 步骤。精确计算需要多长时间？100 次抽样的采样结果有多接近？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| DePIN | "Decentralized physical infrastructure" | Token-incentivized compute/storage/bandwidth. Bittensor, Akash, Render. |
| DID | "Decentralized identifier" | W3C spec for portable IDs. Agent reputation binds to DID, not to a platform. |
| ERC-4337 | "Account abstraction" | Contract accounts that can sponsor gas, enabling agent payments. |
| Shapley value | "Fair credit attribution" | Unique allocation satisfying efficiency, symmetry, linearity, null. |
| Second-price auction | "Vickrey auction" | Truthful mechanism: winner pays second-highest bid. Monotone aggregation compatible. |
| Reputation capital | "Accumulated quality score" | DID-bound score from confirmed contributions; decays over time. |
| Agentic DAO | "Agents + humans govern" | DAO with agent voters as first-class, voting power tied to reputation. |
| TAO / FET / GPU credits | "Token denominations" | Bittensor TAO, Fetch.ai FET, various DePIN tokens. |

## Further Reading

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — 2026 survey of the 5-layer agent-economy stack
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — token auctions with monotone aggregation
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — Shapley-value credit attribution
- [Bittensor TAO documentation](https://docs.bittensor.com/) — subnet structure and reward distribution
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM and FET token
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — identity foundation
