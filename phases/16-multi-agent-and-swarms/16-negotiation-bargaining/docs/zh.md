# 协商与讨价还价 (Negotiation and Bargaining)

> agent (智能体) 协商资源、价格、任务分配和条款。2026 年的基准测试集已经明确：NegotiationArena (arXiv:2402.05863) 表明，LLM 可以通过角色操控（"desperation"）将收益提高约 20%；"Measuring Bargaining Abilities" (arXiv:2402.15813) 表明买方比卖方更难，而规模并无帮助——其 **OG-Narrator**（确定性报价生成器 + LLM 叙述者）将成交率从 26.67% 提升至 88.88%；Large-Scale Autonomous Negotiation Competition (arXiv:2503.06416) 运行了约 18 万次协商，发现 **chain-of-thought-concealing (思维链隐藏)** agent 通过向对手隐藏推理过程而获胜；Bhattacharya 等人 2025 年基于 Harvard Negotiation Project 指标进行排名，Llama-3 最有效，Claude-3 最具攻击性，GPT-4 最公平。本节课实现 Contract Net Protocol (FIPA 的前身，第 02 课)，接入 LLM 风格的买方/卖方，运行 OG-Narrator 风格的分解，并衡量每种结构选择如何改变成交率。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (FIPA-ACL Heritage), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 分钟

## 问题

两个 agent 需要就价格达成一致。如果仅使用纯语言提示，2024-2026 年的 LLM 成交率低得惊人（在 arXiv:2402.15813 的紧密参数化讨价还价中约为 27%）。规模无法解决：GPT-4 在讨价还价方面并不比 GPT-3.5 有结构性优势；它只是更擅长讨价还价的*语言*。

根本问题在于 LLM 混淆了两个工作——决定报价和叙述报价。OG-Narrator 将它们分开：确定性报价生成器计算数字动作；LLM 只负责叙述。成交率跃升至约 89%。

这反映了经典的多 agent 发现：将机制与通信层解耦会获胜。Contract Net Protocol (FIPA, 1996; Smith, 1980) 是参考任务市场机制。将 LLM 插入叙述槽位，你就得到了一个现代 LLM 驱动的任务市场。

## 概念

### Contract Net，一段话

Smith 1980 年的 Contract Net Protocol：一个 **manager (管理者)** 广播 **call for proposals (cfp，征求建议书)**；**bidders (投标者)** 以包含其报价的 **propose (提议)** 消息回应；manager 选择获胜者并向获胜者发送 **accept-proposal (接受提议)**，向失败者发送 **reject-proposal (拒绝提议)**。获胜者执行工作。可选消息：**refuse (拒绝)**（投标者拒绝提议）。FIPA 将其编纂为 `fipa-contract-net` 交互协议。

### 为什么 OG-Narrator 获胜

"Measuring Bargaining Abilities of Language Models" (arXiv:2402.15813) 观察到：

- LLM 经常违反讨价还价规则（以荒谬的价格报价，忽略对方的 ZOPA）。
- 它们的锚定很差（接受糟糕的首轮报价；还价时使用象征性而非战略性金额）。
- 仅靠规模无法解决这些问题。更大的模型产生更合理的语言，但战略错误相似。

OG-Narrator 分解：

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

报价生成器是一种经典的协商策略：Rubinstein bargaining 模型、Zeuthen 策略或简单的价格 tit-for-tat。LLM 负责叙述。消息包含确定性价格和自然语言框架。

成交率跃升的原因是：
- 价格保持在协商区间内。
- 锚定是战略性的，而非情绪化的。
- LLM 做它擅长的事：写作。

### NegotiationArena 发现

arXiv:2402.05863 提供了权威基准。主要发现：

- LLM 可以通过采用角色（"我迫切需要在周五前卖掉这个"）将收益提高约 20%——角色操控是一种真实策略。
- 公平/合作的 agent 被对抗性 agent 利用；防御需要明确的反姿态。
- 对称配对在大约 40% 的基准场景中汇聚到不公平的结果。

这不是"LLM 不擅长协商"。而是"LLM 协商得太像人类，包括可利用的部分"。

### Chain-of-thought concealment (思维链隐藏)

Large-Scale Autonomous Negotiation Competition (arXiv:2503.06416) 在许多 LLM 策略中运行了约 18 万次协商。获胜者向对手隐藏了他们的推理过程：

- 如果 agent 在公开可见的 scratchpad 中打印"我只出到 75 美元；我的保留价是 70 美元"，对手会读到它。
- 获胜者私下计算策略；输出通道只包含报价和最低限度的必要叙述。

这是经典博弈论（Aumann 1976 关于理性和信息）的 2026 年回响：透露你的私人估值会损失收益。LLM 没有直觉到这一点，会愉快地将保留价输入到推理痕迹中，而这些痕迹对对手可见。

工程要点：将 private-scratchpad 上下文与 public-message 上下文分开。这不是可选项。

### Bhattacharya 等人 2025 —— 模型排名

基于 Harvard Negotiation Project 指标（principled negotiation、BATNA 尊重、interest reciprocity）：

- **Llama-3** 在达成交易方面最有效（成交率 + 收益）。
- **Claude-3** 是最具攻击性的协商者（高锚定、晚让步）。
- **GPT-4** 最公平（配对间收益方差最小）。

这是 2025 年的快照。关键不在于 2026 年 4 月哪个模型获胜——而在于不同的基础模型具有持久的协商风格。异构集成（第 15 课）将其作为多样性来源。

### 通过 Contract Net + LLM 进行任务分配

Contract Net 在现代 LLM 多 agent 中的重用：

1. Manager agent 将任务分解为单元。
2. 向 worker agent 广播包含任务描述的 `cfp`。
3. 每个 worker 返回报价：`(price, eta, confidence)`，其中 price 可以是 token、计算单元或美元。
4. Manager 选择获胜者（单个或多个，取决于任务）并授予。
5. 被拒绝的 worker 可以自由地对其他任务投标。

这可以很好地扩展到超过 100 个 worker，因为协调是广播-响应，而不是同步聊天。已用于生产：Microsoft Agent Framework 的 orchestration (编排) 模式，一些 LangGraph 实现。

### LLM-Stakeholders Interactive Negotiation

NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) 引入了具有 **secret scores (秘密分数)** 和 **minimum-acceptance thresholds (最低接受阈值)** 的多方可评分游戏。每个 stakeholder 都有私人效用；LLM 必须从消息中推断它们。这是从双方讨价还价到 N 方 coalition formation (联盟形成) 的推广。与具有异构 worker 能力的生产任务市场相关。

### 叙述与机制的规则

在所有 2024-2026 年的协商基准中，一致的工程规则是：

> 让 LLM 叙述。不要让 LLM 计算报价。

如果报价需要是数字（价格、ETA、数量），从协商状态确定性生成，并让 LLM 生成框架。如果报价需要是提议结构（任务分解、角色分配），让 LLM 起草，但在发送前根据 schema 进行验证和约束检查。

## 构建

`code/main.py` 实现：

- `ContractNetManager`、`ContractNetTask`、`Bid` —— manager + bidders，广播 cfp，收集 proposals，授予。
- `og_narrator_bargain(state, rng)` —— OG-Narrator 买方：向中点的确定性 Zeuthen 风格让步。
- `seller_response(state, rng)` —— 确定性卖方还价策略（两种风格的结构真实值）。
- `naive_llm_bargain(state, rng)` —— 模拟全 LLM 协商者：以高方差选择价格，通常在 ZOPA 之外。
- 测量：在 1000 次试验中，每次试验采样新的保留价格。

运行：

```
python3 code/main.py
```

预期输出：naive-LLM 成交率约 65-75%；OG-Narrator 成交率约 85-95%；15-25 个百分点的差距是将报价生成与叙述分解的结构优势。加上一个包含三个 bidder 和一个任务的 Contract Net 任务市场分配示例。

## 使用

`outputs/skill-bargainer-designer.md` 设计了一个协商协议：谁生成报价（确定性或 LLM），谁叙述，private scratchpad 如何与 public message 分离，以及如何监控成交率。

## 交付

生产协商清单：

- **Separate scratchpad (分离草稿本)。** 私人状态永远不会到达对手的上下文。这是不可协商的。
- **Deterministic offer generation (确定性报价生成)。** 价格、数量、ETA：计算，不要提示。
- **Validate all incoming offers (验证所有传入报价)** 是否符合 schema。在协议边界拒绝超出 ZOPA 的报价。
- **Bound rounds (限制轮次)。** 双方最多 3-5 轮；僵局时升级到 mediator (调解人)。
- **Measure deal rate and payoff variance (持续测量成交率和收益方差)。** 下降的成交率是症状——通常是提示漂移或对手攻击。
- **Log all rejected proposals (记录所有被拒绝的提议)** 并附上确定性理由。对于 Contract Net manager，失败的 bidder 需要理解原因。

## 练习

1. 运行 `code/main.py`。确认 OG-Narrator 在成交率上击败 naive-LLM。差距是多少？
2. 实现 **persona-based payoff improvement (基于角色的收益提升)** (arXiv:2402.05863) —— 买方仅在叙述中采用"本周迫切想购买"的角色，报价生成器不变。成交率或收益会改变吗？
3. 实现 chain-of-thought **concealment (隐藏)**：维护一个不传递给对手的 private scratchpad 字符串。如果你不小心泄露了它（通过交换通道来模拟）会发生什么？
4. 将 Contract Net 扩展到带有保留价的 N-bidder auction (拍卖)。当所有出价都超过保留价时，manager 如何在最低价格和最高质量之间做出决定？你选择哪种授予规则，为什么？
5. 阅读 Bhattacharya 等人 2025 年关于 Harvard Negotiation Project 指标的内容。实现两种不同风格（aggressive vs fair）的协商者。测量对称和非对称配对下的收益方差。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Contract Net | "任务市场" | Smith 1980, FIPA 1996。cfp + propose + accept/reject。经典的任务市场。 |
| ZOPA | "可能达成协议的区域" | 买方最高出价和卖方最低要价之间的重叠。超出此范围的报价无法成交。 |
| BATNA | "谈判协议的最佳替代方案" | 如果这笔交易失败，你的备选方案。设定你的保留价格。 |
| OG-Narrator | "报价生成器 + 叙述者" | 分解：确定性报价，LLM 叙述。 |
| Zeuthen strategy | "风险最小化让步" | 基于风险限制让步的经典报价生成器。 |
| Rubinstein bargaining | "交替报价均衡" | 具有贴现的无限期协商的博弈论模型。 |
| CoT concealment | "隐藏你的推理" | arXiv:2503.06416 的获胜者保留 private scratchpad；公共通道只显示报价。 |
| Persona manipulation | "情绪姿态" | arXiv:2402.05863：通过绝望/紧迫角色获得约 20% 的收益提升。 |

## 延伸阅读

- [NegotiationArena](https://arxiv.org/abs/2402.05863) —— 基准测试；角色操控和利用发现
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) —— OG-Narrator 和买方比卖方更难的结果
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) —— 约 18 万次协商；思维链隐藏获胜
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) —— 具有秘密效用的多方可评分游戏
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) —— 经典机制，IEEE Transactions on Computers
