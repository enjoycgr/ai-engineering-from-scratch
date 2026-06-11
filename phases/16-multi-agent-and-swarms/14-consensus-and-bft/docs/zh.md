# Consensus and Byzantine Fault Tolerance for Agents (智能体共识与拜占庭容错)

> 经典分布式系统的 BFT (拜占庭容错) 遇到了随机性 LLM。2025-2026 年出现了三个研究方向：**CP-WBFT** (arXiv:2511.10400) 通过 confidence probe (置信度探测) 为每票加权；**DecentLLMs** (arXiv:2507.14928) 采用无领导者的并行 worker proposal (工作者提案) 与 geometric-median (几何中位数) 聚合；**WBFT** (arXiv:2505.05103) 将加权投票与 Hierarchical Structure Clustering (层次结构聚类) 相结合，将节点划分为 Core (核心) 和 Edge (边缘)。"Can AI Agents Agree?" (arXiv:2603.01213) 的诚实经验结果是，即使是标量共识在今天也很脆弱——单个欺骗性 agent 就能破坏 Mixture-of-Agents。BFT 是必要但不充分的。本课构建了一个最小化的 BFT 协议，注入三种 agent 特有的攻击（Byzantine lie (拜占庭谎言)、sycophantic conformity (谄媚性从众)、correlated-error monoculture (相关错误单一文化)），并衡量每种共识变体的应对能力。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 分钟

## Problem (问题)

你有 N 个 LLM agent，每个都产生一个答案。它们意见不一致。Majority vote (多数投票) 选错了，因为两个 agent 是 correlated (相关的)（相同的基础模型、相同的训练数据、相同的故障模式）。第三个 agent 碰巧以一种新颖的方式错了——所以多数是一个 false majority (虚假多数)。

现在加入一个欺骗性 agent：它故意撒谎。或者一个 sycophantic agent (谄媚智能体)：它同意最后发言的人。在经典 BFT 中，假设是 Byzantine nodes (拜占庭节点) 占比例 `f < n/3` 且行为任意。2026 年的现实是，LLM 节点即使在诚实的时候也是随机的，跨模型相关，并且相互影响输出。你不能把它们当作独立的 Bernoulli voters (伯努利投票者)。

经典 BFT (PBFT, 1999) 并没有错——它是不完整的。它能处理任意的 bit-flipping (比特翻转)。它不能处理"三个诚实 agent 因为共享训练数据而共享一个幻觉"。本课从 PBFT 的基础出发，并叠加三种 2025-2026 年的适配。

## Concept (概念)

### What classical BFT gives you (经典 BFT 能提供什么)

Practical Byzantine Fault Tolerance (Castro & Liskov, OSDI 1999) 容忍 `f < n/3` 的 Byzantine nodes。协议有三个阶段（pre-prepare、prepare、commit）和两个原语（signed messages (签名消息)、quorum certificates (法定人数证书)）。在 `n >= 3f + 1` 个诚实或恶意节点之间对单个值达成一致。

保证很强，但假设：

1. **Independent faults (独立故障)。** Byzantine 节点不协调。
2. **Honest nodes are truly honest (诚实节点真正诚实)。** 诚实输出的正确性不是问题；协议只解决分歧对齐。
3. **The question has a ground-truth answer (问题有 ground-truth 答案)。** 对错误事实的共识仍然是共识。

LLM agent 违反了所有三条。两个运行相同基础模型的 agent 共享故障。一个"诚实"的 LLM 仍然会产生幻觉。而且在模糊问题上，"真相"是 agent 决定的——没有外部 oracle (神谕)。

### The three LLM-specific attacks (三种 LLM 特有的攻击)

**Byzantine lie (拜占庭谎言)。** 一个 agent 故意输出错误答案。如果 `f < n/3`，经典 BFT 能处理这个。

**Sycophantic conformity (谄媚性从众)。** 一个 agent 在投票前读取其他 agent 的答案，并与最后发言的人对齐。不是恶意的，但与最响亮的声音相关。经典 BFT 无法防止这个，因为该 agent 通过了每个签名检查。

**Correlated-error monoculture (相关错误单一文化)。** 三个 agent 共享一个基础模型。它们幻觉出相同的错误答案。多数是错误的。经典 BFT 没有帮助，因为三个都"诚实地"同意。

### The 2025-2026 responses (2025-2026 年的回应)

**CP-WBFT** (arXiv:2511.10400) — Confidence-Probed Weighted BFT (置信度探测加权 BFT)。每个投票者为其答案附加一个 confidence probe (置信度探测)（自报告概率，或单独 calibration model (校准模型) 的预测）。选票权重随置信度缩放。在 complete graphs (完全图) 上报告 BFT 改进 +85.71%。缓解：sycophantic conformity（从众 agent 对其自愿立场的置信度往往较低）。

**DecentLLMs** (arXiv:2507.14928) — 无领导者。Worker agent 并行 propose (提案)，evaluator agent 为提案评分，最终答案是 scored positions (已评分立场) 的 geometric median (几何中位数)。当 `f < n/2` 时具有鲁棒性。缓解：Byzantine lie 和 correlated errors（geometric median 对 outliers (离群值) 鲁棒，并拉向密集簇，而不是模型偏置的平均值）。

**WBFT** (arXiv:2505.05103) — Weighted BFT with Hierarchical Structure Clustering (带层次结构聚类的加权 BFT)。选票权重由 response quality (响应质量) 加上从历史中学习的 trust score (信任分数) 分配。将 agent 聚类为 Core 和 Edge；Core agent 必须先达成共识，Edge agent 跟随。缓解：scalability (可扩展性)（Core 共识小而快）和部分缓解 monoculture（Core 可以选择多样性）。

### Empirical: "Can AI Agents Agree?" (arXiv:2603.01213)

该论文测量了 scalar agreement (标量共识)（LLM agent 在单个数值上达成一致）在多个 frontier models (前沿模型) 上的表现。发现令人不安：

- 即使没有对手，LLM agent 在许多基准测试上对 scalar questions 的分歧率也超过 30%。
- 一个采用 deceptive persona (欺骗性人格) 的单个 agent 可以将 Mixture-of-Agents 共识从诚实基线拉偏 40 多个百分点。
- 分歧率与 model diversity (模型多样性) 相关——异构 ensemble (集成) 比同构 ensemble 分歧更多（好：uncorrelated errors (不相关错误)），但漂移也更慢（坏：更长的时间达成一致）。

要点：BFT 给你对齐输出的机制，但它不告诉你对齐的输出是否正确。需要结合 verification（第 16 阶段 · 08 角色专业化）、diversity（第 16 阶段 · 15 辩论变体）和 evaluator agent（第 16 阶段 · 24 基准测试）。

### The core protocol, stripped down (精简的核心协议)

LLM agent 的最小化 BFT 轮次：

```
1. 任务到达；每个 agent i 产生答案 a_i
2. 每个 agent 附加 confidence probe c_i 于 [0, 1]
3. aggregator (聚合器) 从所有 n 个 agent 收集 (a_i, c_i)
4. aggregator 按 semantic cluster (语义簇) 分组（等价答案）
5. aggregator 为每个簇 C 计算权重：
     w(C) = sum_{i in C} c_i
6. winner = 权重最大的簇，如果最大值 > threshold * sum(c_i)
   否则：重试或升级
7. 少数簇记录 provenance (溯源) 用于事后审计
```

Semantic clustering (语义聚类) 步骤是 LLM 特有的转折。两个答案"the study reports 4.2%"和"4.2% improvement"是同一个簇。朴素的 string-equality (字符串相等) 检查会错过这个。在生产环境中，使用便宜的 embedding model (嵌入模型) 或 explicit canonicalization (显式规范化)。

### Threshold tuning (阈值调优)

`threshold` 参数决定何时接受、何时重试。太低：你接受弱多数。太高：你永远不接受任何东西。经验范围：`n=5-7` 个 agent 时为 0.5-0.67，`n` 较小时更高。低于阈值时，升级给人类或不同的 agent ensemble。

### Where consensus does not help (共识无法帮助的地方)

- **Ambiguous questions (模糊问题)。** 如果问题没有 ground truth，共识是一种观点。就这么称呼它。
- **Compound questions (复合问题)。** "写代码并解释它"——两个答案。分别对每一部分投票。
- **Adversarial multi-round (对抗性多轮)。** 如果 agent 可以观察前几轮并模仿（Du 2023 debate），它们会开始相互同意，无论真相如何。限制轮次（通常 2-3 轮）。

## Build It (动手实现)

`code/main.py` 实现：

- `AgentVoter` —— 带有 (answer, confidence) 的 scripted policy (脚本策略)。
- `MajorityVote` —— 经典 plurality (相对多数)。
- `CPWBFT` —— 带 semantic clustering 的 confidence-weighted voting (置信度加权投票)。
- `DecentLLMs` —— 对 scored proposals 的 geometric-median aggregation (几何中位数聚合)。
- `Scenario` —— 在三种攻击模式下运行每个 aggregator。

实现的攻击模式：

1. `byzantine`：一个 agent 以高置信度撒谎。
2. `sycophancy`：一个 agent 复制它看到的第一个答案，并匹配置信度。
3. `monoculture`：三个 agent 共享一个错误答案（correlated error），置信度中等。

运行：

```
python3 code/main.py
```

预期输出：一个 (attack, aggregator) -> final answer 的表格，正确决策被高亮显示。Plurality 在 monoculture 情况下失败。CPWBFT 的置信度加权缓解了 sycophancy。当 monoculture 少于一半人口时，DecentLLMs 的 geometric-median 拉向诚实簇。

## Use It (使用它)

`outputs/skill-consensus-designer.md` 为一个 multi-agent ensemble 设计共识协议：clustering method (聚类方法)、weighting (加权)、threshold (阈值)，以及 sub-threshold rounds 的 escalation policy (升级策略)。

## Ship It (交付上线)

在交付任何共识机制之前：

- **用至少上述三种模式进行攻击测试。** 你的协议应该可预测地失败，而不是静默失败。
- **记录每个 minority cluster (少数簇) 及其 provenance。** Minority clusters 是你对 correlated errors 的早期预警系统。
- **强制执行 bounded rounds (有界轮次)。** 不要"一直辩论直到达成一致"——那奖励 sycophancy。
- **将 agreement (一致) 与 correctness (正确性) 分离。** 共识输出进入 verifier；verifier 独立于 ensemble。
- **监控 agreement rate (一致率)。** 急剧上升意味着 conformity bias (从众偏置)；急剧下降意味着 model drift (模型漂移)。

## Exercises (练习)

1. 运行 `code/main.py`。确认 plurality 在 monoculture 攻击下失败，但当 monoculture 置信度低于 0.7 时 CPWBFT 部分缓解。
2. 添加第四种攻击模式：**silent abstention (静默弃权)** —— 一个 agent 拒绝回答（"I don't know"）。每个 aggregator 应如何处理 abstentions (弃权)？实现你的选择。
3. 将 semantic clustering 从 string canonicalization 换成 embedding-similarity (使用任何开源 embedding model)。Sycophancy 攻击会发生什么？
4. 阅读 CP-WBFT (arXiv:2511.10400)。实现 confidence-probe calibration step (置信度探测校准步骤)（一个单独的 calibration model 检查每个 agent 的自报告置信度）。测量 monoculture 场景上的准确率提升。
5. 阅读 "Can AI Agents Agree?" (arXiv:2603.01213)。复现一个简化的 scalar-agreement 实验：三个 agent，一个 scalar question，deceptive-persona prompt。CPWBFT 或 DecentLLMs 能捕获它吗？

## Key Terms (关键术语)

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| BFT | "Byzantine fault tolerance" | Castro-Liskov 1999 协议，在 `f < n/3` 任意故障下达成共识。 |
| Byzantine (拜占庭) | "Any bad behavior" | 可以撒谎、丢弃消息、静默失败的节点——任何行为，除了安全崩溃。 |
| Confidence probe (置信度探测) | "How sure are you?" | 附加到选票的自报告或校准器预测概率。 |
| Semantic clustering (语义聚类) | "Same answer, different words" | 在计票前将等价答案分组。 |
| Geometric median (几何中位数) | "Robust center" | 最小化到样本点距离之和的点。对 outliers 鲁棒，不像 mean (均值)。 |
| Monoculture (单一文化) | "Same model, same failures" | Agent 共享训练数据或基础模型时的 correlated errors。 |
| Sycophantic conformity (谄媚性从众) | "Agreeing with the loud voice" | Agent 的选票偏向第一个/最大声发言的人。 |
| Core/Edge (核心/边缘) | "Hierarchical BFT" | WBFT 拆分：小的 Core 先达成共识，Edge nodes 跟随。限制延迟。 |

## Further Reading (延伸阅读)

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — 基础
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — 按置信度加权投票
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — 几何中位数聚合
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — Core/Edge 拆分以限制延迟
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — 标量共识脆弱性与欺骗性人格攻击
