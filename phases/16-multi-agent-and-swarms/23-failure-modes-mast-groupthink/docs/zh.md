# 故障模式 —— MAST、群体思维、单一文化、级联错误

> 2026 年的参考分类体系是 **MAST**（Cemri 等人，NeurIPS 2025，arXiv:2503.13657），该研究从 7 个最先进的开源多智能体系统（MAS）中提取了 1642 条执行轨迹，显示出 **41–86.7% 的故障率**。三个根本类别为：**规范问题**（41.77%）—— 角色模糊、任务定义不清；**协调失败**（36.94%）—— 通信中断、状态失步；**验证缺口**（21.30%）—— 缺少验证、缺乏质量检查。**群体思维（Groupthink）** 家族（arXiv:2508.05687）进一步补充了：单一文化崩溃（monoculture collapse，相同基模型导致相关故障）、从众偏见（conformity bias，智能体相互强化错误）、心智理论（Theory of Mind, ToM）缺陷、混合动机动态（mixed-motive dynamics）、级联可靠性失败（cascading reliability failures）。级联示例：重试风暴（retry storm），支付失败触发订单重试，订单重试触发库存重试，库存服务在数秒内被压垮（负载达到 10 倍 —— 需要熔断器）。记忆投毒（memory poisoning）：一个智能体的幻觉进入共享记忆，下游智能体将其当作事实；准确率逐渐衰减，导致根因诊断非常困难。**STRATUS**（NeurIPS 2025）报告，通过部署专门的检测 / 诊断 / 验证智能体，缓解成功率提升了 1.5 倍。本课将故障模式视为一等工程目标。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** Phase 16 · 13（共享记忆），Phase 16 · 14（共识与 BFT），Phase 16 · 15（投票与辩论拓扑）
**时间：** ~75 分钟

## 问题

多智能体系统（multi-agent system, MAS）在真实任务上的失败率高达 41-86.7%（Cemri 等人 2025 年对 7 个开源 MAS 的测量结果）。这不是“多加几个智能体”就能调试的。这些失败有结构性原因。MAST 分类法为你提供了类别。本课将每个类别映射到具体的检测、诊断和缓解模式，让这些数字不再看起来是随机的。

2026 年的生产实践是将故障模式视为设计输入。你的架构只有在能针对每个 MAST 类别指出已部署的缓解措施时，才算“足够好”。

## 概念

### MAST 类别

**规范问题（占失败的 41.77%）。** 智能体的任务定义不够严格。示例：

- 角色模糊（role ambiguity）：两个智能体都认为自己是审查者。
- 任务欠规范（task underspecified）：用户说“总结一下这个”，但想要的是特定角度。
- 成功标准隐含（success criteria implicit）：智能体无法判断自己是否成功。

缓解措施：
- 编写显式角色契约（role contract）。每个智能体的提示词说明它做什么，*以及不做什么*。
- 每个任务的验收测试（acceptance test）。在智能体开始之前，定义“完成看起来像 X”。
- 起飞前规范检查（pre-flight spec check）：一个独立的智能体在分派前审查任务定义。

**协调失败（占失败的 36.94%）。** 通信或状态崩溃。

示例：
- 两个智能体在没有同步的情况下更新共享状态。
- 智能体之间的消息丢失（队列故障、超时）。
- 状态漂移（state drift）：智能体 A 认为任务已完成；智能体 B 仍在执行。

缓解措施：
- 带乐观并发控制的版本化共享状态（versioned shared state with optimistic concurrency）。
- 关键消息的显式确认（explicit acknowledgment）（重试直到收到确认）。
- 定期状态同步检查点（periodic state-sync checkpoints）；尽早检测漂移。

**验证缺口（占失败的 21.30%）。** 输出没有独立检查。

示例：
- 一个智能体声称成功；没有人验证。
- 智能体链中的每个智能体都信任前一个的输出。
- 对涌现的组合行为缺少测试覆盖。

缓解措施：
- 独立验证智能体（independent verifier agent）（第 13 课）。只读，独立源访问。
- 显式交接契约（explicit handoff contract）：“A 的输出必须通过检查器 C 的验证，B 才能开始。”
- 结果日志记录（outcome logging），用于事后分析。

### 群体思维家族（arXiv:2508.05687）

当智能体同质化或相互模仿时出现的五种相关故障：

**单一文化崩溃（Monoculture collapse）。** 相同的基模型或训练数据 → 相关错误。当三个智能体共享同一个大语言模型（LLM）时，它们共享其幻觉（hallucinations）。

**从众偏见（Conformity bias）。** 智能体向最响亮或最自信的同伴调整，即使对方是错的。

**ToM 缺陷（Deficient ToM）。** 智能体无法建模彼此的信念；协调崩溃（第 18 课）。

**混合动机动态（Mixed-motive dynamics）。** 具有部分对齐激励的智能体漂向妥协中间态，结果谁都不满意。

**级联可靠性失败（Cascading reliability failures）。** 一个组件的错误模式触发依赖组件的错误模式。

### 级联示例 —— 重试风暴（retry storm）

一个经典的 2026 年事故模式：

```
支付服务 10% 的请求失败
   ↓
订单智能体重试支付（指数退避但过于简单）
   ↓
每次重试都是新的订单-库存检查
   ↓
库存服务看到 2 倍正常负载
   ↓
库存服务开始超时
   ↓
每个订单都重试库存检查
   ↓
库存服务看到 10 倍正常负载
   ↓
集群宕机
```

修复方案是经典的：**熔断器（circuit breaker）**。当下游错误率超过阈值时，用缓存或默认结果短路。加上每个请求的重试预算上限（capped retry budgets）。

熔断器是少数几种可以直接从分布式系统借用、无需修改的多智能体故障缓解措施之一。

### 记忆投毒（revisited）

来自第 13 课：一个智能体的幻觉成为共享记忆的事实；下游智能体基于被污染的事实进行推理。用 MAST 的术语来说，这是共享记忆层的一个验证缺口。

逐渐衰减的准确率是症状。你不会收到崩溃；你得到的是缓慢漂移，很难根因诊断。

缓解措施：仅追加日志（append-only log）、来源追溯（provenance）、不可写的验证器。第 13 课已涵盖。

### STRATUS —— 用于故障检测的专门智能体

STRATUS（NeurIPS 2025）报告，当你部署以下组件时，缓解成功率提升 1.5 倍：

- **检测智能体（Detection agent）。** 监视症状模式（高分歧率、重试峰值、准确率漂移）。
- **诊断智能体（Diagnosis agent）。** 给定症状，从 MAST 分类法中推断可能的根因。
- **验证智能体（Validation agent）。** 缓解措施应用后，检查症状是否消除。

这是 SRE 风格的事故响应，应用于智能体系统。这三个角色都可以是带有专门提示词的大语言模型智能体。

### 故障模式审计（failure-mode audit）

2026 年的最佳实践是每年（或每个主要版本）进行一次故障模式审计：

1. **轨迹采样（Trace sample）。** 收集约 1000 条真实执行轨迹。
2. **分类（Categorize）。** 对每条轨迹的故障，映射到 MAST + 群体思维类别。
3. **计算按类别的故障率（Compute failure-by-category rate）。** 哪些类别在你的系统中占主导？
4. **缓解措施排序（Rank mitigations）。** 哪个修复能消除最多的故障？
5. **选择 2-3 个缓解措施（Pick 2-3 mitigations）。** 实施；下季度重新审计。

纪律比具体选择更重要。没有审计，故障会混入噪音，永远不会被系统性地解决。

### 当系统静默失败时

最危险的故障类别是静默正确性失败（silent correctness failure）。一个大声失败的系统（崩溃、异常、告警）可以被监控。一个产生看似合理但错误输出的系统无法通过异常日志检测。这就是为什么验证缺口虽然只占故障数的 21.30%，但每个故障的代价却是最高的。

投资于：
- 基于采样的真人审查（sample-based human review）。
- 黄金数据集回归测试（golden-dataset regression tests）。
- 对重要输出的跨智能体交叉检查（cross-agent cross-checking）。

### 快速失败 vs 缓慢失败

有些失败是即时的；有些是缓慢的。即时失败（超时、模式不匹配、认证错误）检测成本低。缓慢失败（记忆投毒、单一文化漂移、角色模糊）检测和预防成本高。

2026 年的工程动作：为缓慢失败代理指标（slow-failure proxies）插桩，以便在漂移变成可见错误之前捕捉到它。一致率（agreement rate）、重试率（retry rate）、输出版本之间的输出长度分布、编辑距离（edit-distance）都是有用的代理指标。

## Build It

`code/main.py` 实现了：

- `FailureTaxonomy` —— 将模拟事故分类到 MAST + 群体思维类别。
- `CircuitBreaker` —— 经典模式；当错误率超过阈值时打开。
- `RetryStormSimulator` —— 展示级联失败；可开关熔断器。
- `DetectionAgent` —— 脚本化的 STRATUS 风格症状匹配器。

运行：

```
python3 code/main.py
```

预期输出：
- 无熔断器的重试风暴：库存错误爆发（模拟）。
- 有熔断器：在阈值处封顶；提供降级模式响应。
- 检测智能体标记模式并命名 MAST 类别。

## Use It

`outputs/skill-mast-auditor.md` 对多智能体系统运行 MAST 风格的故障模式审计。轨迹 → 分类 → 缓解排序。

## Ship It

生产中的故障模式纪律：

- **每季度 MAST 审计。** 不是每年。类别会随着系统增长而转移。
- **到处部署熔断器（Circuit breakers）。** 每个对任何依赖服务的外向调用。默认打开阈值设为 5-10% 错误率。
- **黄金数据集（Golden datasets）。** 小型、高质量、人工审计。每周对它们进行回归测试。
- **STRATUS 三人组。** 检测 + 诊断 + 验证智能体监控生产环境。先从检测智能体开始；当症状嘈杂时添加诊断。
- **故障预算（Failure budget）。** 按类别明确设定失败率的 SLO。超出预算触发停止发布的对话。

## Exercises

1. 运行 `code/main.py`。确认熔断器封顶了重试风暴。改变失败阈值并观察权衡。
2. 实现一个**缓慢失败代理指标**：3 个并行智能体之间的一致率（agreement rate）。当它急剧下降时触发告警。通过逐渐相关化智能体输出来模拟单一文化漂移。
3. 阅读 Cemri 等人（arXiv:2503.13657）。选择他们 7 个 MAS 系统中的一个，映射其前 3 个故障类别。这些与 MAST 的预测相比如何？
4. 阅读群体思维论文（arXiv:2508.05687）。识别五种模式中哪一种在生产环境中最难检测。提出一个代理指标。
5. 为你熟悉的特定多智能体系统设计一个 STRATUS 风格的检测-诊断-验证三人组。检测监视哪些症状？诊断推荐哪些缓解措施？验证如何确认它们有效？

## Key Terms

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| MAST | "2026 年的分类法" | Cemri 2025；3 个根类别 + 14 种子类型故障。 |
| Specification Problem | "角色模糊" | 任务或角色定义不足；智能体不知道做什么。 |
| Coordination Failure | "状态漂移" | 智能体之间的通信或同步崩溃。 |
| Verification Gap | "没人检查" | 输出在没有独立验证的情况下被接受。 |
| Groupthink family | "同质化失败" | 单一文化、从众、ToM 缺陷、混合动机、级联。 |
| Monoculture collapse | "相同模型，相同幻觉" | 共享基模型或训练数据导致的相关错误。 |
| Retry storm | "级联错误放大" | 一个失败触发重试，重试放大下游负载。 |
| Circuit breaker | "错误率上快速失败" | 错误率超过阈值时打开；用默认值短路。 |
| STRATUS | "事故响应三人组" | 检测 + 诊断 + 验证智能体。缓解成功率提升 1.5 倍。 |
| Memory poisoning | "幻觉传播" | 共享记忆事实被污染；下游智能体基于毒化事实推理。 |

## Further Reading

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法，NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — 单一文化、从众及五家族分类法
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 会议论文（检测 + 诊断 + 验证）
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — 熔断器的经典参考
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 生产故障模式笔记
