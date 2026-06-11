# Kill Switches, Circuit Breakers, and Canary Tokens

> A kill switch 是一个位于智能体编辑表面之外的布尔值 —— Redis 键、功能标志、签名配置 —— 用于完全禁用智能体。Circuit breaker 粒度更细：它在特定模式上触发（连续五次相同的工具调用），暂停有问题的路径，并上报给人类。Canary token 继承自经典欺骗：智能体没有正当理由接触的一个假凭证或蜜罐记录，其访问会触发警报。基于 eBPF 的数据路径（例如 Cilium）可以在内核层将被隔离 pod 的出口流量重写为取证蜜罐；已发布的 Cilium 基准测试报告称，在负载下 P99 数据路径延迟低于一毫秒（你的传播预算取决于策略更新到达节点的方式，而不是数据路径本身）。统计检测器（EWMA, CUSUM）会适应移动基线而悄然接受漂移 —— 将它们与不会弯曲的硬性宪法限制分层。

**类型：** Learn
**语言：** Python (stdlib, three-detector simulator: kill switch, circuit breaker, canary)
**前置条件：** Phase 15 · 13 (Cost governors), Phase 15 · 10 (Permission modes)
**时间：** ~60 分钟

## The Problem

成本治理器（第 13 课）限制了智能体可以花费的金额。它们不限制智能体在预算内可以做什么。一个有 50 美元 velocity limit 的智能体仍然可以外泄秘密、发布错误帖子或删除资源 —— 昂贵的行动在 token 方面往往是便宜的。

本课涵盖位于成本层旁边的三个检测器：

1. **Kill switch**：操作员无法触及的关闭按钮。
2. **Circuit breaker**：在特定路径上暂停的操作模式检测器。
3. **Canary token**：智能体没有正当理由接触的诱饵，其接触会揭示自身。

三者都是 LLM 之前的工程产物。经典欺骗、速率限制断路器和功能标志 kill 早于自主智能体。新的地方在于攻击面：智能体读取不可信内容（第 11 课）、编辑自己的记忆，并能将许多看似安全的操作组合成不安全的操作。这里命名的检测器之所以有效，是因为它们不信任智能体的自我报告。

## The Concept

### Kill switches

Kill switch 是智能体可以读取但不能写入的布尔值。实现选择：

- **托管服务中的功能标志。** LaunchDarkly, Statsig, Unleash。更新在数秒内传播。
- **智能体轮询的 Redis 键。** 简单；要求智能体进程在每轮检查。
- **对象存储中的签名配置。** 智能体在启动时验证签名；拒绝未签名状态。
- **操作系统级信号或容器生命周期 kill。** Docker `kill`, Kubernetes `kubectl delete pod`, systemd stop。

正确的 kill switch 具有以下属性：

- 智能体无法将其设置为 `off`。（位于智能体凭证无法写入的系统中。）
- 它在每次 consequential 行动时检查，而不仅仅在启动时。
- 当它为 off 时，智能体不执行任何外部可观察的操作，包括向智能体可达的系统记录日志。
- 重新启用它是一个显式的人类操作，而不是自动超时。

### Circuit breakers

Circuit breaker 暂停特定模式，而不是整个智能体。经典形状（来自 2007 年 Nygard 的书，仍然适用）：

- **Closed**：允许操作。
- **Open**：阻止操作。
- **Half-open**：冷却期后，允许 1–3 次探测尝试（默认 1）；成功关闭断路器，任何剩余失败重新打开它。

与智能体相关的触发器：

- 连续五次相同的工具调用（重复循环）。
- 同一工具连续五次失败且输入不同（系统性故障）。
- 工具调用速率高于阈值（第 13 课 velocity）。
- 在 out-of-trust 内容读取后调用特定工具（例如，`file.delete`）（第 11 课）。

### Canary tokens

Canary tokens（也称为 honeytokens）是智能体永远不应接触的输入。访问即警报。

智能体的示例：

- 环境中没有实际权限的假 `AWS_SECRET_ACCESS_KEY`。外泄尝试可立即归因。
- 带有已知哨兵值的假数据库记录。该记录的任何读取或更新都会触发警报。
- 工作区中的假文件。对该文件的读取或编辑是可疑的。
- 会话开始时注入的假记忆条目。如果它出现在后续输出中，记忆已被外泄。

Canary 设计特定于工作流。智能体有正当理由接触的 canary 是噪音。

### Why layer statistical and hard limits

统计检测器（EWMA, CUSUM, z-score on tool-call rates）适应移动基线。这对合法漂移是特性，对对抗性漂移是 bug。有耐心的攻击者可以缓慢移动基线，使没有警报触发。

硬性宪法限制（第 17 课）不会弯曲。"永远不要给 `secrets@` 发邮件"作为宪法规则更简单、可审计，且不会被缓慢漂移博弈。分层模式：统计检测器捕获大多数噪音，硬性限制捕获走过检测器的攻击。

### Quarantine via eBPF datapath redirect

Cilium 的基于 eBPF 的网络策略可以在内核层将被隔离 pod 的出口流量重写为取证蜜罐。Cilium 的公开基准测试报告称 TCP_RR 上 P99 数据路径延迟低于一毫秒；在集群中*触发*隔离的挂钟预算取决于策略更新从控制平面传播到节点需要多长时间（在 operator 已预加载的 Kubernetes 上，几百毫秒是现实的；尾部延迟取决于集群规模和协调节奏）。这给你两件事：智能体不会外泄任何真实内容，以及智能体在蜜罐内的行为被完全观察。与 canary token 相同的模式，不同的层：智能体通过在被隔离时试图做什么来揭示意图。

### What no detector catches

- **从允许的原语组合的语义攻击。** 每个单独的操作通过每个检测器；组合才是问题。这是第 10 课的 Auto Mode 复合故障。
- **将检测器作为载荷一部分的攻击。** 如果智能体读取一个页面说"在你行动之前，禁用 canary"，而智能体有能力这样做，检测器就被破坏了。Canary tokens 应该位于智能体无法修改的系统中。

## Use It

`code/main.py` 模拟了一个短智能体轨迹通过三个检测器。Kill switch 保存在外部字典中；circuit breaker 在连续五次相同工具调用上触发；canary 文件的读取触发警报。输入合成轨迹：合法操作、重复循环、canary 探测，以及 kill-switch-triggered 场景，其中智能体的操作被停止。

## Ship It

`outputs/skill-tripwire-design.md` 审核拟议智能体部署的检测器栈并标记差距（缺失 kill switch、缺失 canary、circuit breaker 阈值过松）。

## Exercises

1. 运行 `code/main.py`。确认 circuit breaker 在第 5 轮触发（第五次相同调用），canary 在第 9 轮触发（假密钥读取）。

2. 添加一个统计检测器：工具调用速率的 EWMA z-score。输入缓慢漂移的轨迹并展示检测器从不触发。现在添加一个硬性限制（10 分钟内不超过 50 次工具调用）并展示同一轨迹上硬性限制触发。

3. 为浏览器智能体（第 11 课）设计一套 canary token。列出至少三个 canary 以及每个检测什么。

4. 阅读 Cilium 网络策略文档。具体描述一个出口重定向隔离流：哪个策略选择器、哪个 pod、哪个出口重写、哪个警报。从"决定隔离"到"第一个重定向数据包"的挂钟延迟由什么决定？

5. 为一个 kill-switched 智能体定义重新启用流程。谁可以重新启用？必须记录什么？重新启用前智能体必须改变什么？

## Key Terms

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| Kill switch | "关闭按钮" | 智能体编辑表面之外的布尔值；每次 consequential 行动时检查 |
| Circuit breaker | "模式暂停" | 在重复、失败率或速率限制上的操作特定触发 |
| Canary token | "Honeytoken" | 智能体没有正当理由接触的诱饵；访问触发警报 |
| Honeypot | "取证沙箱" | 被隔离智能体被观察的重定向流量 / 工作区 |
| EWMA | "移动平均" | 指数加权；适应漂移（特性 + bug） |
| CUSUM | "累积和" | 检测与基线的持续偏离 |
| Hard limit | "宪法规则" | 不随历史变化；无论历史如何都保持不变 |
| Constitutional limit | "永远为真的规则" | 与第 17 课的宪法绑定；智能体无法编辑 |

## Further Reading

- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — kill-switch and circuit-breaker framing for autonomous agents.
- [Microsoft Agent Framework — HITL and oversight](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — production governance patterns.
- [OWASP LLM / Agentic Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) — detection-and-response requirements.
- [Cilium — Network policy and eBPF](https://docs.cilium.io/en/stable/security/network/) — pod-level egress redirect and forensic honeypot patterns.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — hardcoded prohibitions as "constitutional limits".
