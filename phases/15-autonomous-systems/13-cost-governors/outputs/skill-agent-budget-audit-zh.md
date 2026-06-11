---
name: agent-budget-audit
description: 在启用无人值守运行之前，审核智能体部署的成本治理器栈并标记缺失的层。
version: 1.0.0
phase: 15
lesson: 13
tags: [cost-governors, denial-of-wallet, budgets, claude-code-sdk, agent-governance]
---

给定一个拟议的智能体部署，根据十二层参考标准审核其成本治理器栈，并标记哪些层缺失、调优不足或调优过度。

产出：

1. **层清单。** 对于十二层参考标准中的每一层（per-request cap, per-task token budget, per-task dollar budget, per-tool cap, iteration cap, per-minute/hour/day/month rolling caps, velocity limit, tiered routing, prompt caching, context windowing, HITL checkpoints, kill switch），说明它是否已配置及其值。
2. **故障模式映射。** 对于每个时间尺度的故障（runaway loop, slow leak, bad release, legitimate surge），命名捕获它的具体层及其速度。
3. **工具特定上限。** 列出智能体可以调用的每个工具。对每个工具，命名每会话上限及其原因。任何没有显式上限的工具都是开放循环。
4. **警报阈值。** 与上限分开：在什么花费速率下人类会收到页面？观察到的电商案例（1,200 美元 → 4,800 美元）是周环比增长问题，而不是月度上限问题。
5. **Kill-switch 路径。** 当上限触发时会发生什么？干净中止、回滚、警报、重新启用流程。确认 kill switch 在智能体外部（智能体不能编辑自己的上限）。

硬性拒绝：
- 任何没有 per-task dollar budget 的自主部署。
- 任何没有 velocity limit 的无人值守长时程运行。
- 新工具（上线 <30 天）没有每工具上限的工具表面。
- 智能体本身可以修改的 kill switch。
- 仅以月度上限为唯一上限（所有其他时间尺度都未受保护）。

拒绝规则：
- 如果用户无法以今天的模型价格为最坏情况运行定价，拒绝并要求一份成本估算。
- 如果拟议预算超过组织对单次失误的可接受损失，拒绝并要求更低的上限。
- 如果用户将 Auto Mode 分类器（第 10 课）视为预算的替代品，拒绝。分类器与成本正交；两层都是必需的。

输出格式：

返回一份成本治理器审核，包含：
- **层表**（层名称、已配置 y/n、值）
- **故障模式覆盖**（4 行：loop / leak / release / surge）
- **每工具上限**（工具、上限、原因）
- **警报阈值**（速率、负责人、渠道）
- **Kill-switch 路径**（触发器、操作、重新启用流程）
- **就绪度**（production / staging / research-only）
