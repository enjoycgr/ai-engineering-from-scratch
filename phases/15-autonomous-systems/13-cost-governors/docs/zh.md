# 行动预算、迭代上限与成本治理器

> 一家中型电商智能体的月度 LLM 成本从 1,200 美元跃升至 4,800 美元，原因是团队启用了"订单追踪"技能。这不是定价 bug。而是一个智能体发现了新循环并持续在其中消费。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）将针对此类问题的防御编码为：每请求 `max_tokens`、每任务 token 和美元预算、每日/月上限、迭代上限、分层模型路由、prompt caching、上下文窗口压缩、昂贵行动上的人类介入 checkpoint、预算违规时的 kill switch。Anthropic 的 Claude Code Agent SDK 以不同名称提供了相同的原语。财务速度限制 —— 例如 10 分钟内超过 50 美元即切断访问 —— 比月度上限更快捕获循环。

**类型：** Learn
**语言：** Python (stdlib, layered cost-governor simulator)
**前置条件：** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**时间：** ~60 分钟

## The Problem

自主智能体每次轮次都要花费真金白银。聊天机器人的糟糕输出只是一个糟糕回复；智能体的糟糕循环则是一份账单。业界对此故障模式的记录术语是"Denial of Wallet" —— 智能体持续推理、持续调用工具、持续计费，而没有任何东西阻止它，因为没有任何东西被设计来阻止。

解决方案不是单个数字。而是在不同时间尺度和粒度上的一系列限制：每请求、每任务、每小时、每天、每月。设计良好的栈能在几分钟内捕获失控循环，在几小时内捕获缓慢泄漏，在一天内捕获糟糕的发布。当智能体是长时程且自主时，同一个栈也能保持预算受控。

这是一堂工程课：数学很简单，纪律才是团队失败的地方。下面列出的所有限制名称要么出自 Microsoft Agent Governance Toolkit，要么出自 Anthropic Claude Code Agent SDK 文档。

## The Concept

### The cost-governor stack

1. **`max_tokens` per request.** 简单。防止任何一次调用产生无限制的补全。
2. **Per-task token budget.** 整个运行过程中不超过 N 个 token。达到上限时硬停止。
3. **Per-task dollar budget.** 与 token 相同，但以货币计。Claude Code 中的 `max_budget_usd`。
4. **Per-tool call cap.** 不超过 N 次 `WebFetch` 调用，N 次 `shell_exec` 调用等。
5. **Iteration cap (`max_turns`).** 智能体循环的总迭代次数；防止无限推理循环。
6. **Per-minute / per-hour / per-day / per-month cap.** 滚动窗口。在不同时间尺度捕获泄漏。
7. **Financial velocity limit.** 例如，"如果 10 分钟内花费超过 50 美元，切断访问。"在月度上限触发之前捕获基于循环的燃烧。
8. **Tiered model routing.** 默认使用较小的模型；仅当分类器判断任务需要时才升级到较大的模型。
9. **Prompt caching.** 系统提示和稳定上下文存储在提供商缓存中；重新发送的 token 成本接近零。
10. **Context windowing.** 压缩 / 摘要以保持活动上下文低于阈值；直接降低 token 成本。
11. **HITL checkpoints on expensive actions.** 在已知昂贵的行动之前（长工具调用、大下载、昂贵的模型升级），需要人类点击确认。
12. **Kill switch on budget breach.** 任何上限触发时会话中止。上限被记录；需要单独的重新启用路径。

### Why the stack, not one cap

单个月度上限只能在钱包耗尽后捕获失控智能体。单个每请求上限在会话层面什么也捕获不到。不同的故障模式需要不同的时间尺度：

- **Runaway loop** (智能体卡在 5 秒重试中)：由 velocity limit 捕获。
- **Slow leak** (智能体每项任务执行约 2 倍预期工作)：由每日上限捕获。
- **Bad release** (新版本使用 5 倍 token)：由每周 / 月度上限捕获。
- **Legitimate surge** (真实需求，不是 bug)：由小时 / 天上限捕获并带有清晰日志。

### Claude Code's budget surface

Claude Code Agent SDK 公开了（公共文档）：

- `max_turns` —— 迭代上限。
- `max_budget_usd` —— 美元上限；违规时中止会话。
- `allowed_tools` / `disallowed_tools` —— 工具白名单和黑名单。
- 工具使用前的 hook 点用于自定义成本核算。

与权限模式阶梯（第 10 课）结合。没有 `max_budget_usd` 的 `autoMode` 会话是无约束的自主。Anthropic 明确将 Auto Mode 框架为需要预算控制；分类器与成本正交。

### EU AI Act, OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 涵盖了 OWASP Agentic Top 10 和 EU AI Act 第 14 条（人类监督）要求。对于欧盟的生产环境，日志记录和上限执行不是可选的。

### The observed $1,200 → $4,800 case

Microsoft 文档中的真实案例：一个电商智能体在添加新工具后月度成本翻了三倍。该工具允许智能体在每个会话中轮询订单状态。没有循环检测。没有每工具上限。没有对周环比增长发出警报。修复方案是每工具上限加上每日增长警报。这是一个模板：每个新工具表面都是新的潜在循环；每个新工具都需要自己的上限和自己的警报。

## Use It

`code/main.py` 模拟了有和没有分层成本治理器栈的智能体运行。模拟的智能体在某些轮次后漂移进入轮询循环；分层栈在 velocity 窗口内捕获它，而单个月度上限要到几天后才触发。

## Ship It

`outputs/skill-agent-budget-audit.md` 审核拟议智能体部署的成本治理器栈并标记缺失的层。

## Exercises

1. 运行 `code/main.py`。确认在轮询循环轨迹上 velocity limit 在 iteration cap 之前触发。现在禁用 velocity limit 并测量智能体在 iteration cap 捕获它之前"花费"了多少。

2. 为浏览器智能体（第 11 课）设计一套每工具上限。哪个工具需要最紧的上限？哪个工具可以无风险地无限制运行？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出工具包命名的每种上限类型。将每种映射到一种故障模式（runaway loop, slow leak, bad release, surge）。

4. 为一个现实任务（例如"对仓库中的 50 个 issue 进行分类"）的夜间无人值守运行定价。将 `max_budget_usd` 设置为你点估计的 2 倍。证明 2 倍的合理性。

5. Claude Code 的 `max_budget_usd` 在会话累计成本上触发。设计一个你将在外部执行的互补 velocity limit。什么触发切断，重新启用是什么样子？

## Key Terms

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| Denial of Wallet | "失控账单" | 智能体循环在无上限阻止的情况下产生花费 |
| max_tokens | "每请求上限" | 单次补全大小的上限 |
| max_turns | "迭代上限" | 会话中智能体循环迭代次数的上限 |
| max_budget_usd | "美元 kill switch" | 会话成本上限；违规时中止 |
| Velocity limit | "速率上限" | 短窗口内花费的上限（例如 10 分钟 50 美元） |
| Tiered routing | "小模型优先" | 便宜模型默认；仅当分类器需要时才升级 |
| Prompt caching | "缓存系统提示" | 提供商端缓存将重新发送的 token 成本降至接近零 |
| HITL checkpoint | "人类批准门" | 昂贵行动前需要人类点击确认 |

## Further Reading

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`, `max_budget_usd`, tool allowlists.
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor checkpoints.
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — provider-side cost controls.
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — caching mechanics.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — cost profile for long-horizon agents.
