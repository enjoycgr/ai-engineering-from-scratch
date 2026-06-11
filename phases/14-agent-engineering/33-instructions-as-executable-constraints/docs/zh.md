# Agent Instructions as Executable Constraints

> 以散文形式书写的 instructions（指令）是愿望。以 constraints（约束）形式书写的 instructions 是测试。workbench（工作台）将每条规则转化为 agent 可以在 runtime（运行时）检查、reviewer（审查者）可以在事后验证的东西。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 32 (Minimal Workbench)
**Time:** ~50 分钟

## Learning Objectives

- 将 routing prose（路由散文）与 operational rules（操作规则）分开。
- 将 startup rules（启动规则）、forbidden actions（禁止行为）、definition of done（完成定义）、uncertainty handling（不确定性处理）和 approval boundaries（审批边界）表达为 machine-checkable constraints（机器可检查约束）。
- 实现一个 rule checker（规则检查器），根据 rule set（规则集）为一次运行评分。
- 使 rule set 对 diff 友好，以便 review 能看到什么发生了变化。

## The Problem

典型的 `AGENTS.md` 读起来像 onboarding documentation（入职文档）。它告诉 agent 要 "be careful"、"test thoroughly"、"ask if unsure"。三天后，agent 交付了一个没有测试的改动，写入了 forbidden directory（禁止目录），并且从未询问，因为它从未知道边界在哪里。

Instructions 在 operational（可操作）时强大，在 aspirational（空泛）时脆弱。修复方法是编写 workbench 可以解释、reviewer 可以评分的规则。

## The Concept

Rules 属于 `docs/agent-rules.md`，远离简短的根 router。每条规则都有 name、category 和 check。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[Reviewer]
```

### 覆盖大多数规则的五个类别

| Category | 规则回答的问题 | 示例 |
|----------|-------------|------|
| Startup | 工作开始前必须为真的是什么？ | "state file 存在且新鲜" |
| Forbidden | 绝对不能发生什么？ | "不要编辑 `scripts/release.sh`" |
| Definition of done | 什么证明了任务完成？ | "pytest 退出码为 0 且 acceptance line 通过" |
| Uncertainty | 当不确定时 agent 做什么？ | "打开一个 question note（问题注释）而不是猜测" |
| Approval | 什么需要人类审批？ | "任何新依赖，任何生产环境写入" |

一条不适合这五个类别之一的规则通常想变成两条规则。强制拆分。

### Rules 是 machine-readable（机器可读的）

每条规则有一个 slug、一个 category、一行 description 和一个指向 `rule_checker.py` 中 function 的 `check` 字段。添加一条规则意味着添加一个 check；checker 随 workbench 一起成长。

### Rules 是 diff-friendly（对 diff 友好的）

Rules 在单个 markdown 文件中每条规则一个 heading 存在。重命名在 diff 中可见。新规则位于其 category 的顶部。Stale rules 被删除而非注释掉，因为 workbench 是 source of truth，而不是上季度团队感受的 chat log。

### Rules versus framework guardrails

Framework guardrails（OpenAI Agents SDK guardrails、LangGraph interrupts）在 runtime 层面强制执行规则。本课的 rule set 是这些 guardrails 实现的人类可读、可审查的合约。两者都需要：runtime 在 turn 期间捕获违规；rule set 证明 runtime 在做正确的事。

### Progressive disclosure（渐进式披露）：一张地图，而非百科全书

`AGENTS.md` 不断增长的原因是每次 incident 都添加一条规则，但没有 incident 移除规则。一年后，文件长达两千行，agent 读了第一屏，耗尽 attention budget（注意力预算），然后只根据被告知的 fraction（一小部分）行动。一份巨大的 instruction file 失败的原因与四十页的 onboarding doc 失败的原因相同：读者 skim（浏览）一次，然后从不返回重要的部分。

修复方法不是更短的文件。而是分层的文件。根 router 保持小到每次 session 都能阅读，并且只持有 pointers（指针）。深度存在于 topic files 中，agent 仅在任务触及它们时才加载。给 agent 一张地图，而非整本百科全书，让它走到需要的页面。

```
AGENTS.md                  # router，< 50 行：这个仓库是什么、去哪里看、5 条 hard rules
docs/
  agent-rules.md           # 完整 rule set（本课）
  architecture.md          # 当任务触及模块边界时加载
  testing.md               # 当任务写入或运行测试时加载
  deploy.md                # 仅在发布工作中加载，由 approval rule 把关
feature_list.json          # backlog（Phase 14 · 36）
```

| Tier | 存在于 | 何时读取 | 大小预算 |
|------|--------|---------|---------|
| Router | `AGENTS.md` | 每次 session，永远 | 约 50 行以内 |
| Rules | `docs/agent-rules.md` | 每次 session，启动时 | 每类别一屏 |
| Topic docs | `docs/<topic>.md` | 仅当任务触及该 topic 时 | 按需深入 |

两个测试保持 layering（分层）的诚实性。Reachability test（可达性测试）：agent 应该能在从 router 出发最多两跳内到达任何规则，因此 router 必须按 path 链接每个 topic doc，而非用 prose 描述它。Freshness test（新鲜度测试）：router 足够短，以至于 reviewer 在每个 PR 上都会重读它，这是阻止它悄无声息地长回它所替代的百科全书的唯一事物。一个不再解析的 pointer 比缺失的规则更糟，因此 router 中的 broken link 本身就是 startup-check violation。

## Build It

`code/main.py` 交付：

- `agent-rules.md` parser，将 rules 加载到 dataclass 中。
- `rule_checker.py` style checker functions，每个 `check` 引用一个。
- 一个 demo agent run，违反了两条 rules，以及一个捕获它们的 check pass。

运行方式：

```
python3 code/main.py
```

输出：parsed rule set、run trace、每条规则的 pass/fail，以及保存在脚本旁边的 `rule_report.json`。

## Production patterns in the wild

三个模式将能持续一个季度的 rule set 与一周内就会腐烂的 rule set 区分开。

**Severity tagging at write time（编写时的严重度标记）。** 每条规则携带 `severity`：`block`、`warn` 或 `info`。Checker 报告全部三个；runtime 仅在 `block` 时拒绝。大多数团队早期会夸大 severity，然后在 deadline pressure 下悄悄削弱它；编写时标记 severity 会 upfront（预先）强制校准。与 verification gate（Phase 14 · 38）配对，后者将任何对 `block` rule 的 override 签名到 `overrides.jsonl` audit log 中。

**Rule expiry as a forcing function（规则过期作为强制函数）。** 每条规则携带一个 `expires_at` 日期（默认自编写起 90 天）。Checker 在一条未过期规则连续 60 天零违规时发出 warning；下一个季度 review 要么证明保留它，要么削弱为 `info`，要么删除它。Cloudflare 的生产 AI Code Review 数据（2026 年 4 月，30 天内 5,169 个仓库的 131,246 次 review runs）显示，带有明确 expiry 的 rule set 保持在每个仓库 30 条规则以下；没有 expiry 的增长到 80+ 条且大多数从未触发。

**Markdown-as-source, JSON-as-cache（Markdown 作为源，JSON 作为缓存）。** `agent-rules.md` 是 authored file（编写文件）；`agent-rules.lock.json` 是 checker 在 hot path（热路径）中读取的缓存。Lock 由 pre-commit hook 重新生成。Markdown diffs 是可 review 的；JSON parsing 保持在每次 turn 之外。与 `package.json` / `package-lock.json` 和 `Cargo.toml` / `Cargo.lock` 形状相同。

## Use It

在生产环境中：

- Claude Code、Codex、Cursor 在 session start 时读取 rules，并在拒绝 action 时引用它们。Checker 在 CI 中重新运行以捕获 silent drift（静默漂移）。
- OpenAI Agents SDK guardrails 将相同的 checks 注册为 input 和 output guardrails。Markdown 是 docs surface；SDK 是 runtime surface。
- LangGraph interrupts 在 in-flight node 违反规则时触发。Interrupt handler 读取规则，询问人类，然后恢复。

Rule set 可移植到所有三个平台，因为它只是 markdown 加 function names。

## Ship It

`outputs/skill-rule-set-builder.md` 采访 project owner，将现有 prose instructions 分类到五个类别中，并发出一份 versioned `agent-rules.md` 和一个 checker stub。

## Exercises

1. 如果你的产品真正需要，添加第六个类别。论证为什么它不能归入五个之一。
2. 扩展 checker，使规则可以携带 severity（`block`、`warn`、`info`），并且 report 相应聚合。
3. 将 checker 接入 CI：如果 block-severity rule 在最新 agent run 上失败，则 build 失败。
4. 给每条规则添加 "expiry" 字段。90 天内没有 check fail，该规则进入 review。
5. 找到一份真实的 `AGENTS.md` 并将其重写为五类别 rules。它有多少行是 operational 的？有多少是 aspirational 的？

## Key Terms

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|----------|----------------|
| Operational rule | "一条真正的指令" | workbench 可以在 runtime 检查的 rule |
| Aspirational rule | "Be careful" | 没有 check 的 rule；要么删除，要么升级 |
| Definition of done | "Acceptance" | 证明任务完成的客观的、文件支持的证据 |
| Block severity | "Hard rule" | 违规停止运行；没有 operator 无法静默 |
| Rule expiry | "Stale rule 清理" | N 天内零 fail 的规则进入退役流程 |

## Further Reading

- [OpenAI Agents SDK guardrails](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph interrupts](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower, Agent RuleZ: A Deterministic Policy Engine](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产环境中的 block/warn/info severity
- [Cloudflare, Orchestrating AI Code Review at Scale](https://blog.cloudflare.com/ai-code-review/) — 131k review runs，rule composition 经验
- [microservices.io, GenAI development platform — part 1: guardrails](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — rules 和 CI 之间的 defense in depth
- [Type-Checked Compliance: Deterministic Guardrails (arXiv 2604.01483)](https://arxiv.org/pdf/2604.01483) — Lean 4 作为 rule-as-check 的上界
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — merge-gate 实现：scope、mutation testing、violation budgets
- Phase 14 · 32 — 此 rule set 落入的最小 workbench
- Phase 14 · 38 — 消费 rule report 的 verification gate
- Phase 14 · 39 — 评分 rule compliance 的 reviewer agent
