# Checkpoints 与 Rollback（检查点与回滚）

> 每一个 graph-state transition（图状态转换）都会被持久化。当 worker 崩溃时，其租约过期，另一个 worker 会从最新的 checkpoint（检查点）继续执行。Cloudflare Durable Objects 可以跨小时甚至数周保持状态。Propose-then-commit（第 15 课）为每个动作定义了 rollback（回滚）计划。Post-action verification（执行后验证）闭合整个循环。EU AI Act（欧盟人工智能法案）第 14 条要求高风险系统必须具备有效的人类监督——在实践中，这意味着 checkpoint 必须可被查询、rollback 必须经过演练、审计轨迹必须能在部署后存活。最尖锐的失效模式是：如果没有 idempotency key（幂等键）和 precondition check（前置条件检查），在瞬态故障后的重试可能导致已经获批的动作被重复执行两次。Post-action verification 正是用来捕获这种问题的。

**类型：** Learn
**语言：** Python（stdlib，checkpoint 与 rollback 状态机）
**前置条件：** Phase 15 · 12（Durable execution，持久化执行），Phase 15 · 15（Propose-then-commit）
**时间：** ~60 分钟

## 问题

Durable execution（第 12 课）让崩溃后的 agent 可恢复。Propose-then-commit（第 15 课）让已批准的动作可审计。本课将两者结合：当一个已批准的动作部分执行、崩溃并恢复时，会发生什么？rollback 何时运行，针对什么状态？

真实系统的实现方式各有不同：

- **LangGraph** 将每一个 graph-state transition checkpoint 到 PostgreSQL。worker 崩溃后，租约释放，另一个 worker 从最新的 checkpoint 恢复。`interrupt()` 本身也会被持久化。
- **Cloudflare Durable Objects** 可以按 key 保持状态数小时甚至数周。将计算与已批准动作的存储放在同一位置。
- **Microsoft Agent Framework** 在工作流 API 中暴露了 `Checkpoint` 原语；重放加幂等性覆盖了重试。

在所有情况下，真正有效的组合是：idempotency key（防止重复执行）+ precondition check（状态仍与批准时一致）+ post-action verify（副作用确实发生了）+ rollback on verify-fail（验证失败时回滚）。

## 概念

### 每一个 transition 都被持久化

graph-state transition 是工作流从一个命名状态移动到另一个命名状态的任何步骤。简单实现只在特定 commit point 持久化；生产级实现持久化每一个 transition。成本（几次额外写入）相对于可靠性收益（重放可以落在任何位置，租约恢复更精确）来说很小。

### 租约恢复（Lease recovery）

当 worker 崩溃时，工作流不会丢失；租约（一个短期的声明，表示该 worker 正在执行该 run）只是过期了。另一个 worker 会拾取最新的 checkpoint 并恢复。租约机制让生产系统能够在滚动部署中不丢失正在执行的任务。

### 幂等性加前置条件

仅靠幂等性是不够的。考虑以下场景：一个工作流被批准在"余额 > $1000 时将 $100 从 A 转账到 B"。工作流已提交，执行中途崩溃，然后恢复。如果只检查了 idempotency key，执行恢复后转账运行一次（正确）。但考虑在崩溃和恢复之间，A 的余额通过另一个工作流降到了 $500。Idempotency check 仍然通过；precondition 不通过。没有 precondition check，我们就发出了一个透支转账。

每一个有后果的动作都需要两者：

- **Idempotency key**：防止重复执行。
- **Precondition check**：确认状态仍与已批准的动作一致。

### 执行后验证（Post-action verification）

"工具返回了 200" 不等于验证。真正的验证会重新读取目标状态并确认副作用确实发生了。模式：

- 数据库更新：`UPDATE ... RETURNING *`，然后断言返回的行与预期状态匹配。
- 邮件发送：提交后检查已发送文件夹中的消息 ID。
- 文件写入：重新读取文件并计算哈希。
- API 调用：对目标资源执行后续的 `GET`。

如果验证失败，工作流处于已知的坏状态。Rollback 启动。

### Rollback 计划

Propose-then-commit（第 15 课）中的每一个有后果的动作都携带一个 rollback 计划。类型：

- **In-band rollback**：直接反转副作用（`INSERT` 后 `DELETE`，发送后 `Send-correction-email`）。
- **Compensating transaction**：一个中和原始动作的新动作（标准 SAGA 模式）。
- **Out-of-band rollback**：提醒人类，暂停工作流，将坏状态留给调查。

No-op rollback（"我们无法撤销此操作"）必须在提案中命名。没有 rollback 的动作需要在 commit 时更强的 HITL（human-in-the-loop，人在回路）（第 15 课的 challenge-and-response）。

### EU AI Act 第 14 条的实践解读

第 14 条要求高风险系统具备"有效的人类监督"。在实践中，实施者将其解读为：

- Checkpoint 可被审计员查询。
- Rollback 经过演练（至少端到端测试过一次）。
- 审计轨迹在部署后存活（checkpoint 后端不是临时的）。
- 失败的验证会被告警，而不是静默记录。

一个在 commit 中途崩溃、恢复并完成副作用的工作流，如果没有 verify + rollback 路径，无法通过第 14 条测试。

### 最尖锐的失效模式：双重执行

这个领域最常见的生产事故：

1. 动作被批准，idempotency key 为 k。
2. Commit 开始，执行，返回 200。
3. 工作流在持久化"已提交"状态前崩溃。
4. 工作流恢复；看到"已批准但未提交"；重新执行。
5. 副作用触发两次。

缓解措施：在执行前持久化一个"in-flight"意图，使用 idempotency key 执行，然后只在 post-action verification 成功后才标记为"已提交"。如果动作触发但状态写入失败，你知道需要验证并（必要时）重新触发。如果状态写入成功但动作失败，你通过恢复路径验证并精确触发一次。

## 使用

`code/main.py` 实现了一个带 idempotency、preconditions、verify 和 rollback 的 checkpointed workflow。驱动程序模拟四种场景：clean run、崩溃后重试（idempotency 捕获）、precondition 失败（工作流中止而不触发）、verify 失败（rollback 触发）。

## 交付

`outputs/skill-rollback-rehearsal.md` 为一个提议的工作流设计 rollback-rehearsal 测试，并审计 checkpoint 后端以确保审计轨迹的持久性。

## 练习

1. 运行 `code/main.py`。验证四种场景。对于 commit 期间崩溃的情况，确认动作在重试中只触发一次。

2. 修改"先标记完成，再执行"模式，使状态写入在动作之后触发。重新运行崩溃场景。测量有多少重复动作触发。

3. 为一个具体的生产动作（例如"发布到 Slack 频道"）设计 rollback 计划。分类为 in-band、compensating 或 out-of-band。 justify 选择。

4. 找一个你了解的工作流。识别每一个状态转换。为每个标记持久化需求（persist / do not persist）。统计你当前没有持久化的数量。

5. 演练式 rollback 测试：设计一个端到端测试，运行真实工作流，使其崩溃，并确认 rollback 路径触发。测试断言什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| Checkpoint | "保存点" | 每一个 graph-state transition 都被持久化到持久存储 |
| Lease | "Worker 声明" | 短期声明某个 worker 正在执行某个 run；崩溃时过期 |
| Precondition | "状态门" | 断言状态仍与已批准的动作一致 |
| Post-action verify | "重读检查" | 确认副作用确实在目标系统中发生了 |
| In-band rollback | "直接撤销" | 用反向操作直接反转副作用 |
| Compensating transaction | "SAGA 撤销" | 一个中和原始动作的新动作 |
| Mark-as-done-first | "状态写入顺序" | 在从 commit 返回前持久化已提交状态 |
| Article 14 | "EU AI Act 人类监督" | 实践要求：可查询的 checkpoint、演练过的 rollback、可审计的轨迹 |

## 延伸阅读

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — checkpoint 原语和租约恢复。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects 作为状态基板。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 监管基线。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时程工作流的可靠性框架。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Claude Code Routines 的工作流形态。
