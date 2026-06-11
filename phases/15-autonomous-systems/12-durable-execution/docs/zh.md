# 长时程后台智能体：持久执行

> 生产级长时程智能体不在 `while True` 中运行。每次 LLM 调用都成为一个带有 checkpoint、retry 和 replay 的 activity。Temporal 的 OpenAI Agents SDK 集成于 2026 年 3 月 GA。Claude Code Routines（Anthropic）运行预定的 Claude Code 调用而无需持久本地进程。会话在人类输入时暂停，在部署后存活，并从以 `thread_id` 为键的最新 checkpoint 恢复。在新的易用性背后是一个旧模式 —— 工作流编排 —— 加上一个新输入：LLM 调用作为必须在恢复时确定性 replay 的非确定性 activity。

**类型：** 学习
**语言：** Python（标准库，最小持久执行状态机）
**前置：** Phase 15 · 10（权限模式），Phase 15 · 01（长时程智能体）
**时间：** ~60 分钟

## 问题

考虑一个运行四小时的智能体。它调用三个工具，向用户提示两次，并进行四十次 LLM 调用。运行到一半时，其所在主机重启。会发生什么？

- 在天真的 `while True` 循环中：一切丢失。运行从头重启。三个工具调用（具有真实副作用）再次执行。用户被再次提示已经批准过的事项。四十次 LLM 调用被重复计费。
- 在持久执行中：运行从最近的 checkpoint 恢复。已完成的 activity 不会重新执行；其结果从持久日志中 replay。用户不必重新批准已经批准过的事项。已进行的 LLM 调用不会被重复计费。

这是工作流引擎已经交付十年的相同模式（Temporal、Cadence、Uber 的 Cherami）。新的地方在于 LLM 调用现在是一种 activity —— 非确定性、昂贵、有副作用 —— 并且它们干净地适配这一模式。

本课的主题是：长时程可靠性会衰减（METR 观察到"35 分钟衰减" —— 成功率随时程大致呈二次方下降）。持久执行使运行能够超过可靠性剖面支持的长度，如果设计正确，这是一种新的安全失败方式；如果设计错误，则是一种新的不安全失败方式。

## 概念

### Activities、workflows 和 replay

- **Workflow（工作流）：** 确定性编排代码。定义 activity 序列、分支、等待。必须是确定性的，以便可以从事件日志 replay 而不产生意外分歧。
- **Activity（活动）：** 非确定性、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求。每个 activity 在开始前和完成后都被记录其输入和输出。
- **Event log（事件日志）：** 持久后端存储。记录每个 activity 的启动、完成、失败、retry，以及每个工作流决策。
- **Replay（回放）：** 恢复时，工作流代码从头重新运行；每个已完成的 activity 返回其记录的结果而不重新执行。只有未完成的 activity 才真正运行。

这与 React 针对虚拟 DOM 重新渲染，或 Git 从 commits 重建工作树的形状相同。编排器中的确定性使持久性变得廉价。

### 为什么 LLM 调用适配这一模式

LLM 调用具有以下特征：
- 非确定性（temperature > 0；即使 temperature 为 0，跨模型版本也会漂移）。
- 昂贵（金钱和延迟）。
- 可能失败（速率限制、超时）。
- 有副作用（如果它们调用工具）。

这正是 activity 的 profile。将每次 LLM 调用包装为 activity 可以获得带指数退避的 retry、跨重启的 checkpointing，以及可 replay 的调试轨迹。

### 以 `thread_id` 为键的 Checkpoints

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛于相同的 API 形状：`thread_id`（或等效物）标识会话；每个状态转换持久化到后端（PostgreSQL 默认、SQLite 用于开发、Redis 用于缓存）；恢复时读取最新的 checkpoint。

后端选择很重要：

- **PostgreSQL：** 持久、可查询、在部署后存活。LangGraph 的默认选择。
- **SQLite：** 仅本地开发；跨主机丢失数据。
- **Redis：** 快速但短暂，除非配置 AOF/快照。
- **Cloudflare Durable Objects：** 透明分布式；由唯一键限定范围；可存活数小时到数周。

### 人类输入作为一等状态

Propose-then-commit（第 15 课）需要一个持久的"等待人类"状态。工作流暂停，外部队列持有待处理请求，批准从该确切点恢复。没有持久性，这是尽力而为；有了持久性，隔夜批准到达，工作流在第二天早上继续。

### 35 分钟衰减

METR 观察到，超过约 35 分钟连续运行后，每个测量的智能体类别都显示可靠性衰减。将任务时长翻倍大致将失败率翻两番。持久执行并不能修复这一点；它让你运行超过可靠性剖面支持的长度。安全模式是将持久性与重新进入时需要新 HITL 的 checkpoint 结合，并与预算 kill switch（第 13 课）结合，以限制总计算量而不考虑挂钟时间。

### 持久执行是错误答案的情况

- 运行时间短于几分钟且无人类输入。开销 > 收益。
- 严格只读的信息检索。
- 正确性要求在一个上下文窗口内端到端完成的任务（某些推理任务；某些一次性生成）。

## 使用

`code/main.py` 在标准库 Python 中实现了一个最小持久执行引擎。它支持：

- 将输入和输出记录到 JSON 事件日志的 `@activity` 装饰器。
- 编排 activity 的工作流函数。
- 一个 `run_or_replay(workflow, event_log)` 函数，replay 已完成的 activity 而不重新执行它们。

驱动程序模拟一个三 activity 工作流，在中间崩溃，并展示 (a) 天真 retry 重新执行所有内容与 (b) replay 只运行缺失的 activity 的对比。

## 交付

`outputs/skill-durable-execution-review.md` 审核一个拟议的长时程智能体部署的持久执行形状是否正确：activities、确定性、checkpoint 后端、人类输入状态，以及恢复时的 HITL 策略。

## 练习

1. 运行 `code/main.py`。观察天真 retry 与 replay 之间 activity 执行次数的差异。改变崩溃点并展示 replay 次数相应变化。
2. 将玩具引擎显式改为使用 `thread_id`。模拟两个共享引擎的并发会话，并确认它们的事件日志不会冲突。
3. 在玩具引擎中取一个 activity。引入一个非确定性（工作流决策内的挂钟时间戳）。展示 replay 时的分歧。解释真实引擎如何处理这个问题（副作用注册、`Workflow.now()` API）。
4. 阅读 LangChain "Runtime behind production deep agents" 文章。列出运行时持久化的每个状态，并命名每个状态覆盖的失败模式。
5. 为一个 6 小时的自主编码任务设计 checkpoint 策略。你在哪里 checkpoint？崩溃后恢复是什么样子？什么需要新的 HITL？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| Workflow | "智能体的脚本" | 确定性编排代码；可从事件日志 replay |
| Activity | "一步" | 非确定性单元（LLM 调用、工具调用）；前后记录 |
| Event log | "后端存储" | 每个状态转换的持久记录 |
| Replay | "恢复" | 重新运行工作流；已完成的 activity 返回记录的结果而不重新执行 |
| Checkpoint | "保存点" | 以 thread_id 为键的持久状态；恢复时取最新 |
| thread_id | "会话键" | 限定持久状态范围的标识符 |
| 35-minute degradation | "可靠性衰减" | METR：成功率随时程大致呈二次方下降 |
| Non-determinism | "replay 漂移" | 挂钟时间、随机数、LLM 输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算、轮次和恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent 形状。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体的运行时需求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM 调用的 activity 形状。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35 分钟衰减参考。
