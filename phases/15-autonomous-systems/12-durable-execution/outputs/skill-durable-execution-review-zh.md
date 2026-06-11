---
name: durable-execution-review
description: 审核拟议的长时程智能体部署的持久执行形状（activities、确定性、checkpoint 后端、人类输入状态、恢复时的 HITL）。
version: 1.0.0
phase: 15
lesson: 12
tags: [durable-execution, workflows, checkpointing, temporal, langgraph, agents-sdk]
---

给定一个拟议的长时程智能体部署（Temporal + OpenAI Agents SDK、带 PostgreSQL checkpointer 的 LangGraph、Microsoft Agent Framework、Claude Code Routines、Cloudflare Durable Objects 或内部等效方案），根据持久执行模式审核其设计。

产出：

1. **Activity 清单。** 列出每个 activity（LLM 调用、工具调用、HTTP 请求、文件写入）。对每个，确认它已包装为带有 retry 策略、超时和幂等键的 activity。activity 信封外的原始 LLM 调用是一个可靠性漏洞。
2. **工作流确定性。** 识别工作流代码内的每个非确定性读取（挂钟时间、随机数、外部状态）。每个必须注册为副作用 activity，以便 replay 返回相同的值。隐藏的非确定性是 replay 漂移最常见的原因。
3. **Checkpoint 后端。** 命名后端（PostgreSQL、SQLite、Redis、Durable Objects）。确认它在部署后存活。SQLite 仅用于开发。Redis 需要 AOF 或快照配置。Cloudflare Durable Objects 是透明的，但需要唯一键规范。
4. **人类输入状态。** 确认 HITL 暂停是工作流的一等状态，而非轮询循环。工作流应阻塞在外部信号（批准队列、webhook、`interrupt()` 原语）上，该信号在批准到达时精确恢复。
5. **恢复时的 HITL 策略。** 对于崩溃后的任何恢复，说明在执行下一个 activity 之前是否需要新的 HITL。没有这一点，持久执行加上崩溃前授予的批准可能在上下文已改变时重新触发已批准的操作。对长时程至关重要。

硬性拒绝：
- LLM 调用未包装为 activity 的 Agent SDK 使用。
- 在部署后无法存活的 checkpoint 后端。
- 未将挂钟时间或随机数包装为 activity 的工作流。
- 将人类输入建模为轮询循环而非信号。
- 超过一小时且没有恢复时 HITL 策略的长时程运行。
- 没有在持久性之上叠加预算 kill switch（第 13 课）的运行。

拒绝规则：
- 如果用户提议的持久工作流对副作用 activity 没有显式幂等性，拒绝并要求先提供幂等键。否则 retry 会导致重复执行。
- 如果用户无法展示 replay 测试（运行工作流、中间崩溃、replay、断言无双副作用），拒绝并要求在生产前进行该测试。
- 如果用户提议没有 HITL checkpoint 的 24 小时无人值守运行，拒绝。35 分钟衰减（第 12 课笔记）使其成为可靠性问题，即使持久性设计正确。

输出格式：

返回一份设计审核备忘录，包含：
- **Activity 表**（activity、retry 策略、超时、幂等键）
- **确定性审核**（非确定性读取及其处理方式）
- **Checkpoint 后端**（名称、部署后存活 y/n、replay 测试状态）
- **HITL 状态形状**（一等状态 / 轮询 / 缺失）
- **恢复时的 HITL 策略**（显式，附理由）
- **就绪度**（production / staging / research-only）
