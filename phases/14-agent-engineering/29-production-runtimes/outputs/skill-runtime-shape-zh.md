---
name: runtime-shape
description: 为任务选择生产运行时形态（request-response、streaming、queue、event、cron、durable）并连接可观测性。
version: 1.0.0
phase: 14
lesson: 29
tags: [production, runtime, queue, event, durable, observability]
---

给定任务类别（预期持续时间、步数、触发类型、延迟预算），选择运行时形态。

决策：

1. < 30 秒，用户等待 -> **request-response**。
2. 渐进式 UX 或语音 -> **streaming**。
3. 分钟到小时，用户不等待 -> **queue-based**。
4. 响应外部事件 -> **event-driven**。
5. 定期内务 -> **cron**。
6. 上述任何一种重启成本高的情况 -> 添加 **durable execution**。

产出：

1. 你的技术栈中的形态脚手架。
2. 可观测性：OTel GenAI spans（Lesson 23），后端连接（Lesson 24）。
3. 队列：DLQ + 重试策略 + 队列深度指标。
4. 事件：显式订阅注册表 + 重放路径。
5. Cron：锁文件或分布式锁以防止重叠运行。
6. 持久：检查点后端 + 恢复语义。

硬性拒绝：

- 5 分钟任务使用同步 HTTP。用户挂断；工作者堆积。
- 没有 DLQ 的基于队列。失败的作业消失。
- 没有导出追踪的后台工作。故障 invisible 直到用户投诉。
- "没有持久状态，我们只重试。"长程必须检查点。

拒绝规则：

- 如果产品有 SLA + 重放要求，拒绝 swarm 拓扑 + 非持久运行时。
- 如果任务受合规约束，拒绝没有审计跟踪的事件驱动。
- 如果用户想要 cron + 无锁，拒绝。重叠的 cron 运行至少是重复工作，最坏是数据损坏。

输出：运行时脚手架 + 可观测性钩子 + 带 SLA、重试策略、检查点选择的 README。结尾附上"接下来读什么"，指向 Lesson 23（OTel）、Lesson 24（可观测性），或 Lesson 17（Managed Agents 了解托管长程）。
