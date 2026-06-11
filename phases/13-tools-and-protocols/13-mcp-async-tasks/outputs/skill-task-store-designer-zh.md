---
name: task-store-designer
description: 为长时间运行工具设计任务存储（状态形状、ttl、持久性），选择正确的 taskSupport 标志并草拟进度通知。
version: 1.0.0
phase: 13
lesson: 13
tags: [mcp, tasks, async, sep-1686]
---

给定一个长时间运行的工具（研究、构建、导出），设计任务存储、生命周期和通知策略。

产出：

1. 状态形状。Task JSON 包含 `id`、`state`、`progress`、`result`、`error`、`ttl_ms`、`created_at`。
2. 持久层。为 ttl 内崩溃恢复选择 SQLite（推荐）、Redis 或文件系统。
3. taskSupport 标志。`forbidden` 用于 <1 秒工具；`optional` 用于 1-30 秒工具；`required` 用于 >30 秒或不可预测的工具。
4. 进度通知。设计 `notifications/tasks/updated` 载荷包含 `taskId`、`state`、`progress`（0.0-1.0）。
5. 取消语义。工作者线程通过共享事件检查取消；终端状态忽略重复取消。

硬拒绝项：
- 任何无 ttl 的任务。服务器必须承诺状态保留窗口。
- 任何无持久层的 "required" 任务。崩溃将丢失飞行中的工作。
- 任何不在终端状态上停止进度通知的任务。

拒绝规则：
- 如果工具运行 <1 秒，拒绝使其成为任务；同步调用是正确的。
- 如果用户要求无限制 ttl，拒绝并建议使用 15 分钟上限加可配置例外。
- 如果任务产生流式结果（例如实时日志），建议通过 `notifications/progress` 流式传输并仅在完成时存储最终结果。

输出：状态机图（text 或 mermaid）、任务存储 schema、示例 `tasks/status` 响应和进度通知 JSON。以推荐的默认 ttl 结束。
