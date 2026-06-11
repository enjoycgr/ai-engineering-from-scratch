---
name: stategraph-designer
description: 将一个 agent 任务转化为 LangGraph StateGraph，包含命名节点、类型化状态、reducers、checkpointer 和人工中断。
version: 1.0.0
phase: 11
lesson: 16
tags: [langgraph, stategraph, checkpointer, interrupt, time-travel, react-agent, human-in-the-loop]
---

给定 agent 任务（面向用户的目标、可用工具、预期轮数、具有安全影响半径的副作用、持久化需求、目标延迟预算），输出：

1. 节点列表。命名每个离散步：LLM thinker、每个 tool runner、每个人工 review 步骤、任何 summarizer 或 critic、任何 retriever。如果任何节点触及多个关注点，拒绝该设计；拆分它。
2. 状态 schema。TypedDict（或 Pydantic）字段，每个 list 都有 reducer。Message log 始终使用 Annotated[list, add_messages]。将任何任务特定列表从 messages 中提升出来（plan、budget counter、retrieved-docs list），以便 reducer 在并行更新下保持正确。
3. 边映射。下一步是确定性的地方使用 static edges。模型选择下一步的地方使用 conditional edges 并带命名 router 函数。拒绝任何 router 函数依赖你尚未在先前节点中进行的 fresh LLM call 的图。
4. 中断放置。在每个具有不可逆副作用的节点上使用 interrupt_before（写入、删除、支付、具有成本的外部 API 调用）。在模型节点上使用 interrupt_after，当输出验证在单独进程中运行时。拒绝在 any side-effecting 节点上使用 interrupt_after；到那时副作用已经发生。
5. Checkpointer。测试仅用 MemorySaver。对于必须 survive restart 的任何环境，从 PostgresSaver、SQLiteSaver、RedisSaver 中选择。确认 thread_id 策略（per-user、per-session、per-conversation）和 checkpoint TTL。

拒绝交付没有 checkpointer 的 LangGraph。没有 checkpointer 意味着没有 resume、没有 time-travel、没有 human-in-the-loop replay。拒绝交付 messages 字段没有 add_messages 的那个；第二次写入会静默覆盖第一次，半轮对话消失。拒绝每个转换都是 planner LLM 路由的 conditional edge 的图；那是 AutoGen 多此一举，每轮都烧 token。

示例输入："Refund-handling agent over Anthropic Claude with three tools (lookup_order, issue_refund, send_email), must pause for a human before any refund over 100 dollars, must resume after server restart, p95 latency budget 8 seconds."

示例输出：
- Nodes: agent (LLM call), lookup_tool, refund_tool, email_tool, human_review.
- State: messages with add_messages, order_context (overwrite), refund_amount (overwrite), reviewer_decision (overwrite).
- Edges: agent to should_continue router with branches lookup_tool, refund_tool, email_tool, human_review, END. Tool nodes go back to agent.
- Interrupts: interrupt_before on refund_tool when refund_amount > 100. No interrupt on lookup_tool or email_tool.
- Checkpointer: PostgresSaver with thread_id "user:{user_id}:case:{case_id}" and 30-day TTL.
