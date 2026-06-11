---
name: handoff-designer
description: 为Swarm/Agents-SDK风格的系统设计handoff topology (交接拓扑)：存在哪些agent (智能体)、它们可以调用哪些handoffs (交接)、什么context transfers (上下文转移)。
version: 1.0.0
phase: 16
lesson: 11
tags: [multi-agent, swarm, handoff, openai-agents-sdk]
---

给定一个user-facing task (面向用户的任务)（通常是triage (分诊)或skill-based routing (基于技能的路由)），生成一个可映射到OpenAI Swarm或OpenAI Agents SDK的handoff topology (交接拓扑)。

生成内容：

1. **Agent roster (智能体名册)。** 每个agent：name (名称)、one-sentence purpose (一句话目的)、tools (工具)以及它可以hand off (交接)给哪些其他agent。
2. **Handoff functions (交接函数)。** 每个agent的tool signatures (工具签名)。每个handoff function返回一个target Agent (目标智能体)。
3. **Context transfer policy (上下文转移策略)。** 在每个handoff edge (交接边上)：full history (完整历史)、last N messages (最近N条消息)或summarized snapshot (摘要快照)。说明理由。
4. **Guardrails (护栏)。** 每个agent的input validation (输入验证)（允许哪些prompts (提示词)触发向敏感specialists (专家)的handoffs (交接)），必要时在handoff上进行authentication (认证)。
5. **Loop detection (循环检测)。** 检测ping-pong (乒乓)的规则（例如，"A交接给B；B又交接回A"连续发生超过一次）。
6. **Fallback behavior (回退行为)。** 如果handoff target (交接目标)缺失（removed agent (已移除的智能体)、auth failure (认证失败)），哪个agent处理该session (会话)。
7. **Session / memory plan (会话/内存计划)。** 是否使用Agents SDK sessions、caller-managed memory (调用者管理内存)或no memory (无内存)。

Hard rejects (硬性拒绝)：

- 任何没有loop detection (循环检测)的handoff design (交接设计)。
- 向具有不同tool permissions (工具权限)的specialists (专家)传递full history (完整历史)的handoff functions（安全风险）。
- 假设Swarm的stateless behavior (无状态行为)但又需要multi-turn memory (多轮内存)的设计——改用Agents SDK sessions。

Refusal rules (拒绝规则)：

- 如果任务需要parallel execution (并行执行)，拒绝Swarm并推荐supervisor (监督者)（第05课）。
- 如果任务需要deterministic audit/replay (确定性审计/重放)，拒绝并推荐LangGraph static graph (静态图)。
- 如果任务是simple DAG of stages (简单阶段DAG)（research → code → review），推荐CrewAI Sequential (顺序模式)。

Output (输出)：一页handoff brief (交接简报)。以security note (安全注释)收尾，说明prompt injection (提示注入)如何可能触发不必要的handoffs (交接)以及什么guardrails (护栏)可以阻止它。
