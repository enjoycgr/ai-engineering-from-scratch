---
name: role-designer
description: 为multi-agent system (多智能体系统)生成role roster (角色名册)，为给定任务指定planner/executor/critic/verifier (规划者/执行者/批评者/验证者)及显式I/O schemas (输入/输出模式)。
version: 1.0.0
phase: 16
lesson: 08
tags: [multi-agent, role-specialization, metagpt, chatdev, verification]
---

给定一个任务，生成一个specialized role roster (特化角色名册)，包含I/O schemas (输入/输出模式)和一个deterministic verifier (确定性验证器)。准备映射到CrewAI、LangGraph、AutoGen或custom loops (自定义循环)。

生成内容：

1. **Role roster (角色名册)。** 3-5个角色。每个角色命名。至少包含：planner (规划者)、executor (执行者)、verifier (验证者)。critic (批评者)可选。
2. **I/O schema per role (每个角色的输入/输出模式)。** 对于每个角色：它consume (消费)什么（来自upstream role (上游角色)）和produce (生产)什么（schema (模式)，不是散文）。使用dataclass-style notation (数据类风格表示法)。
3. **Verifier specification (验证器规范)。** 命名deterministic check (确定性检查)：test suite (测试套件)、type checker (类型检查器)、schema validator (模式验证器)、linter (代码检查器)。描述pass/fail criteria (通过/失败标准)。
4. **Critic specification (批评者规范)（可选）。** 如果包含，命名它评判的subjective quality (主观质量)。具体的checklist (检查清单)，不是"good code (好代码)"。
5. **Communicative dehallucination rules (通信反幻觉规则)。** 命名每个downstream role (下游角色)在细节缺失时可以向上游发送的问题，这样它们就不会invent (编造)。
6. **Revision loop budget (修订循环预算)。** 在escalation to human (升级给人类)前的最大rounds (轮次)。默认2轮。
7. **Framework mapping (框架映射)。** 每行一句：如何在CrewAI、LangGraph、AutoGen中表达这个roster (名册)。

Hard rejects (硬性拒绝)：

- 任何没有deterministic verifier (确定性验证器)的roster (名册)。All-LLM rosters (全LLM名册)无法通过MAST check (MAST检查)。
- 模糊的I/O（"the executor returns output (执行者返回输出)"）。始终声明schema (模式)。
- Critic (批评者)和verifier (验证者)混淆。它们捕获不同的bug；如果两者都需要，两者都必须存在。

Refusal rules (拒绝规则)：

- 如果任务没有deterministic correctness check (确定性正确性检查)（纯粹generative work (生成性工作)、creative writing (创意写作)），拒绝并推荐human reviewer loop (人工审核循环)或multi-agent debate (多智能体辩论)（第07课）。
- 如果任务太小，不需要3+个角色（少于10分钟人工工作量），拒绝并推荐single-agent (单智能体)。

Output (输出)：一页role-design brief (角色设计简报)。以MAST failure-gap check (MAST故障缺口检查)收尾：确认至少存在一个deterministic verifier (确定性验证器)。
