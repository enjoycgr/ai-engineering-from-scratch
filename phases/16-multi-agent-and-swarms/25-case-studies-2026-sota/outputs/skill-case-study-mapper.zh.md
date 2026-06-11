---
name: case-study-mapper
description: 将提议的multi-agent system design (多智能体系统设计)映射到最接近的2026 production reference (生产参考)（Anthropic Research、MetaGPT/ChatDev或OpenClaw/Moltbook）。展示已知trade-offs (权衡)、推荐的framework (框架)以及已在生产中测试的specific design decisions (具体设计决策)。
version: 1.0.0
phase: 16
lesson: 25
tags: [multi-agent, case-studies, production, framework-selection, reference-architectures]
---

给定一个proposed multi-agent system design (提议的多智能体系统设计)，选择最接近的canonical 2026 case study (经典案例研究)并进行适配。

生成内容：

1. **Design fingerprint (设计指纹)。** Task type (任务类型)（research (研究) / engineering (工程) / population (群体) / automation (自动化)）、agent count (智能体数量)、verification requirement (验证需求)、runtime duration (运行时长)、role distinctness (角色区分度)、user-facing network exposure (面向用户的网络暴露)。
2. **Closest case study (最接近的案例研究)。**
   - **Anthropic Research** 如果：research or knowledge-retrieval task (研究或知识检索任务)、verification mandatory (验证强制)、multi-hour runs (数小时运行)、agents主要因context and scope (上下文和范围)而不同（fresh-context subagents (全新上下文子智能体)获胜）。
   - **MetaGPT / ChatDev** 如果：engineering or structured workflow (工程或结构化工作流)、roles clearly distinguishable (角色明显可区分)（planner / coder / reviewer / tester）、handoff artifacts are well-typed (交接产物类型良好)。
   - **OpenClaw / Moltbook** 如果：population-scale (群体规模)、user-facing agent network (面向用户的智能体网络)、prompt-injection是meaningful threat (实质性威胁)、emergent economy matters (涌现经济重要)。
3. **Patterns to copy (复制的模式)。** 所选案例研究中适用的specific design decisions (具体设计决策)：fresh-context subagents、rainbow deploy、communicative dehallucination、DAG routing、unwritable verifier、substrate-level security。
4. **Framework recommendation (框架推荐)。** LangGraph、CrewAI、AG2、Microsoft Agent Framework、OpenAI Agents SDK、Google ADK、Anthropic Claude Agent SDK或custom (自定义)。默认使用案例研究的typical framework (典型框架)；注意如果特定设计有更好的fit (适配)。
5. **Anti-patterns from the case (案例中的反模式)。** 参考案例发现NOT to work (不起作用)的事物。在新设计中避免。
6. **Cost projection (成本预测)。** 预期token multiplier (token倍数)（Anthropic Research: ~15x；MetaGPT: ~5x；OpenClaw: 取决于network effects (网络效应)）。预期wall-clock和dollar cost range (美元成本范围)。
7. **Evaluation approach (评估方法)。** 哪个benchmark（MARBLE、SWE-bench Pro、internal (内部)）相关；case-study baseline的reasonable delta (合理增量)是多少。

Hard rejects (硬性拒绝)：

- 在任务有correctness requirements (正确性需求)时忽略verification (验证)的设计。每个案例研究都支付verification tax (验证税)。
- 声称new substrate (新底层)而未承认prompt-injection是attack surface (攻击面)的设计。OpenClaw/Moltbook案例显示这是production concern (生产关注点)，不是hypothetical (假设性的)。
- 未映射到任何case study的"Revolutionary (革命性)"声明。Multi-agent自2024年以来已在production中；novel claims需要explicit comparison (显式比较)。
- 未经justification (论证)就跳过MCP或A2A adoption的设计。Protocol support是table stakes (基本要求)。

Refusal rules (拒绝规则)：

- 如果设计没有clear task type (明确的任务类型)，推荐在挑选case study前先scoping the task (确定任务范围)。"Multi-agent for everything (万能的multi-agent)"不是设计。
- 如果设计声称production readiness (生产就绪)但没有failure-mode audit (故障模式审计)，推荐在reference mapping前先进行MAST-style audit (MAST风格审计)（第23课）。
- 如果设计 purely experimental / research (纯粹实验/研究)，注意哪些方面在采用任何case study的production patterns前需要hardening (加固)。

Output (输出)：两页brief (简报)。以one-sentence summary (单句摘要)开头（"Closest case study: MetaGPT / ChatDev. Adopt role-SOP decomposition, communicative dehallucination, and structured handoff artifacts; use CrewAI or custom."），然后是上述七个部分。以90-day adaptation plan (90天适配计划)收尾：从reference复制什么、customize (定制)什么、以及针对benchmarks验证什么。
