---
name: supervisor-designer
description: 为给定的research-style query (研究型查询)设计一个supervisor/orchestrator-worker (监督者/编排器-工作者)系统，指定主prompt、worker角色、分解规则和合成模板。
version: 1.0.0
phase: 16
lesson: 05
tags: [multi-agent, supervisor, orchestrator, anthropic-research, langgraph]
---

给定一个受益于parallel subagent research (并行子智能体研究)的用户查询，生成一个supervisor-pattern (监督者模式)设计，准备接入任何框架（LangGraph、OpenAI Agents SDK、CrewAI Hierarchical）。

生成内容：

1. **Complexity estimate (复杂度估计)。** 该查询是简单（1个agent、3-10个tool calls (工具调用)）、中等（2-4个workers (工作者)）还是复杂（5+个workers）？使用Anthropic的scale-effort heuristic (规模-工作量启发式)用一句话说明理由。
2. **Lead system prompt (主系统提示词)。** 必须包含：(a) decomposition instructions (分解指令)，(b) synthesis instructions (合成指令)，(c) 明确规则：lead (主控)从不读取raw source content (原始源内容)，只读取worker summaries (工作者摘要)。
3. **Worker system prompts (工作者系统提示词)。** 每个角色一个，每个都命名其narrow scope (狭窄范围)和lead期望的output format (输出格式)。
4. **Sub-question decomposition rules (子问题分解规则)。** Lead如何拆分查询？Broad-first-then-narrow (先广后窄)还是direct decomposition (直接分解)？什么情况下disqualifies (排除)一个sub-question (子问题)（与另一个重叠、过于宽泛）？
5. **Synthesis template (合成模板)。** 显式conflict-handling rule (冲突处理规则)：如果两个worker返回了contradictory facts (矛盾事实)，合成必须surface the disagreement (呈现分歧)而不是 silently picking one (默默选择一个)。
6. **Model pairing (模型配对)。** Lead使用哪个model (模型)（reasoning tier (推理层级)），worker使用哪个（faster/cheaper tier (更快/更便宜层级)）。解释tradeoff (权衡)。
7. **Observability requirements (可观测性要求)。** 最低trace points (跟踪点)：plan (计划)、每个worker start/end (开始/结束)、synthesis input (合成输入)、synthesis output (合成输出)。

Hard rejects (硬性拒绝)：

- 任何让lead本身进行tool-use (工具使用)的设计。Lead只负责planning (计划)和synthesis (合成)。
- 允许scope drift (范围漂移)的worker prompts（例如，"research anything related to X (研究与X相关的任何内容)"而没有bound (边界)）。
- 隐藏conflicts (冲突)的synthesis templates (合成模板)。

Refusal rules (拒绝规则)：

- 如果查询简单（估计总共少于10个tool calls (工具调用)），拒绝该设计并推荐single-agent (单智能体)。引用Anthropic的15× token cost finding (token成本发现)。
- 如果查询是sequential (顺序的)（步骤2需要步骤1的输出），拒绝并推荐pipeline/chain pattern (流水线/链式模式)。
- 如果用户优化的是determinism (确定性)和audit (审计)，拒绝supervisor (监督者)并推荐LangGraph static graph (静态图)。

Output (输出)：一页design brief (设计简报)。以complexity estimate (复杂度估计)和pattern-fit verdict (模式适配判定)（"supervisor fits (监督者适合)"）开头。如果系统将持续运行，以rainbow-deployment reminder (彩虹部署提醒)收尾。
