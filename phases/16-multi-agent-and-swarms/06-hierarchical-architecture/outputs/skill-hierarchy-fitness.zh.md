---
name: hierarchy-fitness
description: 判断一个multi-agent task (多智能体任务)适合hierarchical (层级)、flat supervisor (扁平监督者)还是sequential (顺序)模式。指出重要的failure modes (故障模式)。
version: 1.0.0
phase: 16
lesson: 06
tags: [multi-agent, hierarchy, crewai, langgraph, decomposition-drift]
---

给定一个task description (任务描述)和可选的org structure (组织结构)，推荐coordination pattern (协调模式)（flat supervisor (扁平监督者)、hierarchical (层级)、sequential (顺序)）并列出需要防范的具体failure modes (故障模式)。

生成内容：

1. **Task shape analysis (任务形状分析)。** 该任务是单一线性流程、具有独立分支的fan-out (扇出)，还是具有自己sub-teams (子团队)的nested teams (嵌套团队)？说明理由。
2. **Pattern verdict (模式判定)。** Sequential (顺序)、flat supervisor (扁平监督者)或hierarchical (层级)。如果是hierarchical (层级)，指定depth (深度)（强烈推荐2层；只有强audit (审计)需求时才用3层）。
3. **Decomposition plan (分解计划)。** 顶层manager (管理者)应进行的精确拆分。对每个分支，命名sub-manager (子管理者)和bounded scope (有界范围)。
4. **Reconciliation budget (协调预算)。** 在顶层manager (管理者)必须做出决定之前允许的rounds (轮次)数量。默认2轮。
5. **Guardrails (护栏)。** 三个最低guardrails (护栏)：每层的canary worker (金丝雀工作者)、每次synthesis (合成)的provenance chain (来源链)、decomposition drift (分解漂移)告警。
6. **Failure-mode checklist (故障模式检查清单)。** 给定任务形状，{task-assignment error (任务分配错误)、output misinterpretation (输出误解)、consensus loop (共识循环)}中哪个最可能发生？描述每个模式的一个具体symptom (症状)和一个mitigation (缓解措施)。

Hard rejects (硬性拒绝)：

- 任何提议depth (深度) > 2但未命名具体audit (审计)或org requirement (组织需求)的推荐。
- 对single-linear-flow tasks (单一线性流程任务)使用hierarchical (层级)。这些应该使用sequential pipelines (顺序流水线)。
- 没有explicit reconciliation budget (明确协调预算)的设计。

Refusal rules (拒绝规则)：

- 如果任务足够简单，可以放入一个agent (智能体)（少于约10个tool calls (工具调用)），拒绝hierarchy (层级)并推荐single-agent (单智能体)。
- 如果任务没有自然的team boundaries (团队边界)（每个子步骤都依赖其他所有步骤），拒绝并推荐group chat pattern (群聊模式)。
- 如果用户想要hierarchical (层级)是为了"realism (真实感)"（因为人类组织很深），指出human hierarchy (人类层级)并不映射到LLM hierarchy (LLM层级)，并推荐更扁平的结构。

Output (输出)：一页brief (简报)。以pattern verdict (模式判定)开头，以三个最大风险及其guardrails (护栏)收尾。
