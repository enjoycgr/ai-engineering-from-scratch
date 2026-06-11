# 层级架构及其失效模式

> 层级架构是嵌套的 supervisor (监督者)。manager agent (管理者智能体) 管理 sub-manager (子管理者)，sub-manager 再管理 worker (工作者)。CrewAI 的 `Process.hierarchical` 是教科书版本：一个 `manager_llm` 动态地分配任务并验证输出。LangGraph 的等价实现是 `create_supervisor(create_supervisor(...))`。当任务对应真实的组织架构图时，这是自然的模式。但它也是最可能 collapse (崩溃) 成管理循环的模式——manager agent 分配工作不当、误解子输出，或无法达成共识。Sequential (顺序) 模式往往更优。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern)
**Time:** ~60 minutes

## Problem

一旦理解了 supervisor pattern (监督者模式)，自然的下一步就是"如果 worker 本身也是 supervisor 呢？"团队有子团队；公司有部门的部门。层级架构反映了这一点。

问题在于：LLM manager 不同于人类 manager。人类 manager 对其下属的能力有稳定的先验认知。而 LLM manager 每轮都根据其 context window (上下文窗口) 中的内容重新推理组织架构。context 中的微小 drift (漂移) 就会导致整棵树错误分配工作。

## Concept

### The shape

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

每个内部节点负责规划、委派和综合。只有叶子节点执行实际工作。

### Where it shines

- **Clear org mapping.** 如果真实任务是按部门划分的（"法律部门审查文档，财务部门审查文档，工程部门审查文档，然后为高管总结"），层级结构是明确的。
- **Local summarization.** 每个 sub-manager 在 top manager 看到之前先综合其团队的输出。Top manager 看到的是三个 sub-manager 的摘要，而不是十五个 worker 的输出。

### Where it breaks

2026 年的 post-mortems (事后分析) 反复发现的三种失效模式：

1. **Task assignment error (任务分配错误).** Manager 读取目标后，hallucinate (幻觉) 出一个分解方案，并委派给了错误的 sub-manager。由于 sub-manager 会 obediently (顺从地) 处理被分配的任务，错误只在 top synthesis (顶层综合) 时才暴露——比人类能够发现的位置高了一层。
2. **Output misinterpretation (输出误解).** Sub-manager 返回"无法验证声明 X"。Top manager 总结为"声明 X 未确认"。含义在每一层都发生 drift (漂移)。
3. **Consensus loops (共识循环).** 两个 sub-manager 意见不一致；top manager 要求他们协调；他们向下重新委派；worker 重新运行；sub-manager 返回略有不同的答案；循环。CrewAI 的 `Process.hierarchical` 通过 step limit (步骤限制) 来防范这种情况，但限制本身现在变成了一个 hyperparameter (超参数)。

### The deciding question

Sequential (顺序流水线) vs hierarchical (层级)：你的任务是否真的有独立的 sub-team (子团队)，还是只是一个假装成树的 linear flow (线性流程)？如果是后者，使用 sequential。如果是前者，使用 hierarchical，但要设定明确的 reconciliation (协调) 规则。

### CrewAI's implementation

`Process.hierarchical` 在一个 specialist crew (专家小组) 之上连接了一个 manager LLM。该 manager：

- 接收顶层任务，
- 向 crew 分配 subtask (子任务)，
- 评估 crew 的输出，
- 决定接受、重新委派还是迭代。

Documentation: https://docs.crewai.com/en/introduction (在 Core Concepts 下查找 "Hierarchical Process")。

### LangGraph's implementation

LangGraph 使用嵌套的 `create_supervisor` 调用。内部的 supervisor 有自己的 graph (图)；外部的 supervisor 将内部 graph 视为一个 opaque node (不透明节点)。这在调试方面比 CrewAI 更清晰（你可以分别逐步执行每个 graph），但更难表达树的动态重塑。

Reference: https://reference.langchain.com/python/langgraph-supervisor。

## Build It

`code/main.py` 运行一个 3-level hierarchy (3 层层级结构)：

- top manager：将任务拆分为 "engineering" 和 "legal" 分支，
- engineering sub-manager：拆分为 "frontend" 和 "backend" worker，
- legal sub-manager：一个 worker。

Demo 对比了 happy path (正常路径)（所有人达成一致）和一条 **perturbed path (扰动路径)**，其中 top manager 的 decomposition (分解) 将 "legal" 误标为 "finance"，并观察错误如何 cascade (级联)——sub-manager 顺从地执行财务工作，top synthesizer (顶层综合器) 报告财务发现，原始的法律问题无人回答。

Run:

```
python3 code/main.py
```

输出显示了两条路径，并清晰对比了"问了什么"vs"交付了什么"。

## Use It

`outputs/skill-hierarchy-fitness.md` 评估给定任务应该使用 hierarchical、sequential 还是 flat supervisor (扁平监督者)。输入：任务描述、组织架构、reconciliation budget (协调预算)。输出：模式推荐及需要防范的具体失效模式。

## Ship It

如果你要部署 hierarchical 架构：

- **Cap tree depth at 2.** 三层已经让大多数错误无法被观测到。
- **Explicit reconciliation budget.** 设定 top manager 必须做出决定前的最大轮数。通常为 2。
- **Provenance on every synthesis.** 每个节点的 summary 必须引用哪些 leaf output 产生了它。
- **Alert on decomposition drift.** 记录 manager 每步的 decomposition；与用户查询进行 diff。如果 decomposition 不再覆盖查询，触发 alert (告警)。

## Exercises

1. 运行 `code/main.py` 并对比 happy vs perturbed。在 top output 完全偏离用户问题之前，需要经过多少层 manager hand-off (管理者交接)？
2. 添加第三层（top → sub → sub-sub → worker）。测量 perturbed path 在深度增加时自我纠正 vs 完全偏离的频率。
3. 在每个 sub-manager 实现一个 "canary" worker，始终被询问原始用户问题而不做修改。使用 canary 的答案来检测 decomposition drift。当 canary 与 synthesized answer (综合答案) 不一致时，manager 应该如何反应？
4. 阅读 CrewAI 的 `Process.hierarchical` 文档。找出 CrewAI 应用的一个具体 guardrail (防护栏)（step limit、manager_llm constraint），并描述它针对什么失效模式。
5. 对比嵌套的 LangGraph supervisor 与 CrewAI hierarchical。哪种让 reconciliation loop 更容易检测？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hierarchical | "Org chart pattern" | Supervisors over supervisors; only leaves do work. |
| Manager LLM | "The boss" | The LLM that decomposes, assigns, and validates at an internal node. |
| Decomposition drift | "The boss lost the plot" | Top manager's split no longer covers the original question. |
| Reconciliation loop | "Endless meetings" | Sub-managers disagree; top re-delegates; workers re-run; loop until budget exhausted. |
| Depth-2 ceiling | "Don't go deeper than 2 levels" | Empirical guardrail: 3+ levels collapses observability. |
| Canary question | "Ground truth at every level" | A worker that is always asked the original query unchanged, to detect drift. |
| Provenance chain | "Who said what" | Trace from each synthesis back to the leaf outputs that produced it. |

## Further Reading

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — textbook hierarchical with a manager LLM
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — nested supervisor via `create_supervisor`
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — why Anthropic deliberately chose flat supervisor over hierarchical
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy; section on coordination failures documents decomposition drift
