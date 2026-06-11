# 角色特化（Role Specialization）—— Planner、Critic、Executor、Verifier

> 2026 年最常见的多智能体（multi-agent）分解方式：一个智能体（agent）负责规划，一个执行，一个批判或验证。MetaGPT（arXiv:2308.00352）将其形式化为嵌入角色提示词（role prompts）中的 SOP——产品经理、架构师、项目经理、工程师、QA 工程师——遵循 `Code = SOP(Team)`。ChatDev（arXiv:2307.07924）通过"聊天链"（chat chain）将设计师、程序员、审查员、测试员串联起来，并引入"交流式去幻觉"（communicative dehallucination，智能体在缺失细节时主动请求补充）。Verifier（验证器）是承重角色：Cemri 等人（MAST，arXiv:2503.13657）指出，每一个多智能体失败都可以追溯到缺失或损坏的验证环节。PwC 报告称，在 CrewAI 中引入结构化验证循环后，准确率获得了 7 倍提升（10% → 70%）。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 分钟

## 问题

通用多智能体系统产生通用输出。群聊中的三个程序员写出三种大同小异的平庸代码。你可以增加更多智能体、增加更多轮次，仍然无法跨越质量门槛。

解决方案不是更多智能体——而是*不同*的智能体。分配不同的角色。给 critic（批判者）赋予 planner（规划者）没有的工具。给 verifier（验证器）一个客观的测试套件。现在系统拥有基于事实修正的内部分歧，而不只是并行猜测。

## 概念

### 四个经典角色

**Planner（规划者）。** 读取目标，产出步骤列表或规格说明。工具：知识检索、文档。输出：结构化计划。

**Executor（执行者）。** 每次读取计划的一个步骤，产出产物（artifact）。工具：实际工作工具（代码编译器、shell、API 客户端）。输出：产物。

**Critic（批判者）。** 对照 planner 的意图审阅 executor 的输出。工具：对产物的只读访问、静态分析。输出：接受/拒绝及理由。

**Verifier（验证器）。** 阅读产物并运行确定性检查。工具：测试运行器、类型检查器、schema 验证器。输出：通过/失败及证据。

Critic 是主观的、有观点的，通常基于 LLM。Verifier 是客观的、确定性的，通常基于代码。它们不是同一个角色。

### MetaGPT 的 SOP 模式

MetaGPT（arXiv:2308.00352）将软件工程 SOP 编码为角色提示词：

- **Product Manager（产品经理）** 编写 PRD。
- **Architect（架构师）** 产出系统设计。
- **Project Manager（项目经理）** 拆分任务。
- **Engineer（工程师）** 实现代码。
- **QA Engineer（QA 工程师）** 运行测试。

每个角色都有严格的输入/输出 schema。角色提示词说明该角色*是什么*以及它*必须产出什么*。`Code = SOP(Team)` 的表述——确定性的 SOP 将一群 LLM 变成可预测的流水线（pipeline）。

### ChatDev 的交流式去幻觉

ChatDev 增加了一个关键动作：当执行者需要某个计划中未提供的具体细节时，它会明确向设计师询问后再继续。这防止了 LLM 的经典失败模式——看似合理地编造细节。

实现方式：角色提示词中包含"当你需要未被提供的具体信息时，先按名称向相关角色询问，再产出输出。"

### 为什么验证器最重要

Cemri 等人（MAST）追踪了 1642 个多智能体执行失败。21.3% 是验证缺口（verification gaps）——系统交付了一个没人检查过的答案。剩余 79% 往往可以追溯到"存在一次检查，但它静默失败或从未运行"。验证是承重角色。

PwC 报告称（CrewAI 部署，2025），添加结构化验证循环将准确率从 10% 提升到 70%。一个角色带来 7 倍收益。

### Critic vs Verifier

- Critic 是一个审阅产物质量的 LLM。主观的。可能被看似合理的文字欺骗。
- Verifier 是一个在产物上运行的确定性程序。客观的。给出通过/失败及证据。

两者都要用。Critic 能抓住 verifier 无法表达的口感问题。Verifier 能抓住 critic 看不到的 bug，因为它们只在运行时暴露。

### 反模式

系统中每个角色都是 LLM，每个角色的输出都是"看起来没问题"。这是 MAST 的经典失败模式。至少添加一个由代码而非 LLM 决定通过/失败的 verifier。

### 框架映射

- **CrewAI** — `Agent(role, goal, backstory)` 是角色特化的教科书级接口。
- **LangGraph** — 节点可以拥有特化提示词；边（edges）强制执行流水线。
- **AutoGen** — 具有角色特化的 ConversableAgents，在 GroupChat 中使用单字名称。
- **OpenAI Agents SDK** — 在角色特化的 Agent 之间使用交接工具（handoff tools）。

## 构建

`code/main.py` 实现了一个构建简单 Python 函数的 4 角色流水线：

- **Planner** 产出规格说明。
- **Executor** 生成代码字符串。
- **Critic**（LLM 模拟）标记明显问题。
- **Verifier** 在沙箱（`exec`）中运行生成的代码，并用测试用例检查。

演示运行两次：第一次 executor 产出正确代码（critic + verifier 均通过），第二次 executor 产出偏离规格的代码（critic 因为看起来合理而漏掉 bug，verifier 因为测试失败而抓住它）。

运行：

```
python3 code/main.py
```

## 使用

`outputs/skill-role-designer.md` 接收一个任务并产出角色名单（3-5 个角色）、每个角色的输入/输出 schema，以及验证检查。在将智能体接入框架之前使用它。

## 交付

检查清单：

- **至少一个确定性验证器。** 永远不要全是 LLM。
- **每个角色的显式 I/O schema。** Planner 返回规格说明，不是散文；执行者读取该 schema。
- **交流式去幻觉。** 执行者在信息缺失时必须询问规划者；绝不编造。
- **Critic/Verifier 的执行顺序。** 先运行 critic（便宜，抓住设计问题），再运行 verifier（慢，抓住 bug）。
- **循环预算。** 在升级给人类之前，critic-执行者 的修订轮次最多 2 轮。

## 练习

1. 运行 `code/main.py`，观察 verifier 如何抓住 critic 漏掉的 bug。添加一个静态分析检查（统计 `return` 出现次数）作为额外的 verifier。它能抓住什么运行时测试遗漏的问题？
2. 添加第 5 个角色："需求分析师"，将用户意愿转换为 planner 可用的规格说明。应该向它发起哪些交流式去幻觉请求？
3. 阅读 MetaGPT 第 3 节（"Agents"）。列出 MetaGPT 5 个角色的输入/输出 schema。
4. 阅读 ChatDev 的聊天链图示（arXiv:2307.07924 图 3）。找出交流式去幻觉在哪里打断了一个否则会无限循环的回路。
5. PwC 的 7 倍准确率提升来自验证循环。假设三种任务，添加 verifier 不会有帮助——即确定性正确性检查不可能或成本过高。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Role specialization | "不同的智能体，不同的工作" | 为 planner/executor/critic/verifier 角色调优的独立系统提示词。 |
| SOP pattern | "编码的标准操作流程" | MetaGPT 的框架：每个角色的严格 I/O schema 将团队变成流水线。 |
| Communicative dehallucination | "先问再编" | ChatDev 模式：执行者在缺失细节时询问规划者，而不是编造一个。 |
| Critic | "LLM 审查员" | 主观的、有观点的审查者。抓住口感问题。可能被看似合理的文字欺骗。 |
| Verifier | "确定性检查" | 基于代码的通过/失败。测试运行器、类型检查器、schema 验证器。无法被欺骗。 |
| Verification gap | "没人检查" | 占 MAST 失败的 21.3%。答案在未经能抓住 bug 的检查的情况下被交付。 |
| Revision loop | "Critic 打回重写" | Critic 的拒绝触发带反馈的执行者重新运行。需要设定预算。 |
| All-LLM anti-pattern | "看起来没问题" | 每个角色都是 LLM，没有确定性检查。MAST 的经典失败模式。 |

## 延伸阅读

- [Hong et al. — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — SOP 即角色提示词的参考论文
- [Qian et al. — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — 聊天链 + 交流式去幻觉
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分类法；验证缺口占失败的 21.3%
- [CrewAI docs — Agent roles](https://docs.crewai.com/en/introduction) — 生产级角色规范接口
