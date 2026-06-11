# CrewAI: 基于角色的 Crews 与 Flows

> CrewAI 是 2026 年基于角色的 multi-agent framework (多智能体框架)。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自主、基于角色的协作）和 Flows（事件驱动、确定性）。文档直言："对于任何生产级应用，从 Flow 开始。"

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）以及每个原语拥有什么。
- 区分 Sequential（顺序）、Hierarchical（层级）和计划中的 Consensus（共识）process；为每个工作负载选择一种。
- 区分 Crews（自主、基于角色）与 Flows（事件驱动、确定性），并解释文档的生产建议。
- 通过 `@tool` decorator 和 `BaseTool` subclass 接入工具；推理 structured outputs (结构化输出) vs free text (自由文本)。
- 说出四种 CrewAI memory types (记忆类型) 以及每种何时有效。
- 实现一个标准库三智能体 crew（researcher、writer、editor），生成一份 brief (简报)。
- 识别三种 CrewAI 失效模式：prompt-bloat (提示膨胀)、manager-LLM tax (管理者 LLM 税费)、brittle handoffs (脆弱交接)。

## 问题背景

采用 multi-agent frameworks 的团队都会撞到同一堵墙。"自主协作" 在 demo 中听起来很棒。然后客户提交了一个 bug，而你需要 deterministic replay (确定性重放)。或者财务询问一个 LLM-routed crew 每次运行的成本。或者值班人员需要知道凌晨 3 点哪个 agent 卡住了。

自由形式的 LLM-routed crews 无法干净地回答这些问题。纯 DAGs 可以回答所有问题，但失去了头脑风暴 agent 所需的探索性形态。

CrewAI 的分割诚实地面对了这种权衡。Crews 用于协作、基于角色、探索性的工作。Flows 用于事件驱动、代码拥有、可审计的生产环境。同一个框架，两种形态，按场景选择。

## 核心概念

### 四个原语

CrewAI 的表面很小。记住这个，其余都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。Backstory (背景故事) 是 load-bearing (承重性的)。它塑造语气、判断、agent 何时停止。Tools 是 agent 可以调用的函数（更多见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。一个可复用的工作单元。`expected_output` 是 contract (契约)。`context` 列出上游任务，它们的输出会被传入。`output_pydantic` 强制要求 structured shape (结构化形状)。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** Execution strategy (执行策略)。Sequential、Hierarchical、Consensus（计划中）。选择运行的形态。

Agents 不能直接看到彼此。Tasks 引用 agents。Crew 对 tasks 排序。Process 决定谁选择下一个 task。这就是完整的心智模型。

> **Validated against** CrewAI 0.86 (2026-05)。新版本可能会重命名或合并 process types；依赖特定形状前请检查 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** Tasks 按声明顺序运行。Task N 的输出可作为 `context` 提供给 Task N+1。成本最低。最可预测。当顺序固定时使用。
- **Hierarchical。** 一个 manager Agent (单独的 LLM 调用) 在 specialists (专家) 之间路由。CrewAI 从你的 `manager_llm` 配置或默认值中生成 manager。Manager 每轮选择下一个 task，可以拒绝或重新路由。当你有四个或更多 specialists 且顺序确实依赖于先前输出时使用。
- **Consensus。** 计划中，目前未在公共 API 中实现。文档预留了该名称用于未来的基于投票的 process。今天不要依赖它。

Hierarchical 在每轮 specialist 调用之上增加了一个 per-round LLM 调用（manager）。在五次运行的流程中，token 成本可能增加两倍。只在需要路由时才支付。

### Crews vs Flows

这是 2026 年文档首先提出的框架。

- **Crew。** LLM 驱动的自主性。Framework 在运行时选择形态。适用于：研究、头脑风暴、初稿、路径本身就是答案的任何地方。难以重放。难以测试。原型设计便宜。
- **Flow。** 事件驱动的、你拥有的图。`@start` 标记入口。`@listen(topic)` 标记当另一个步骤发出该 topic 时触发的步骤。每个步骤都是纯 Python（可以在内部调用 Crew）。适用于：生产。可观察。可测试。确定性。

文档的 2026 生产建议：从 Flow 开始。当自主性值得其成本时，在 Flow 步骤内部以 `Crew.kickoff()` 调用的形式嵌入 Crews。Flow 给你 audit trail (审计轨迹)，Crew 给你 exploration (探索能力)。组合使用，不要二选一。

### 工具集成

三种方式给 Agent 一个工具。选择最简单的适合你的方式。

1. **`@tool` decorator。** 纯函数变成 tools。Signature 是 schema；docstring 是 LLM 看到的 description。最适合一次性 helper。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass。** 基于类的工具，带有显式 args schema、async support、retries。当工具有 state（client、cache）或需要 structured args 时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits。** CrewAI 附带 first-party adapters：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一次导入即可接入。

Structured outputs 使用 Pydantic。在 Task 上传递 `output_pydantic=MyModel`。CrewAI 根据模型验证 LLM 响应，要么 coerce (强制转换) 要么 retry (重试)。将其与 tight `expected_output` 字符串配对。Free-text outputs 对草稿没问题；structured outputs 是下游 Flows 可以消费的东西。

### Memory hooks (记忆钩子)

CrewAI 开箱即用提供四种 memory types。它们可以组合：一个 Crew 可以同时启用全部四种。

> **Validated against** CrewAI 0.86 (2026-05)。近期版本将所有内容路由通过统一的 `Memory` 系统，该系统包装了这四个存储。下面的概念模型仍然成立，但公共类表面可能 collapsing 为单个 `Memory` 入口点；请检查 [CrewAI memory docs](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **Short-term (短期)。** 单次运行内的对话缓冲区。运行结束时清除。
- **Long-term (长期)。** 跨运行持久化。存储在 vector DB 中（默认 Chroma，可替换）。通过 similarity (相似性) 检索当前 task。
- **Entity (实体)。** 每个实体的事实。"客户 X 使用企业版计划。" 按实体 keyed (键控)，而非按相似性。跨运行存活。
- **Contextual (上下文)。** 组装时检索。在 Agent 需要它的时刻拉取相关 memory，而非预加载。

在 Crew 上通过 `memory=True` 或 per-type config 启用。由你配置的 embeddings provider 支持（默认 OpenAI，可替换为 local）。Memory 是 CrewAI 相对于更薄框架的优势领域之一；纯 LangGraph 要求你自己接入每一种。

### CrewAI 适合的场景

- 三到六个带有命名角色的 agents 和协作工作流。起草、审查、规划、头脑风暴。
- Routing，其中 LLM 对下一步的判断是价值的一部分（Hierarchical）。
- 任何团队更愿意阅读 `role + goal + backstory` 而不是图定义的地方。

### CrewAI 不适合的场景

- 严格排序的确定性 DAGs。使用 LangGraph (Lesson 13)。图的形状是正确的抽象；CrewAI 的角色框架是 friction (阻力)。
- Sub-second latency budgets。Hierarchical 增加了往返。即使是 Sequential 也会序列化包含 backstories 和先前输出的 prompts。
- 单智能体循环。跳过框架；一个 agent loop (Lesson 1) 加上一个工具注册表更短。

Lesson 17 (Agent Framework Tradeoffs) 以矩阵形式展示了这一点。简而言之：CrewAI 位于 "协作、基于角色" 的角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。Star 数：见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（截至 2026-05 的快照）。AWS Bedrock 集成有文档记录；供应商基准测试报告称在 QA 工作负载上相比 LangGraph 有显著加速，但 methodology（数据集、硬件、评估指标）未公布，因此将框架供应商数字视为方向性参考即可。

### 这种模式的失效场景

- **来自 backstories 的 prompt-bloat。** 每个 agent 2000 词的 backstory 和五个 agent 的 crew 会在第一个 tool call 之前烧完 context budget。将 backstories 控制在 200 词以内。跨 agents 复用短语；不要重复 house style 五次。
- **Manager-LLM token tax。** Hierarchical process 在每个 specialist 调用之前增加一个 manager LLM 调用。在一个五 task crew 上，这是六次 LLM 调用而不是五次，而且 manager 调用携带完整的 task list 加上先前输出。除非路由依赖输出，否则切换到 Sequential。
- **Brittle handoffs (脆弱交接)。** Task N 的 `expected_output` 是 "an outline"。Task N+1 将其作为 `context` 读取并尝试解析三个 sections。LLM 产生了四个。下游 Agent 即兴发挥。通过在 Task N 上使用 `output_pydantic` 修复，这样 Task N+1 读取的是 typed object (类型化对象)，而非 free text。
- **Crew-as-prod。** 将自由形式的 Crew 直接投入生产而没有 Flow wrapper。输出可变性高；重放不可能；值班人员无法将 bad run 与 good run 进行 diff。用 Flow 包装。

## 动手实现

`code/main.py` 实现了两种形态的标准库版本以及一个三智能体 crew。

形态：

- `Agent`、`Task` dataclasses 匹配 CrewAI 的表面。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行 tasks，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 添加一个 manager Agent，每轮选择下一个 specialist，在 "done" 时停止。
- `Flow`，带有 `@start` 和 `@listen(topic)` decorators、一个 tiny event loop (微型事件循环) 和一个 trace。
- `tool(name)` decorator 镜像 CrewAI 的 `@tool` 形态。
- `Memory`，带有 `short_term`、`long_term`、`entity` stores；mocked similarity 使用 numpy。
- Mock LLM responses 是基于 role 加 input prefix 的 hardcoded strings (硬编码字符串)。没有网络。确定性的。

具体 demo：researcher、writer、editor crew 生成一份关于 "agent engineering 2026" 的 brief。Researcher 拉取（模拟的）来源。Writer 起草。Editor 精简。同一个 crew 通过 Flow 运行以展示确定性形态。

运行方式：

```bash
python3 code/main.py
```

Trace 覆盖：通过 `context` 传递输出的 sequential crew、带 manager picks 的 hierarchical crew（researcher、writer、editor，然后 "done"）、通过显式 topics（`researched`、`drafted`、`edited`）运行相同三个步骤的 Flow、通过 `@tool` 路由的 tool calls，以及跨两次 kickoffs 存活的 long-term memory。

Crew trace 是 fluid (流动的)；manager 原则上可以重新排序。Flow trace 是固定的。这种选择就是课程的核心。

## 如何使用

- **CrewAI Flow** 用于生产。即使 Flow 只是一步调用 `Crew.kickoff()`。Flow 给你 audit boundary (审计边界)。
- **CrewAI Crew (Sequential)** 用于顺序清晰的协作工作，尤其是初稿和审查循环。
- **CrewAI Crew (Hierarchical)** 当路由依赖于输出且你有四个或更多 specialists 时。
- **LangGraph** (Lesson 13) 用于显式状态机、durable resume、严格排序。
- **AutoGen v0.4** (Lesson 14) 用于 actor-model concurrency 和 fault isolation。
- **OpenAI Agents SDK** (Lesson 16) 用于 OpenAI 优先的产品，带有 handoffs 和 guardrails。
- **Claude Agent SDK** (Lesson 17) 用于 Claude 优先的产品，带有 subagents 和 session store。

## 输出产物

`outputs/skill-crew-or-flow.md` 为任务选择 Crew vs Flow 并搭建最小实现。对没有 backstory 的 Crew、没有 explicit topics 的 Flow、 specialist 少于三个的 Hierarchical 进行硬性拒绝。

## 陷阱

- **Backstory 作为 flavor (风味)。** 它塑造输出。为每个 agent 测试三个变体；variance (方差) 是真实的。选择一种，冻结它。
- **跳过 `expected_output`。** 没有每个 task 的契约，下游 tasks 只能拿到 LLM 产生的任何东西。Crew 运行；审计失败。
- **Memory always-on。** Long-term 每次运行都写入。Vector DB 增长。Retrieval 变得嘈杂。将写入范围限定到事实持久化的 tasks。
- **Manager prompt drift。** Hierarchical 的 manager prompt 是隐式的。如果路由变得奇怪，在 verbose mode 下 dump 并阅读。
- **Crews 中的工具 side effects。** Crew 调用工具的次数可能超过预期。POST、DELETE、支付属于 Flow 步骤，绝不属于 Crew tool。

## 练习题

1. 将 Sequential crew 转换为 Flow。计算 variability (可变性) 下降的 touchpoints。注意 readability (可读性) 下降的地方。
2. 为 crew 添加 entity memory：关于客户的事实跨 kickoffs 持久化。验证 retrieval 拉取了正确的实体。
3. 实现一个 Hierarchical process，其中 manager 拒绝路由到 editor，直到 writer 的输出至少有 three paragraphs。Trace 重试过程。
4. 为一个（模拟的）web search 接入 `BaseTool` subclass。比较 trace 形状 vs `@tool` decorator 版本。
5. 为 editor task 添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让 writer task 一次输出 malformed JSON；验证 trace 中 CrewAI 的 retry 行为。
6. 阅读 CrewAI 的文档 intro。将玩具示例移植到真正的 `crewai` API。Stdlib 版本跳过了哪些保证？
7. 将 AgentOps 或 Langfuse (Lesson 24) 接入真实运行。你在 stdlib 版本中错过了哪些 traces？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|---|---|---|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "工作单元" | Description + expected output + assignee + optional structured output |
| Crew | "Agent 团队" | Agents + Tasks + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus (计划中) |
| Flow | "确定性工作流" | 事件驱动、代码拥有、可测试 |
| Backstory | "Persona prompt" | Agent 的语气和判断塑造器 |
| `@tool` | "函数工具" | 将函数变成 Agent 可调用的工具的 decorator |
| `BaseTool` | "类工具" | 带有 args schema、retries、async support 的基于类的工具 |
| Entity memory | "每个实体的事实" | 按客户/账户/问题 scoped (限定范围) 的记忆 |
| Long-term memory | "跨运行记忆" | 在 kickoffs 之间存活的向量支持的记忆 |
| Contextual memory | "即时检索" | 在 Agent 需要它的时刻拉取的记忆 |
| Manager LLM | "路由器 agent" | Hierarchical process 中选择下一个 task 的额外 LLM |
| `expected_output` | "Task 契约" | 告诉 Agent（和审计）应返回什么形状的字符串 |

## 延伸阅读

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): 概念和推荐的生产路径
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): 事件驱动形态、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools): `@tool`、`BaseTool`、内置 toolkits
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): short-term、long-term、entity、contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): 何时多智能体有帮助、何时没有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): 状态机替代方案
