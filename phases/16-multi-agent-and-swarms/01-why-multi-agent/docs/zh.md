# 为什么使用 Multi-Agent？

> 单个 agent 会撞墙。聪明的做法不是造一个更大的 agent，而是使用更多 agent。

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 分钟

## 学习目标

- 识别 single-agent ceiling（单智能体天花板）（上下文溢出、混合 expertise、顺序瓶颈），并解释何时拆分为多个 agent 是正确的选择
- 比较 orchestration patterns（编排模式）（pipeline、parallel fan-out、supervisor、hierarchical），并为给定的任务结构选择正确的模式
- 设计一个具有清晰角色边界、shared state（共享状态）和通信契约的 multi-agent（多智能体）系统
- 分析 multi-agent 复杂性（延迟、成本、调试难度）与单 agent 简单性之间的权衡

## 问题所在

你在 Phase 14 构建了一个单 agent。它能工作。它可以读取文件、运行命令、调用 API，并对结果进行推理。然后你让它处理一个真实的代码库：200 个文件、三种语言、依赖基础设施的测试，以及一个需要在编写代码前研究外部 API 的需求。

这个 agent 卡住了。不是因为 LLM 笨，而是因为任务超出了单个 agent 循环所能处理的范围。Context window（上下文窗口）被文件内容填满。Agent 忘记了 40 个 tool call（工具调用）之前读过的内容。它试图同时充当 researcher、coder 和 reviewer，结果三件事都做得不好。

这就是 single-agent ceiling（单智能体天花板）。每当任务需要以下能力时，你就会遇到它：

- **比一个窗口能容纳更多的上下文** —— 读取 50 个文件会超过 200k token
- **不同阶段需要不同的 expertise** —— 研究需要与代码生成不同的 prompting
- **可以并行处理的工作** —— 为什么要顺序读取三个文件，而不是同时读取？

## 核心概念

### Single-Agent Ceiling（单智能体天花板）

单个 agent 是一个循环、一个 context window、一个 system prompt（系统提示词）。想象一下：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

有三样东西会崩溃：

1. **Context saturation（上下文饱和）** —— tool result 不断堆积。到第 30 轮时，agent 已经消耗了 150k token 的文件内容、命令输出和先前推理。第 5 轮的关键细节丢失了。

2. **Role confusion（角色混淆）** —— 一个写着"你是 researcher、coder、reviewer 和 tester"的 system prompt 会产生一个半吊子研究、半吊子编码、永远完不成 review 的 agent。

3. **Sequential bottleneck（顺序瓶颈）** —— agent 先读文件 A，再读文件 B，然后读文件 C。三个串行的 LLM 调用。三个串行的 tool 执行。没有并行性。

### Multi-Agent 解决方案

拆分工作。给每个 agent 一个任务、一个 context window，以及一个针对该任务调优的 system prompt：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每个 agent 拥有：
- 一个专注的 system prompt（"你是一个 code reviewer。你的唯一工作是发现 bug。"）
- 自己的 context window（不会被其他 agent 的工作污染）
- 清晰的输入/输出契约（接收研究笔记，输出代码）

### 实际使用这些的系统

**Claude Code subagents** —— 当 Claude Code 用 `Task` 生成 subagent 时，它会创建一个带有限定任务的子 agent。父 agent 保持其上下文干净。子 agent 执行专注的工作并返回摘要。

**Devin** —— 运行 planner agent、coder agent 和 browser agent。Planner 将工作分解为步骤。Coder 编写代码。Browser 研究文档。每个都有独立的上下文。

**Multi-agent coding teams (SWE-bench)** —— SWE-bench 上表现最好的系统使用 researcher 读取代码库，planner 设计修复方案，coder 实现它。单 agent 系统得分更低。

**ChatGPT Deep Research** —— 并行生成多个 search agent，每个探索不同的角度，然后综合结果。

### 光谱

Multi-agent 不是二元的。它是一个光谱：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** —— 一个循环，一个 prompt。适合简单任务。

**Subagents** —— 父 agent 为专注的子任务生成子 agent。父 agent 维护计划。子 agent 回报。这就是 Claude Code 的做法。

**Pipeline** —— agent 按顺序运行。Agent A 的输出成为 Agent B 的输入。适合分阶段工作流：research -> code -> review -> test。

**Team** —— agent 通过 shared message bus（共享消息总线）并行运行。每个都有角色。Orchestrator 协调。当需要不同技能同时工作时很有用。

**Swarm** —— 许多相同或近相同的 agent 带有 shared state。没有固定的 orchestrator。Agent 从队列中拾取工作。适合高吞吐量的并行任务。

### 四种 Multi-Agent 模式

#### 模式 1：Pipeline（流水线）

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每个 agent 转换数据并向前传递。简单易懂。一个阶段的失败会阻塞其余阶段。

#### 模式 2：Fan-out / Fan-in（扇出 / 扇入）

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

将工作拆分到并行 agent，然后合并结果。适合可分解为独立子任务的任务。

#### 模式 3：Orchestrator-Worker（编排器-工作者）

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一个智能的 orchestrator 决定做什么，委派给 worker，并综合结果。Orchestrator 本身是一个带有生成 worker 工具的 agent。

#### 模式 4：Peer Swarm（对等集群）

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

没有中央 orchestrator。Agent 点对点通信。决策从交互中涌现。更难调试，但可扩展到许多 agent。

### 何时不使用 Multi-Agent

Multi-agent 增加了复杂性。Agent 之间的每条消息都是一个潜在的故障点。调试从"读取一个对话"变成了"追踪五个 agent 之间的消息"。

**在以下情况保持单 agent：**
- 任务适合一个 context window（工作数据少于 ~100k token）
- 不同阶段不需要不同的 system prompt
- 顺序执行已经足够快
- 任务足够简单，拆分它增加的开销大于价值

**复杂性成本：**
- 每个 agent 边界都是一个有损压缩步骤：agent A 的完整上下文被压缩成一条给 agent B 的消息
- 协调逻辑（谁做什么、何时、以什么顺序）本身就是 bug 的来源
- 延迟增加：N 个 agent 意味着至少 N 个串行 LLM 调用，如果它们需要来回通信则更多
- 成本倍增：每个 agent 独立消耗 token

经验法则：如果任务需要少于 20 个 tool call 并且适合 100k token，保持单 agent。

## 动手实现

### 步骤 1：过载的单 Agent

这里是一个试图做所有事情的单 agent。它有一个庞大的 system prompt 和一个包含 research、code 和 review 的 context window：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

这种方法的问题：
- Context window 随每个阶段增长。到 review 步骤时，它包含 research note AND code AND prior reasoning。
- System prompt 是通用的。无法为每个阶段调优。
- 没有并行执行。

### 步骤 2：专家 Agent

现在拆分它。每个 agent 得到一个任务：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每个 specialist 有一个专注的 prompt。每个得到一个只包含它需要的内容的干净 context window。

### 步骤 3：通过消息协调

用显式的 message passing（消息传递）将 specialist 连接起来：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每个 agent 只接收发给它的消息。没有上下文污染。Researcher 的 50k token 文档阅读永远不会进入 reviewer 的上下文。

### 步骤 4：对比

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

Multi-agent 版本使用更多的总 token（三个 agent，三个独立的 LLM 调用），但每个 agent 的上下文保持干净。每个阶段的质量提高，因为 system prompt 是 specialized 的。

## 使用它

本节课产生了一个可复用的 prompt，用于决定何时使用 multi-agent。见 `outputs/prompt-multi-agent-decision.md`。

## 练习

1. 添加第四个 specialist：一个 "tester" agent，它从 coder 接收代码，从 reviewer 接收 review feedback，然后编写测试
2. 修改 pipeline，使 reviewer 可以将 feedback 发回给 coder 进行修订循环（最多 2 轮）
3. 将顺序 pipeline 转换为 fan-out：并行运行 researcher 和 "requirements analyzer" agent，然后在传给 coder 之前合并它们的输出

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Swarm | "AI agent 的蜂巢思维" | 一组带有 shared state 且没有固定领导者的对等 agent。行为从局部交互中涌现。 |
| Orchestrator | "老板 agent" | 一个工具包括生成和管理其他 agent 的 agent。它计划和委派，但可能不执行实际工作。 |
| Coordinator | "交通警察" | 一个非 agent 组件（通常只是代码，不是 LLM），根据规则在 agent 之间路由消息。 |
| Consensus | "Agent 们达成一致" | 多个 agent 必须在继续之前达成一致的协议。用于需要解决冲突输出时。 |
| Emergent behavior | "Agent 自己搞明白了" | 从 agent 交互中出现的系统级模式，但没有被显式编程。可能是有益的或有害的。 |
| Fan-out / fan-in | "Agent 的 Map-reduce" | 将任务拆分到并行 agent（fan-out），然后合并它们的结果（fan-in）。 |
| Message passing | "Agent 互相交谈" | Agent 之间的通信机制：从一个 agent 发送到另一个 agent 的结构化数据，替代共享的 context window。 |

## 延伸阅读

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - multi-agent patterns 综述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft 的 multi-agent 对话框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何用 Task 委派
- [CrewAI documentation](https://docs.crewai.com/) - 基于角色的 multi-agent 框架
