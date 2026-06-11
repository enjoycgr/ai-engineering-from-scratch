type LLMResponse = {
  output: string;
  tokens: number;
  calls: number;
};

type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

async function fakeLLMCall(
  systemPrompt: string,
  userMessage: string
): Promise<LLMResponse> {
  const inputLength = systemPrompt.length + userMessage.length;
  const simulatedTokens = Math.floor(inputLength / 4) + 500;

  await new Promise((resolve) => setTimeout(resolve, 50));

  return {
    output: `[Response to: ${userMessage.slice(0, 80)}...]`,
    tokens: simulatedTokens,
    calls: Math.floor(Math.random() * 5) + 1,
  };
}

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

function createSpecialist(
  name: string,
  systemPrompt: string
): SpecialistAgent {
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

async function multiAgentPipeline(task: string): Promise<AgentResult> {
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
    content: messages
      .map((m) => `[${m.from} -> ${m.to}]: ${m.content}`)
      .join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}

async function multiAgentFanOut(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const [researchResult, requirementsResult] = await Promise.all([
    researcher.run(`Research technical approach for: ${task}`),
    createSpecialist(
      "requirements",
      "You are a requirements analyst. Extract functional and non-functional requirements. Be exhaustive."
    ).run(`Analyze requirements for: ${task}`),
  ]);

  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  messages.push({
    from: "requirements",
    to: "coder",
    content: requirementsResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed + requirementsResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls + requirementsResult.toolCalls;

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

  const reviewResult = await reviewer.run(codeResult.content);
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages
      .map((m) => `[${m.from} -> ${m.to}]: ${m.content}`)
      .join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}

async function main() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== 单智能体方法 ===\n");
  const singleResult = await singleAgentApproach(task);
  console.log(`Token 用量: ${singleResult.tokensUsed}`);
  console.log(`工具调用: ${singleResult.toolCalls}`);
  console.log(`上下文: 所有内容在一个窗口中\n`);

  console.log("=== 多智能体流水线 ===\n");
  const pipelineResult = await multiAgentPipeline(task);
  console.log(`Token 用量: ${pipelineResult.tokensUsed}`);
  console.log(`工具调用: ${pipelineResult.toolCalls}`);
  console.log(`上下文: 每个智能体只获取它需要的内容\n`);

  console.log("=== 多智能体扇出 ===\n");
  const fanOutResult = await multiAgentFanOut(task);
  console.log(`Token 用量: ${fanOutResult.tokensUsed}`);
  console.log(`工具调用: ${fanOutResult.toolCalls}`);
  console.log(`上下文: 研究者 + 需求分析并行运行\n`);

  console.log("=== 对比 ===\n");
  console.log(
    `单智能体上下文污染: 所有 ${singleResult.tokensUsed} 个 token 在一个窗口中`
  );
  console.log(
    `多智能体隔离: ${pipelineResult.tokensUsed} 个总 token 分布在 3 个隔离窗口中`
  );
  console.log(
    `扇出并行: 研究与需求分析同时运行`
  );
}

main();
