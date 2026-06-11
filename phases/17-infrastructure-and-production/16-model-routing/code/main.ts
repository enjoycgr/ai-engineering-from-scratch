/**
 * Model routing — TypeScript 移植版 + 基于规则的路由器。
 *
 * 两部分：
 *   1. ModelRouter：基于 (model catalog, request signals) 的规则选择器。
 *      每条规则按 capability fit 对候选模型打分，然后按调用方提供的策略
 *      权衡 latency vs cost vs capability。匹配 docs/en.md 中的四种信号
 *      （task class、prompt length、与 hard set 的 similarity、self-confidence）。
 *   2. 与 main.py 一致的成本/质量模拟器：NO_ROUTE / PRE_ROUTE /
 *      CASCADE 模式在混合难度工作负载上。
 *
 * 引用：
 *   - RouteLLM (LMSYS): https://github.com/lm-sys/RouteLLM
 *   - OpenRouter recommendation/routing primitives: https://openrouter.ai/
 *   - LiteLLM router config with fallback + cost-routing（docs 中引用）
 *
 * 在 Node 20+ stdlib 上运行。无 npm 依赖。
 */

// -- Pricing (2026-04 approximations) -------------------------------------

const CHEAP_INPUT = 0.25;
const CHEAP_OUTPUT = 1.0;
const FRONTIER_INPUT = 3.0;
const FRONTIER_OUTPUT = 15.0;

// -- Model catalog + router primitive --------------------------------------

type Capability =
  | "chat"
  | "code"
  | "math"
  | "vision"
  | "long-context"
  | "tool-use";

type Model = {
  id: string;
  // 每百万 token。
  inputPrice: number;
  outputPrice: number;
  // P50 首 token 延迟（ms）。
  latencyMs: number;
  // 最大上下文长度（token）。
  contextWindow: number;
  // Capability bag。路由器按此打分。
  capabilities: Set<Capability>;
  // 按 docs 粗略映射的 0–1 主观质量。
  qualityFloor: number;
};

const CATALOG: Model[] = [
  {
    id: "haiku-class",
    inputPrice: CHEAP_INPUT,
    outputPrice: CHEAP_OUTPUT,
    latencyMs: 250,
    contextWindow: 200_000,
    capabilities: new Set<Capability>(["chat", "tool-use"]),
    qualityFloor: 0.75,
  },
  {
    id: "sonnet-class",
    inputPrice: 1.0,
    outputPrice: 5.0,
    latencyMs: 450,
    contextWindow: 200_000,
    capabilities: new Set<Capability>([
      "chat",
      "code",
      "tool-use",
      "long-context",
    ]),
    qualityFloor: 0.9,
  },
  {
    id: "frontier",
    inputPrice: FRONTIER_INPUT,
    outputPrice: FRONTIER_OUTPUT,
    latencyMs: 800,
    contextWindow: 1_000_000,
    capabilities: new Set<Capability>([
      "chat",
      "code",
      "math",
      "vision",
      "tool-use",
      "long-context",
    ]),
    qualityFloor: 1.0,
  },
];

type RouteSignals = {
  // Task class derived from a small upstream classifier.
  taskClass: "simple" | "medium" | "hard";
  // Approximate prompt token count.
  promptTokens: number;
  // 0–1 cosine similarity to a curated known-hard set.
  hardSetSimilarity: number;
  // Required capabilities for this request.
  required: Capability[];
};

type RoutePolicy = {
  // Weights sum to 1; how much we care about each axis.
  weightCost: number;
  weightLatency: number;
  weightCapability: number;
  // Quality floor any chosen model must clear.
  minQuality: number;
};

type RouteDecision = {
  model: Model;
  estCost: number;
  reasoning: string;
};

class ModelRouter {
  private readonly catalog: readonly Model[];
  private readonly hardSetThreshold: number;

  constructor(catalog: readonly Model[], hardSetThreshold = 0.88) {
    this.catalog = catalog;
    this.hardSetThreshold = hardSetThreshold;
  }

  // 估算请求在模型上的 blended cost。假设输出 200 token，
  // 除非调用方在其他地方传入真实输出估算。
  estCost(model: Model, promptTokens: number, outputTokens = 200): number {
    return (
      (promptTokens / 1e6) * model.inputPrice +
      (outputTokens / 1e6) * model.outputPrice
    );
  }

  // 将 catalog 过滤为满足以下条件的模型：
  //  (a) 覆盖所有必需 capability，
  //  (b) 提示在其上下文窗口内，
  //  (c) 达到策略质量底线。
  candidates(signals: RouteSignals, policy: RoutePolicy): Model[] {
    return this.catalog.filter((m) => {
      for (const c of signals.required) if (!m.capabilities.has(c)) return false;
      if (signals.promptTokens > m.contextWindow) return false;
      if (m.qualityFloor < policy.minQuality) return false;
      return true;
    });
  }

  // 加权选择：越低 cost / 越低 latency / 越高 capability fit 越好。
  // 'hard set' similarity 短路到 frontier（匹配 docs 规则）。
  pick(signals: RouteSignals, policy: RoutePolicy): RouteDecision {
    if (signals.hardSetSimilarity >= this.hardSetThreshold) {
      const frontier = this.catalog.find((m) => m.id === "frontier");
      if (frontier) {
        return {
          model: frontier,
          estCost: this.estCost(frontier, signals.promptTokens),
          reasoning: `hard-set similarity ${signals.hardSetSimilarity.toFixed(2)} >= ${this.hardSetThreshold} — pinned to frontier`,
        };
      }
    }

    const cands = this.candidates(signals, policy);
    if (cands.length === 0) {
      throw new Error("no candidate model clears policy + required caps");
    }
    // Normalise for fair weighting.
    const costs = cands.map((m) => this.estCost(m, signals.promptTokens));
    const latencies = cands.map((m) => m.latencyMs);
    const caps = cands.map((m) => m.capabilities.size);
    const maxCost = Math.max(...costs);
    const maxLat = Math.max(...latencies);
    const maxCap = Math.max(...caps);

    let bestIdx = 0;
    let bestScore = -Infinity;
    let bestReason = "";
    for (let i = 0; i < cands.length; i++) {
      const costScore = 1 - costs[i] / (maxCost || 1);
      const latScore = 1 - latencies[i] / (maxLat || 1);
      const capScore = caps[i] / (maxCap || 1);
      const score =
        policy.weightCost * costScore +
        policy.weightLatency * latScore +
        policy.weightCapability * capScore;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
        bestReason =
          `cost=${costScore.toFixed(2)} latency=${latScore.toFixed(2)} cap=${capScore.toFixed(2)} ` +
          `weighted=${score.toFixed(3)}`;
      }
    }

    return {
      model: cands[bestIdx],
      estCost: costs[bestIdx],
      reasoning: bestReason,
    };
  }
}

// -- Workload + simulator (matches main.py) --------------------------------

type Difficulty = "simple" | "medium" | "hard";
type Query = {
  difficulty: Difficulty;
  promptTokens: number;
  outputTokens: number;
};

function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randint(rng: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

function makeWorkload(n = 1000, seed = 7): Query[] {
  const rng = makeRng(seed);
  const reqs: Query[] = [];
  for (let i = 0; i < n; i++) {
    const p = rng();
    if (p < 0.6) {
      reqs.push({
        difficulty: "simple",
        promptTokens: randint(rng, 200, 1000),
        outputTokens: randint(rng, 50, 200),
      });
    } else if (p < 0.9) {
      reqs.push({
        difficulty: "medium",
        promptTokens: randint(rng, 800, 3000),
        outputTokens: randint(rng, 100, 400),
      });
    } else {
      reqs.push({
        difficulty: "hard",
        promptTokens: randint(rng, 2000, 8000),
        outputTokens: randint(rng, 200, 1500),
      });
    }
  }
  return reqs;
}

function costOf(route: "cheap" | "frontier", q: Query): number {
  if (route === "cheap") {
    return (
      (q.promptTokens / 1e6) * CHEAP_INPUT +
      (q.outputTokens / 1e6) * CHEAP_OUTPUT
    );
  }
  return (
    (q.promptTokens / 1e6) * FRONTIER_INPUT +
    (q.outputTokens / 1e6) * FRONTIER_OUTPUT
  );
}

function quality(route: "cheap" | "frontier", q: Query): number {
  if (route === "frontier") return 1.0;
  return { simple: 0.99, medium: 0.92, hard: 0.75 }[q.difficulty];
}

type SimRow = {
  pattern: string;
  cost: number;
  meanQuality: number;
  escalated: number;
};

function simulate(pattern: string, reqs: readonly Query[]): SimRow {
  let totalCost = 0;
  let totalQ = 0;
  let escalated = 0;
  const rng = makeRng(11);

  for (const q of reqs) {
    if (pattern === "NO_ROUTE") {
      totalCost += costOf("frontier", q);
      totalQ += 1.0;
    } else if (pattern === "PRE_ROUTE") {
      if (q.difficulty === "simple") {
        totalCost += costOf("cheap", q);
        totalQ += quality("cheap", q);
      } else {
        totalCost += costOf("frontier", q);
        totalQ += 1.0;
      }
    } else if (pattern === "CASCADE") {
      totalCost += costOf("cheap", q);
      const confident =
        q.difficulty === "simple" ||
        (q.difficulty === "medium" && rng() < 0.5);
      if (confident) {
        totalQ += quality("cheap", q);
      } else {
        escalated++;
        totalCost += costOf("frontier", q);
        totalQ += 1.0;
      }
    }
  }

  return {
    pattern,
    cost: totalCost,
    meanQuality: totalQ / reqs.length,
    escalated,
  };
}

function reportRow(row: SimRow, baseline: number): void {
  const save = ((baseline - row.cost) / baseline) * 100;
  console.log(
    `${row.pattern.padEnd(12)}  cost=$${row.cost.toFixed(2).padStart(7)}  ` +
      `save=${save.toFixed(1).padStart(5)}%  ` +
      `quality=${(row.meanQuality * 100).toFixed(1).padStart(5)}%  ` +
      `escalated=${String(row.escalated).padStart(4)}`,
  );
}

// -- Demos -----------------------------------------------------------------

function routerDemo(): void {
  console.log("--- Rule-based ModelRouter ---");
  const router = new ModelRouter(CATALOG);

  const balanced: RoutePolicy = {
    weightCost: 0.5,
    weightLatency: 0.2,
    weightCapability: 0.3,
    minQuality: 0.7,
  };
  const latencyFirst: RoutePolicy = {
    weightCost: 0.1,
    weightLatency: 0.7,
    weightCapability: 0.2,
    minQuality: 0.7,
  };

  const cases: { name: string; signals: RouteSignals; policy: RoutePolicy }[] = [
    {
      name: "FAQ-style short prompt (balanced policy)",
      signals: {
        taskClass: "simple",
        promptTokens: 400,
        hardSetSimilarity: 0.2,
        required: ["chat"],
      },
      policy: balanced,
    },
    {
      name: "code-gen with tool use (balanced)",
      signals: {
        taskClass: "medium",
        promptTokens: 2500,
        hardSetSimilarity: 0.4,
        required: ["chat", "code", "tool-use"],
      },
      policy: balanced,
    },
    {
      name: "math near known-hard set (auto-pin frontier)",
      signals: {
        taskClass: "hard",
        promptTokens: 1500,
        hardSetSimilarity: 0.92,
        required: ["chat", "math"],
      },
      policy: balanced,
    },
    {
      name: "long-context 800K tokens (frontier only fits)",
      signals: {
        taskClass: "hard",
        promptTokens: 800_000,
        hardSetSimilarity: 0.1,
        required: ["chat", "long-context"],
      },
      policy: balanced,
    },
    {
      name: "FAQ-style short prompt (latency-first)",
      signals: {
        taskClass: "simple",
        promptTokens: 300,
        hardSetSimilarity: 0.1,
        required: ["chat"],
      },
      policy: latencyFirst,
    },
  ];

  for (const c of cases) {
    const d = router.pick(c.signals, c.policy);
    console.log(`  ${c.name}`);
    console.log(
      `    → ${d.model.id}  est_cost=$${d.estCost.toFixed(5)}  reason=${d.reasoning}`,
    );
  }
}

function patternsDemo(): void {
  console.log("\n" + "=".repeat(80));
  console.log("MODEL ROUTING — three patterns, 1000 requests, mixed difficulty");
  console.log("=".repeat(80));
  const reqs = makeWorkload();
  const baseline = simulate("NO_ROUTE", reqs).cost;
  for (const p of ["NO_ROUTE", "PRE_ROUTE", "CASCADE"]) {
    reportRow(simulate(p, reqs), baseline);
  }
  console.log(
    "\nRead: PRE_ROUTE saves big when the classifier is accurate. CASCADE",
  );
  console.log(
    "guarantees quality floor but adds latency on escalated requests.",
  );
}

function main(): void {
  routerDemo();
  patternsDemo();
}

main();
