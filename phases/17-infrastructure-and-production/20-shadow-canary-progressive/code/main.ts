/**
 * Shadow + canary + progressive rollout — TypeScript 移植版 + policy engine。
 *
 * 三种策略：
 *   1. Shadow mode：将每个请求复制到候选模型；记录差异；
 *      绝不将候选输出返回给用户。在用户暴露前捕获成本/长度回退。
 *   2. Canary rollout：分阶段渐进式流量切换，带五个 LLM 专用 gates。
 *      任一 gate 违规时立即停止。
 *   3. Progressive policy：组合 shadow → canary → 100%，并带支持秒级回滚的 policy flag。
 *
 * 外加与 main.py 相同的 canary 模拟器（六个阶段、五个 gates、六种回退场景），
 * 以保持数字可复现。
 *
 * 引用：
 *   - Argo Rollouts（Kubernetes progressive delivery）
 *     https://argo-rollouts.readthedocs.io/
 *   - Flagger（progressive delivery operator）
 *     https://docs.flagger.app/
 *   - docs/en.md 中引用的非确定性 ~15% 运行间差异（GPU FP
 *     non-associativity + batch-size variance + sampling）。
 *
 * 在 Node 20+ stdlib 上运行。无 npm 依赖。
 */

// -- Baseline + gates ------------------------------------------------------

type Metrics = {
  latencyP99Ms: number;
  costPerReq: number;
  errorRate: number;
  outputLenP99: number;
  thumbsDownRate: number;
};

const BASELINE: Metrics = {
  latencyP99Ms: 900,
  costPerReq: 0.02,
  errorRate: 0.02,
  outputLenP99: 450,
  thumbsDownRate: 0.03,
};

// 高于基线即构成违规的乘数。设得足够高以
// 保持在 LLM 非确定性噪声底 (~15%，见 docs/en.md) 之上。
const GATES: Record<keyof Metrics, number> = {
  latencyP99Ms: 1.5,
  costPerReq: 1.2,
  errorRate: 2.0,
  outputLenP99: 1.4,
  thumbsDownRate: 1.5,
};

const STAGES = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0];

// -- Mulberry32 PRNG ------------------------------------------------------

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

function stageSeed(i: number): number {
  return 11 + i * 3;
}

// -- Regression injector --------------------------------------------------

type Regression = {
  latencyMult: number;
  costMult: number;
  errorMult: number;
  outputLenMult: number;
  thumbsDownMult: number;
};

const NO_REGRESSION: Regression = {
  latencyMult: 1,
  costMult: 1,
  errorMult: 1,
  outputLenMult: 1,
  thumbsDownMult: 1,
};

function measureStage(_stage: number, reg: Regression, seed: number): Metrics {
  const rng = makeRng(seed);
  // 噪声底即 docs/en.md 描述的非确定性：每次测量约 ±8%。
  const noise = (v: number): number => v * (0.92 + rng() * 0.16);
  return {
    latencyP99Ms: noise(BASELINE.latencyP99Ms * reg.latencyMult),
    costPerReq: noise(BASELINE.costPerReq * reg.costMult),
    errorRate: noise(BASELINE.errorRate * reg.errorMult),
    outputLenP99: noise(BASELINE.outputLenP99 * reg.outputLenMult),
    thumbsDownRate: noise(BASELINE.thumbsDownRate * reg.thumbsDownMult),
  };
}

function checkGates(metrics: Metrics): (keyof Metrics)[] {
  const breaches: (keyof Metrics)[] = [];
  for (const k of Object.keys(GATES) as (keyof Metrics)[]) {
    if (metrics[k] > BASELINE[k] * GATES[k]) breaches.push(k);
  }
  return breaches;
}

// -- Policy engine --------------------------------------------------------

type ShadowSample = {
  baselineCost: number;
  candidateCost: number;
  baselineLatencyMs: number;
  candidateLatencyMs: number;
};

type ShadowReport = {
  n: number;
  meanCostDeltaPct: number;
  meanLatencyDeltaPct: number;
  // 若 shadow 本身足以在 canary 前停止，则为 true。
  alert: boolean;
  reasons: string[];
};

function shadowEvaluate(samples: ShadowSample[]): ShadowReport {
  if (samples.length === 0) {
    return {
      n: 0,
      meanCostDeltaPct: 0,
      meanLatencyDeltaPct: 0,
      alert: false,
      reasons: [],
    };
  }
  let costDelta = 0;
  let latDelta = 0;
  let costN = 0;
  let latN = 0;
  for (const s of samples) {
    // 跳过非正基线行，防止单行零值将均值变成 Infinity/NaN 并破坏 gate 决策。
    if (s.baselineCost > 0) {
      costDelta += (s.candidateCost - s.baselineCost) / s.baselineCost;
      costN++;
    }
    if (s.baselineLatencyMs > 0) {
      latDelta += (s.candidateLatencyMs - s.baselineLatencyMs) / s.baselineLatencyMs;
      latN++;
    }
  }
  const meanCost = costN > 0 ? (costDelta / costN) * 100 : 0;
  const meanLat = latN > 0 ? (latDelta / latN) * 100 : 0;
  const reasons: string[] = [];
  if (meanCost > 30) reasons.push(`cost +${meanCost.toFixed(1)}% (>30%)`);
  if (meanLat > 50) reasons.push(`latency +${meanLat.toFixed(1)}% (>50%)`);
  return {
    n: samples.length,
    meanCostDeltaPct: meanCost,
    meanLatencyDeltaPct: meanLat,
    alert: reasons.length > 0,
    reasons,
  };
}

type CanaryDecision = {
  promoted: boolean;
  stagesAdvanced: number;
  breaches: (keyof Metrics)[];
};

function canaryRollout(reg: Regression): CanaryDecision {
  for (let i = 0; i < STAGES.length; i++) {
    const metrics = measureStage(STAGES[i], reg, stageSeed(i));
    const breaches = checkGates(metrics);
    if (breaches.length > 0) {
      return { promoted: false, stagesAdvanced: i, breaches };
    }
  }
  return { promoted: true, stagesAdvanced: STAGES.length, breaches: [] };
}

// PolicyEngine wraps a feature flag — flip pinnedModel from candidate back to
// baseline in O(1). Mirrors LaunchDarkly/Flagsmith/Unleash flag-flip rollback.
class PolicyEngine {
  private baselineDigest: string;
  private pinnedDigest: string;
  private rolloutPct = 0;

  constructor(initialDigest: string) {
    this.baselineDigest = initialDigest;
    this.pinnedDigest = initialDigest;
  }

  promote(candidateDigest: string, pct: number): void {
    this.pinnedDigest = candidateDigest;
    this.rolloutPct = pct;
  }

  // 常数时间回滚——runbook 中翻转的内容。恢复到构造时捕获的基线
  // （或最近的回滚覆盖）。
  rollback(baselineDigest?: string): void {
    if (baselineDigest !== undefined) this.baselineDigest = baselineDigest;
    this.pinnedDigest = this.baselineDigest;
    this.rolloutPct = 0;
  }

  pick(rng: () => number): { digest: string; chose: "baseline" | "candidate" } {
    return rng() < this.rolloutPct
      ? { digest: this.pinnedDigest, chose: "candidate" }
      : { digest: this.baselineDigest, chose: "baseline" };
  }
}

// -- Reporting ------------------------------------------------------------

function rolloutReport(name: string, reg: Regression): void {
  console.log(`\n${name}`);
  console.log(
    `Regression: latency=${reg.latencyMult}, cost=${reg.costMult}, error=${reg.errorMult}, len=${reg.outputLenMult}, thumbs=${reg.thumbsDownMult}`,
  );
  for (let i = 0; i < STAGES.length; i++) {
    const stage = STAGES[i];
    const metrics = measureStage(stage, reg, stageSeed(i));
    const breaches = checkGates(metrics);
    const status =
      breaches.length === 0 ? "PASS" : `HALT (${breaches.join(",")})`;
    const pct = Math.round(stage * 100);
    console.log(
      `  stage ${String(pct).padStart(3)}%  ` +
        `lat_p99=${metrics.latencyP99Ms.toFixed(0).padStart(5)}  ` +
        `cost=$${metrics.costPerReq.toFixed(4)}  ` +
        `err=${(metrics.errorRate * 100).toFixed(1).padStart(4)}%  ` +
        `thumbs_dn=${(metrics.thumbsDownRate * 100).toFixed(1).padStart(4)}%  ` +
        `${status}`,
    );
    if (breaches.length > 0) {
      console.log("  → ROLLBACK (policy flip, pinned model reverted)");
      return;
    }
  }
  console.log("  → PROMOTED to 100%");
}

// -- Demo ------------------------------------------------------------------

function shadowDemo(): void {
  console.log("--- Shadow-mode evaluation (zero user impact) ---");
  // Three scenarios: candidate roughly comparable, candidate cheaper, candidate
  // 40% more expensive (the docs' canonical bad scenario).
  const rng = makeRng(99);
  const mkSamples = (costMult: number, latMult: number): ShadowSample[] =>
    Array.from({ length: 200 }, () => ({
      baselineCost: 0.02 * (0.95 + rng() * 0.1),
      candidateCost: 0.02 * costMult * (0.95 + rng() * 0.1),
      baselineLatencyMs: 800 * (0.95 + rng() * 0.1),
      candidateLatencyMs: 800 * latMult * (0.95 + rng() * 0.1),
    }));

  const scenarios: { name: string; samples: ShadowSample[] }[] = [
    { name: "comparable candidate", samples: mkSamples(1.05, 1.02) },
    { name: "candidate 20% cheaper", samples: mkSamples(0.8, 0.95) },
    { name: "candidate 40% more expensive (rollback case)", samples: mkSamples(1.4, 1.0) },
  ];

  for (const s of scenarios) {
    const r = shadowEvaluate(s.samples);
    console.log(
      `  ${s.name}: n=${r.n} cost_delta=${r.meanCostDeltaPct.toFixed(1)}%  ` +
        `lat_delta=${r.meanLatencyDeltaPct.toFixed(1)}%  ` +
        `alert=${r.alert}${r.reasons.length ? "  reasons=" + r.reasons.join("; ") : ""}`,
    );
  }
}

function policyEngineDemo(): void {
  console.log("\n--- PolicyEngine — promote then rollback in O(1) ---");
  const engine = new PolicyEngine("baseline-digest");
  engine.promote("candidate-digest-v2", 0.1);
  const rng = makeRng(42);
  let candidateCount = 0;
  for (let i = 0; i < 1000; i++) {
    if (engine.pick(rng).chose === "candidate") candidateCount++;
  }
  console.log(
    `  after promote to 10%: ${candidateCount}/1000 picks chose candidate (target ~100)`,
  );
  engine.rollback();
  let postCount = 0;
  for (let i = 0; i < 1000; i++) {
    if (engine.pick(rng).chose === "candidate") postCount++;
  }
  console.log(`  after rollback: ${postCount}/1000 (target 0)`);
}

function canaryDemo(): void {
  console.log("\n" + "=".repeat(95));
  console.log("CANARY ROLLOUT — six stages, five gates, injected regressions");
  console.log("=".repeat(95));

  rolloutReport("Clean promotion", NO_REGRESSION);
  rolloutReport("Small cost regression (10%) — within gate", {
    ...NO_REGRESSION,
    costMult: 1.1,
  });
  rolloutReport("Cost regression 25%", { ...NO_REGRESSION, costMult: 1.25 });
  rolloutReport("Latency regression 80%", {
    ...NO_REGRESSION,
    latencyMult: 1.8,
  });
  rolloutReport("Thumbs-down regression 60%", {
    ...NO_REGRESSION,
    thumbsDownMult: 1.6,
  });
  rolloutReport("Quality silent + cost creep", {
    ...NO_REGRESSION,
    costMult: 1.15,
    thumbsDownMult: 1.45,
  });

  // Programmatic outcome of canaryRollout() for the same six scenarios.
  console.log("\n--- canaryRollout() programmatic verdict ---");
  const scenarios: { name: string; reg: Regression }[] = [
    { name: "clean", reg: NO_REGRESSION },
    { name: "cost 10%", reg: { ...NO_REGRESSION, costMult: 1.1 } },
    { name: "cost 25%", reg: { ...NO_REGRESSION, costMult: 1.25 } },
    { name: "latency 80%", reg: { ...NO_REGRESSION, latencyMult: 1.8 } },
    { name: "thumbs 60%", reg: { ...NO_REGRESSION, thumbsDownMult: 1.6 } },
    {
      name: "cost 15% + thumbs 45%",
      reg: { ...NO_REGRESSION, costMult: 1.15, thumbsDownMult: 1.45 },
    },
  ];
  for (const s of scenarios) {
    const d = canaryRollout(s.reg);
    const verdict = d.promoted
      ? "PROMOTED"
      : `HALT @ stage ${d.stagesAdvanced} on ${d.breaches.join(",")}`;
    console.log(`  ${s.name.padEnd(28)} → ${verdict}`);
  }
}

function main(): void {
  shadowDemo();
  policyEngineDemo();
  canaryDemo();
}

main();
