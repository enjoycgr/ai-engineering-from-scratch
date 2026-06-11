/**
 * 可观测性 —— OpenTelemetry 风格的 GenAI 追踪器 + 保留模拟器（TypeScript）。
 *
 * 两部分：
 *   1. 使用 OpenTelemetry GenAI 语义约定属性名
 *      （gen_ai.system、gen_ai.request.model、gen_ai.usage.*）的最小内存追踪器。
 *      无 SDK。只需一个结构化日志发射器，你可以通过更换导出器将其发送到
 *      Helicone/Phoenix/Langfuse。
 *   2. 与 main.py 相同的 1M 追踪天保留模拟器，包含五种
 *      采样策略和 2026 年价格近似。
 *
 * 引用：参见 docs/en.md 中的 OpenTelemetry GenAI 约定、Arize AX 零拷贝
 * 定价声明、Langfuse/Helicone 层级比较。
 *
 * 在 Node 20+ 标准库上运行。无 npm 依赖。
 */

import { randomUUID, createHash } from "node:crypto";

// -- 追踪器 ----------------------------------------------------------------

// OpenTelemetry GenAI 语义约定（2025 规范）。
// https://opentelemetry.io/docs/specs/semconv/gen-ai/
type GenAIAttributes = {
  "gen_ai.system": string;
  "gen_ai.request.model": string;
  "gen_ai.operation.name": "chat" | "text_completion" | "embeddings";
  "gen_ai.usage.input_tokens"?: number;
  "gen_ai.usage.output_tokens"?: number;
  "gen_ai.response.model"?: string;
  "gen_ai.response.finish_reasons"?: string[];
  "gen_ai.response.id"?: string;
  // 可选但对成本 / 缓存分析有用。
  "gen_ai.usage.cached_input_tokens"?: number;
  "gen_ai.request.temperature"?: number;
};

type SpanStatus = "OK" | "ERROR";

type Span = {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  startNs: bigint;
  endNs?: bigint;
  status: SpanStatus;
  attributes: GenAIAttributes & Record<string, unknown>;
  events: SpanEvent[];
};

type SpanEvent = {
  ts: bigint;
  name: string;
  attributes?: Record<string, unknown>;
};

// 导出器契约：真正的发货器（Helicone、OpenLLMetry、Phoenix）如何
// 接收完成的 span。在生产环境中将其替换为真正的 OTLP HTTP 导出器。
type SpanExporter = (span: Readonly<Span>) => void;

class GenAITracer {
  private active: Span[] = [];
  private readonly exporter: SpanExporter;

  constructor(exporter: SpanExporter) {
    this.exporter = exporter;
  }

  startSpan(name: string, attributes: GenAIAttributes): Span {
    const parent = this.active[this.active.length - 1];
    const span: Span = {
      traceId: parent ? parent.traceId : randomUUID().replace(/-/g, ""),
      spanId: randomUUID().replace(/-/g, "").slice(0, 16),
      parentSpanId: parent?.spanId,
      name,
      startNs: process.hrtime.bigint(),
      status: "OK",
      attributes: { ...attributes },
      events: [],
    };
    this.active.push(span);
    return span;
  }

  addEvent(span: Span, name: string, attributes?: Record<string, unknown>): void {
    span.events.push({ ts: process.hrtime.bigint(), name, attributes });
  }

  endSpan(span: Span, status: SpanStatus = "OK"): void {
    span.endNs = process.hrtime.bigint();
    span.status = status;
    // 无论严格顺序如何，都从活动栈中移除。
    const idx = this.active.lastIndexOf(span);
    if (idx >= 0) this.active.splice(idx, 1);
    this.exporter(span);
  }
}

// 控制台导出器（开发）。真正的导出器会批量并 POST 到 OTLP。
function consoleExporter(span: Readonly<Span>): void {
  const durMs =
    span.endNs !== undefined
      ? Number(span.endNs - span.startNs) / 1_000_000
      : 0;
  const obj = {
    trace_id: span.traceId,
    span_id: span.spanId,
    parent_span_id: span.parentSpanId,
    name: span.name,
    duration_ms: Number(durMs.toFixed(3)),
    status: span.status,
    attributes: span.attributes,
    events: span.events.map((e) => ({
      name: e.name,
      attributes: e.attributes,
    })),
  };
  console.log(JSON.stringify(obj));
}

// 采样导出器 —— 包装另一个导出器。匹配下面
// 保留模拟器中的规则集：保留所有错误 + 高成本，以概率 p 采样成功。
function makeSamplingExporter(
  inner: SpanExporter,
  successRate: number,
  rng: () => number = Math.random,
): SpanExporter {
  return (span) => {
    const isError = span.status === "ERROR";
    const inTokens = (span.attributes["gen_ai.usage.input_tokens"] as number) ?? 0;
    const outTokens =
      (span.attributes["gen_ai.usage.output_tokens"] as number) ?? 0;
    const totalTokens = inTokens + outTokens;
    const isHighCost = totalTokens > 8000;
    if (isError || isHighCost) {
      inner(span);
      return;
    }
    if (rng() < successRate) inner(span);
  };
}

// -- Mocked LLM call（无网络）------------------------------------------

type MockProvider = "openai" | "anthropic" | "self-hosted";

type MockLLMResult = {
  text: string;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  finishReason: "stop" | "length" | "content_filter";
  responseId: string;
};

function mockLLMCall(
  provider: MockProvider,
  model: string,
  prompt: string,
  forceError = false,
): MockLLMResult {
  if (forceError) {
    throw new Error(`${provider}/${model}: simulated rate_limit_exceeded`);
  }
  // 玩具 token 计数器 —— 4 字符/token，每个 prompt 确定性的。
  const inputTokens = Math.max(1, Math.floor(prompt.length / 4));
  const seed = parseInt(
    createHash("sha256").update(prompt).digest("hex").slice(0, 8),
    16,
  );
  const outputTokens = 80 + (seed % 220);
  const cachedInputTokens = prompt.includes("system prompt cached")
    ? Math.floor(inputTokens * 0.9)
    : 0;
  return {
    text: `[mock ${provider}/${model}] echo: ${prompt.slice(0, 40)}`,
    inputTokens,
    outputTokens,
    cachedInputTokens,
    finishReason: outputTokens > 250 ? "length" : "stop",
    responseId: `resp_${seed.toString(16)}`,
  };
}

function traceLLMCall(
  tracer: GenAITracer,
  provider: MockProvider,
  model: string,
  prompt: string,
  forceError = false,
): MockLLMResult | undefined {
  const span = tracer.startSpan("chat.completion", {
    "gen_ai.system": provider,
    "gen_ai.request.model": model,
    "gen_ai.operation.name": "chat",
    "gen_ai.request.temperature": 0.7,
  });
  tracer.addEvent(span, "prompt.user", { length: prompt.length });
  try {
    const result = mockLLMCall(provider, model, prompt, forceError);
    span.attributes["gen_ai.response.model"] = model;
    span.attributes["gen_ai.usage.input_tokens"] = result.inputTokens;
    span.attributes["gen_ai.usage.output_tokens"] = result.outputTokens;
    span.attributes["gen_ai.usage.cached_input_tokens"] =
      result.cachedInputTokens;
    span.attributes["gen_ai.response.finish_reasons"] = [result.finishReason];
    span.attributes["gen_ai.response.id"] = result.responseId;
    tracer.endSpan(span, "OK");
    return result;
  } catch (err) {
    span.attributes["error.type"] = "rate_limit_exceeded";
    tracer.addEvent(span, "exception", { message: String(err) });
    tracer.endSpan(span, "ERROR");
    return undefined;
  }
}

// -- 保留 / 成本模拟器 -------------------------------------------

const BYTES_PER_TRACE = 4500;
const COST_PER_GB_MONTH = 0.023; // S3 standard 2026 approx
const OBSERVABILITY_INGEST_PER_GB = 0.5; // Datadog-class
const ARIZE_AX_PER_GB = 0.005; // zero-copy Iceberg claim

type Strategy = {
  name: string;
  sampleRate: number;
  keepErrors: boolean;
  keepHighCost: boolean;
};

const STRATEGIES: Strategy[] = [
  { name: "100% retain", sampleRate: 1.0, keepErrors: true, keepHighCost: true },
  { name: "10% random sample", sampleRate: 0.1, keepErrors: false, keepHighCost: false },
  { name: "5% success + 100% errors", sampleRate: 0.05, keepErrors: true, keepHighCost: false },
  { name: "5% success + errors + $$$", sampleRate: 0.05, keepErrors: true, keepHighCost: true },
  { name: "1% aggregates only", sampleRate: 0.01, keepErrors: true, keepHighCost: true },
];

// Mulberry32 PRNG —— 确定性，无依赖。
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

type SimResult = {
  name: string;
  retained: number;
  lost: number;
  gbPerDay: number;
  s3Month: number;
  monolithicMonth: number;
  arizeMonth: number;
};

function simulateDay(strategy: Strategy, tracesPerDay = 1_000_000): SimResult {
  const rng = makeRng(7);
  let retained = 0;
  let lost = 0;
  for (let i = 0; i < tracesPerDay; i++) {
    const isError = rng() < 0.02;
    const isHighCost = rng() < 0.01;
    let keep = rng() < strategy.sampleRate;
    if (strategy.keepErrors && isError) keep = true;
    if (strategy.keepHighCost && isHighCost) keep = true;
    if (keep) retained++;
    else lost++;
  }
  const bytesRetained = retained * BYTES_PER_TRACE;
  const gb = bytesRetained / 1e9;
  return {
    name: strategy.name,
    retained,
    lost,
    gbPerDay: gb,
    s3Month: gb * 30 * COST_PER_GB_MONTH,
    monolithicMonth: gb * 30 * OBSERVABILITY_INGEST_PER_GB,
    arizeMonth: gb * 30 * ARIZE_AX_PER_GB,
  };
}

function pad(s: string | number, n: number, left = true): string {
  const str = String(s);
  if (str.length >= n) return str;
  const padding = " ".repeat(n - str.length);
  return left ? padding + str : str + padding;
}

function reportRow(r: SimResult): void {
  console.log(
    `${pad(r.name, 30, false)}  ` +
      `retained=${pad(r.retained, 7)}  ` +
      `lost=${pad(r.lost, 7)}  ` +
      `${pad(r.gbPerDay.toFixed(2), 6)} GB/day  ` +
      `mono=$${pad(r.monolithicMonth.toFixed(2), 8)}  ` +
      `arize=$${pad(r.arizeMonth.toFixed(2), 6)}  ` +
      `s3=$${pad(r.s3Month.toFixed(2), 5)}`,
  );
}

// -- 演示 ------------------------------------------------------------------

function tracerDemo(): void {
  console.log("--- GenAI tracer (OpenTelemetry attribute shape) ---");
  const tracer = new GenAITracer(consoleExporter);
  traceLLMCall(tracer, "openai", "gpt-4o-mini", "What is the capital of France?");
  traceLLMCall(tracer, "anthropic", "claude-3-5-sonnet", "Summarise system prompt cached document");
  // 模拟错误路径。
  traceLLMCall(tracer, "self-hosted", "llama-3-70b", "boom", true);

  console.log("\n--- Sampling exporter: 5% success + 100% errors + high-cost ---");
  const sampled = new GenAITracer(
    makeSamplingExporter(consoleExporter, 0.05, makeRng(42)),
  );
  for (let i = 0; i < 5; i++) {
    traceLLMCall(sampled, "openai", "gpt-4o-mini", `query ${i}`);
  }
  traceLLMCall(sampled, "openai", "gpt-4o-mini", "ratelimit", true);
}

function retentionDemo(): void {
  console.log("\n" + "=".repeat(120));
  console.log(
    "可观测性采样 —— 1M 追踪/天，2026 年价格近似",
  );
  console.log("=".repeat(120));
  for (const s of STRATEGIES) reportRow(simulateDay(s));
  console.log(
    "\n结论: Datadog 级 100% 保留每天花费数百美元。",
  );
  console.log(
    "5% 成功 + 100% 错误 + 高成本保留信号，削减 90% 账单。",
  );
  console.log(
    "当你已有数据湖时，Arize AX 零拷贝模式在规模化时获胜。",
  );
}

function main(): void {
  tracerDemo();
  retentionDemo();
}

main();
