/**
 * Batch APIs — TypeScript 移植版 + deferred-future dispatcher（延迟 future 调度器）。
 *
 * 两部分：
 *   1. BatchDispatcher：提交 N 个任务，每个任务返回一个 promise，在 batch 完成时 resolve。
 *      模拟 OpenAI / Anthropic JSONL batch 生命周期（in_progress → completed），无需真实网络。
 *      "deferred future" 模式是你在调用现场的做法——先发射后忘记，promise 在数小时后给出答案。
 *   2. 与 main.py 一致的成本模拟器：SYNC、SYNC+CACHE、BATCH、BATCH+CACHE
 *      跨三种工作负载。定价常数 2026-04，来自 docs/en.md。
 *
 * 引用：
 *   - OpenAI Batch API: platform.openai.com/docs/guides/batch
 *   - Anthropic Message Batches: docs.anthropic.com/en/docs/build-with-claude/batch-processing
 *   - Vertex AI Batch Prediction: cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction
 *
 * 在 Node 20+ stdlib 上运行。无 npm 依赖。
 */

import { randomUUID } from "node:crypto";

// -- Cost constants (2026-04) ---------------------------------------------

const BASE_INPUT = 3.0;
const BASE_OUTPUT = 15.0;
const CACHED_INPUT = 0.3;
const CACHE_WRITE_5MIN = 1.25 * BASE_INPUT;
const BATCH_DISCOUNT = 0.5;

// -- Batch dispatcher with deferred futures -------------------------------

type BatchStatus = "queued" | "in_progress" | "completed" | "failed";

type BatchJob<I, O> = {
  id: string;
  input: I;
  promise: Promise<O>;
  // Internal: resolver functions captured at dispatch.
  resolve: (out: O) => void;
  reject: (err: Error) => void;
};

type Batch<I, O> = {
  id: string;
  status: BatchStatus;
  createdAt: number;
  completedAt?: number;
  jobs: BatchJob<I, O>[];
};

class BatchDispatcher<I, O> {
  private readonly batches = new Map<string, Batch<I, O>>();
  private readonly processor: (input: I) => Promise<O>;
  // 模拟 turnaround。真实提供商承诺 24h SLA；典型 P50 为 2-6h。
  // 演示中使用较小的 ms 值以保持运行轻快。
  private readonly turnaroundMs: number;

  constructor(
    processor: (input: I) => Promise<O>,
    turnaroundMs: number,
  ) {
    this.processor = processor;
    this.turnaroundMs = turnaroundMs;
  }

  // 打开一个新 batch。返回可追加任务的 batch id。
  openBatch(): string {
    const id = `batch_${randomUUID().slice(0, 12)}`;
    this.batches.set(id, {
      id,
      status: "queued",
      createdAt: Date.now(),
      jobs: [],
    });
    return id;
  }

  // 向 queued batch 追加任务。返回 caller 在 batch 关闭并处理后 await 的 deferred Promise<O>。
  // 与 OpenAI batch.create + retrieve 流程的用户侧形态一致。
  addJob(batchId: string, input: I): Promise<O> {
    const batch = this.requireBatch(batchId);
    if (batch.status !== "queued") {
      return Promise.reject(
        new Error(`batch ${batchId} not queued (status=${batch.status})`),
      );
    }
    // Hand-rolled deferred so we can resolve from the processor loop.
    let resolve!: (out: O) => void;
    let reject!: (err: Error) => void;
    const promise = new Promise<O>((res, rej) => {
      resolve = res;
      reject = rej;
    });
    batch.jobs.push({
      id: `req_${randomUUID().slice(0, 8)}`,
      input,
      promise,
      resolve,
      reject,
    });
    return promise;
  }

  // 关闭并处理。所有任务 resolve/reject 后返回。
  // 异步迭代模型与真实 batch 相同：你不 await 每个任务；你 await 整个 batch。
  async closeBatch(batchId: string): Promise<Batch<I, O>> {
    const batch = this.requireBatch(batchId);
    batch.status = "in_progress";
    // Simulate provider scheduling delay.
    await new Promise<void>((res) => setTimeout(res, this.turnaroundMs));
    const settlements: Promise<void>[] = batch.jobs.map(async (j) => {
      try {
        j.resolve(await this.processor(j.input));
      } catch (err) {
        j.reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
    await Promise.all(settlements);
    batch.status = "completed";
    batch.completedAt = Date.now();
    return batch;
  }

  getStatus(batchId: string): BatchStatus {
    return this.requireBatch(batchId).status;
  }

  private requireBatch(id: string): Batch<I, O> {
    const b = this.batches.get(id);
    if (!b) throw new Error(`no such batch: ${id}`);
    return b;
  }
}

// -- 模拟分类处理器（无网络） --------------------------

type ClassifyIn = { docId: string; text: string };
type ClassifyOut = { docId: string; label: string; confidence: number };

async function fakeClassifier(input: ClassifyIn): Promise<ClassifyOut> {
  // 基于输入长度奇偶性的确定性玩具分类器。
  const label = input.text.length % 2 === 0 ? "positive" : "neutral";
  return {
    docId: input.docId,
    label,
    confidence: 0.5 + (input.text.length % 5) / 10,
  };
}

async function batchDemo(): Promise<void> {
  console.log("--- Batch dispatcher with deferred futures ---");
  // 演示中 turnaround 设为 50ms（生产环境：24h SLA）。
  const dispatcher = new BatchDispatcher<ClassifyIn, ClassifyOut>(
    fakeClassifier,
    50,
  );
  const batchId = dispatcher.openBatch();
  const futures: Promise<ClassifyOut>[] = [];
  for (let i = 0; i < 6; i++) {
    futures.push(
      dispatcher.addJob(batchId, {
        docId: `doc-${i}`,
        text: `document body number ${i}`,
      }),
    );
  }
  console.log(`status before close: ${dispatcher.getStatus(batchId)}`);
  // Caller awaits jobs; dispatcher closes the batch concurrently.
  const closePromise = dispatcher.closeBatch(batchId);
  const results = await Promise.all(futures);
  await closePromise;
  console.log(`status after close: ${dispatcher.getStatus(batchId)}`);
  for (const r of results) {
    console.log(
      `  ${r.docId} → label=${r.label} confidence=${r.confidence.toFixed(2)}`,
    );
  }
}

// -- Cost simulator -------------------------------------------------------

function costSync(
  docs: number,
  prefixTokens: number,
  perDocTokens: number,
  outTokens: number,
): number {
  let cost = 0;
  for (let i = 0; i < docs; i++) {
    cost += (prefixTokens / 1e6) * BASE_INPUT;
    cost += (perDocTokens / 1e6) * BASE_INPUT;
    cost += (outTokens / 1e6) * BASE_OUTPUT;
  }
  return cost;
}

function costSyncCache(
  docs: number,
  prefixTokens: number,
  perDocTokens: number,
  outTokens: number,
): number {
  let cost = (prefixTokens / 1e6) * CACHE_WRITE_5MIN;
  for (let i = 0; i < docs; i++) {
    if (i > 0) cost += (prefixTokens / 1e6) * CACHED_INPUT;
    cost += (perDocTokens / 1e6) * BASE_INPUT;
    cost += (outTokens / 1e6) * BASE_OUTPUT;
  }
  return cost;
}

function costBatch(
  docs: number,
  prefixTokens: number,
  perDocTokens: number,
  outTokens: number,
): number {
  return costSync(docs, prefixTokens, perDocTokens, outTokens) * BATCH_DISCOUNT;
}

function costBatchCache(
  docs: number,
  prefixTokens: number,
  perDocTokens: number,
  outTokens: number,
): number {
  return (
    costSyncCache(docs, prefixTokens, perDocTokens, outTokens) * BATCH_DISCOUNT
  );
}

function fmtCost(n: number): string {
  return `$${n.toFixed(2)}`.padStart(10);
}

function fmtPct(n: number, baseline: number): string {
  return `${((n / baseline) * 100).toFixed(1)}%`.padStart(5);
}

function runScenario(
  label: string,
  docs: number,
  prefix: number,
  perDoc: number,
  output: number,
): void {
  const sc = costSync(docs, prefix, perDoc, output);
  const scc = costSyncCache(docs, prefix, perDoc, output);
  const bc = costBatch(docs, prefix, perDoc, output);
  const bcc = costBatchCache(docs, prefix, perDoc, output);
  console.log(`\n${label}`);
  console.log(
    `  docs=${docs}, prefix=${prefix}, per_doc=${perDoc}, output=${output}`,
  );
  console.log(`  SYNC            : ${fmtCost(sc)}  (baseline)`);
  console.log(`  SYNC + CACHE    : ${fmtCost(scc)}  (${fmtPct(scc, sc)} of baseline)`);
  console.log(`  BATCH           : ${fmtCost(bc)}  (${fmtPct(bc, sc)} of baseline)`);
  console.log(`  BATCH + CACHE   : ${fmtCost(bcc)}  (${fmtPct(bcc, sc)} of baseline)`);
}

async function main(): Promise<void> {
  await batchDemo();
  console.log("\n" + "=".repeat(80));
  console.log(
    "BATCH API ECONOMICS — batch 叠加 prompt caching，约为 sync 账单的 ~10%",
  );
  console.log("=".repeat(80));
  runScenario(
    "Nightly doc summarization (50k docs)",
    50_000,
    4000,
    2000,
    200,
  );
  runScenario(
    "Content classification (200k items, short per item)",
    200_000,
    1500,
    300,
    50,
  );
  runScenario(
    "Large report draft (small N, heavy per item)",
    1_000,
    6000,
    15_000,
    2000,
  );
}

main().catch((err: unknown) => {
  console.error(err);
  process.exitCode = 1;
});
