/**
 * 提示 + 语义缓存 —— TypeScript 移植。
 *
 * 三部分：
 *   1. 带 TTL 的 LRU 缓存（L2 提示前缀层的接口 —— 供应商做
 *      这个；我们模拟它）。
 *   2. 带余弦相似度阈值的语义缓存（L1 层）。使用
 *      确定性词哈希"嵌入"，使演示可复现且
 *      无需模型。在生产中将 embed() 替换为真正的嵌入调用。
 *   3. 与 main.py 匹配的双层模拟器，练习并行写入
 *      反模式，包含 5 分钟 vs 1 小时 TTL 溢价。
 *
 * 定价快照：2026-04，来自 docs.anthropic.com / platform.openai.com
 * 通过 docs/en.md。引用前核实价目表。
 *
 * 引用：
 *   - Anthropic prompt-caching: docs.anthropic.com/en/docs/build-with-claude/prompt-caching
 *   - OpenAI prompt-caching: platform.openai.com/docs/guides/prompt-caching
 *   - ProjectDiscovery 7%→74% by moving dynamic content out of prefix
 *     https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching
 *
 * 在 Node 20+ 标准库上运行。无 npm 依赖。
 */

import { createHash } from "node:crypto";

// -- 定价常量（2026-04）-------------------------------------------

const BASE_INPUT = 3.0; // $/M 输入 token（Claude Sonnet 级）
const BASE_OUTPUT = 15.0; // $/M 输出 token
const CACHED_INPUT = 0.3; // ~读取便宜 10 倍
const CACHE_WRITE_5MIN = 1.25 * BASE_INPUT;
const CACHE_WRITE_1HR = 2.0 * BASE_INPUT;

// -- 带 TTL 的 LRU 缓存 ----------------------------------------------------

// Map 在 JS 中保留插入顺序；我们利用这一点实现 LRU。
class LRUCache<K, V> {
  private readonly map = new Map<K, { value: V; expiresAt: number }>();
  private readonly capacity: number;
  private readonly ttlMs: number;
  private readonly now: () => number;

  constructor(capacity: number, ttlMs: number, now: () => number = Date.now) {
    if (capacity <= 0) throw new Error("capacity must be positive");
    this.capacity = capacity;
    this.ttlMs = ttlMs;
    this.now = now;
  }

  get(key: K): V | undefined {
    const entry = this.map.get(key);
    if (!entry) return undefined;
    if (entry.expiresAt <= this.now()) {
      this.map.delete(key);
      return undefined;
    }
    // 刷新 LRU 位置。
    this.map.delete(key);
    this.map.set(key, entry);
    return entry.value;
  }

  set(key: K, value: V): void {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, { value, expiresAt: this.now() + this.ttlMs });
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next();
      if (!oldest.done) this.map.delete(oldest.value);
    }
  }

  has(key: K): boolean {
    return this.get(key) !== undefined;
  }

  get size(): number {
    return this.map.size;
  }
}

// -- 语义缓存 --------------------------------------------------------

// 玩具确定性嵌入：将每个小写词按哈希分桶到 64 维。
// 这足以演示余弦阈值行为；在生产中替换为
// 真正的嵌入提供商（text-embedding-3-small、voyage-3 等）。
const EMBED_DIM = 64;

function embed(text: string): Float32Array {
  const vec = new Float32Array(EMBED_DIM);
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9 ]/g, " ")
    .split(/\s+/)
    .filter((s) => s.length > 0);
  for (const tok of tokens) {
    const h = createHash("sha256").update(tok).digest();
    const idx = h.readUInt16BE(0) % EMBED_DIM;
    // 从第二对取符号位以获得分布，而非纯正值。
    const sign = h[2] & 1 ? 1 : -1;
    vec[idx] += sign;
  }
  // L2 归一化使余弦 = 点积。
  let norm = 0;
  for (let i = 0; i < EMBED_DIM; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < EMBED_DIM; i++) vec[i] /= norm;
  return vec;
}

function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < EMBED_DIM; i++) dot += a[i] * b[i];
  return dot;
}

type SemanticEntry = { vec: Float32Array; response: string };

class SemanticCache {
  private readonly entries: SemanticEntry[] = [];
  private readonly threshold: number;
  private readonly capacity: number;

  constructor(threshold = 0.95, capacity = 1000) {
    if (threshold < 0 || threshold > 1) {
      throw new Error("threshold must be in [0,1]");
    }
    this.threshold = threshold;
    this.capacity = capacity;
  }

  // 返回阈值以上的最佳匹配，或 undefined。
  lookup(prompt: string): { response: string; similarity: number } | undefined {
    const q = embed(prompt);
    let bestSim = -1;
    let bestIdx = -1;
    for (let i = 0; i < this.entries.length; i++) {
      const sim = cosine(q, this.entries[i].vec);
      if (sim > bestSim) {
        bestSim = sim;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0 && bestSim >= this.threshold) {
      return { response: this.entries[bestIdx].response, similarity: bestSim };
    }
    return undefined;
  }

  store(prompt: string, response: string): void {
    if (this.entries.length >= this.capacity) this.entries.shift();
    this.entries.push({ vec: embed(prompt), response });
  }

  get size(): number {
    return this.entries.length;
  }
}

// -- 工作负载 + 模拟器 --------------------------------------------------

// Mulberry32 PRNG。
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

function pickFrom<T>(rng: () => number, arr: readonly T[]): T {
  return arr[Math.floor(rng() * arr.length)];
}

type Request = {
  promptTokens: number;
  prefixHash: string;
  isParallelWave: boolean;
  arrivedAt: number;
  semanticKey: string;
};

function makeWorkload(n = 500, seed = 7): Request[] {
  const rng = makeRng(seed);
  const reqs: Request[] = [];
  const prefixes = Array.from({ length: 12 }, (_, i) => `prefix_${i}`);
  // 一小套 FAQ 风格规范查询 —— 驱动 L1 命中率。
  const faqs = [
    "what is your refund policy",
    "how do I reset my password",
    "what are your office hours",
    "how do I contact support",
  ];
  let now = 0.0;
  while (reqs.length < n) {
    if (rng() < 0.4) {
      for (let k = 0; k < 5; k++) {
        reqs.push({
          promptTokens: pickFrom(rng, [2000, 4000, 8000]),
          prefixHash: pickFrom(rng, prefixes),
          isParallelWave: true,
          arrivedAt: now,
          semanticKey: pickFrom(rng, faqs),
        });
      }
      now += 0.1 + rng() * 1.9;
    } else {
      reqs.push({
        promptTokens: pickFrom(rng, [2000, 4000, 8000]),
        prefixHash: pickFrom(rng, prefixes),
        isParallelWave: false,
        arrivedAt: now,
        semanticKey: pickFrom(rng, faqs),
      });
      now += 0.1 + rng() * 1.9;
    }
  }
  return reqs;
}

type Config = {
  l1Enabled: boolean;
  l2Enabled: boolean;
  parallelPenalty: boolean;
  l1Threshold: number;
  l1HitProb: number;
  ttl: "5min" | "1hr";
};

type SimResult = {
  cost: number;
  l1Hits: number;
  l2Reads: number;
  l2Writes: number;
};

function simulate(reqs: readonly Request[], cfg: Config): SimResult {
  // L2 建模为"足够久以前"见过的前缀哈希集合以被缓存。
  // 这里的 L2 LRU 用于演示 API；模拟器使用更简单的
  // 集合 + 并行波标志（匹配 main.py 的语义）。
  const _l2Lru = new LRUCache<string, true>(
    1024,
    cfg.ttl === "5min" ? 5 * 60_000 : 60 * 60_000,
  );
  void _l2Lru; // 引用以使缓存被练习；行为与下面集合绑定
  const l2Cache = new Set<string>();
  const semantic = new SemanticCache(cfg.l1Threshold);

  // 用 FAQ 键的预制答案预热语义缓存以获得命中。
  semantic.store("what is your refund policy", "Refunds within 30 days.");
  semantic.store("how do I reset my password", "Use the forgot-password link.");
  semantic.store("what are your office hours", "Mon–Fri 9–5 PT.");
  semantic.store("how do I contact support", "Email support@example.com.");

  let l2Writes = 0;
  let l2Reads = 0;
  let l1Hits = 0;
  let cost = 0.0;
  const rng = makeRng(11);

  for (const r of reqs) {
    // L1 层。
    if (cfg.l1Enabled) {
      // 按模拟器契约注入随机化命中率：
      // l1HitProb 比例的请求"语义上足够接近"
      // 预热的 FAQ 条目；我们查找它以保持路径真实。
      if (rng() < cfg.l1HitProb) {
        const hit = semantic.lookup(r.semanticKey);
        if (hit) {
          l1Hits++;
          continue;
        }
      }
    }

    // L2 层。
    if (cfg.l2Enabled) {
      if (l2Cache.has(r.prefixHash)) {
        l2Reads++;
        cost += (r.promptTokens / 1e6) * CACHED_INPUT;
      } else {
        const writeCost =
          cfg.ttl === "5min" ? CACHE_WRITE_5MIN : CACHE_WRITE_1HR;
        cost += (r.promptTokens / 1e6) * writeCost;
        l2Writes++;
        if (!(cfg.parallelPenalty && r.isParallelWave)) {
          l2Cache.add(r.prefixHash);
        }
      }
    } else {
      cost += (r.promptTokens / 1e6) * BASE_INPUT;
    }

    // 输出成本 —— 恒定为 200 token。
    cost += (200 / 1e6) * BASE_OUTPUT;
  }

  return { cost, l1Hits, l2Reads, l2Writes };
}

function report(label: string, cfg: Config, reqs: readonly Request[]): void {
  const res = simulate(reqs, cfg);
  const padLabel = label.padEnd(45);
  const cost = `$${res.cost.toFixed(2)}`.padStart(8);
  console.log(
    `${padLabel}  cost=${cost}  L1=${String(res.l1Hits).padStart(4)}  ` +
      `L2_reads=${String(res.l2Reads).padStart(4)}  ` +
      `L2_writes=${String(res.l2Writes).padStart(4)}`,
  );
}

function main(): void {
  console.log("=".repeat(95));
  console.log(
    "提示 + 语义缓存 —— 500 请求，Claude Sonnet 级定价（2026-04）",
  );
  console.log("=".repeat(95));
  const reqs = makeWorkload();

  report(
    "无缓存",
    {
      l1Enabled: false,
      l2Enabled: false,
      parallelPenalty: true,
      l1Threshold: 0.95,
      l1HitProb: 0.0,
      ttl: "5min",
    },
    reqs,
  );
  report(
    "L2 5 分钟，并行惩罚生效",
    {
      l1Enabled: false,
      l2Enabled: true,
      parallelPenalty: true,
      l1Threshold: 0.95,
      l1HitProb: 0.0,
      ttl: "5min",
    },
    reqs,
  );
  report(
    "L2 5 分钟，并行修复（先串行）",
    {
      l1Enabled: false,
      l2Enabled: true,
      parallelPenalty: false,
      l1Threshold: 0.95,
      l1HitProb: 0.0,
      ttl: "5min",
    },
    reqs,
  );
  report(
    "L2 1 小时 + L1 语义 30%",
    {
      l1Enabled: true,
      l2Enabled: true,
      parallelPenalty: false,
      l1Threshold: 0.95,
      l1HitProb: 0.3,
      ttl: "1hr",
    },
    reqs,
  );
  report(
    "L2 1 小时 + L1 语义 70%（结构化 FAQ）",
    {
      l1Enabled: true,
      l2Enabled: true,
      parallelPenalty: false,
      l1Threshold: 0.95,
      l1HitProb: 0.7,
      ttl: "1hr",
    },
    reqs,
  );

  // 直接演示 LRU + TTL 原语，使 API 可见。
  console.log("\n--- LRU+TTL primitive demo ---");
  const lru = new LRUCache<string, number>(2, 1000);
  lru.set("a", 1);
  lru.set("b", 2);
  lru.set("c", 3); // 驱逐 "a"
  console.log(`after inserting a,b,c with cap=2: has(a)=${lru.has("a")}, has(b)=${lru.has("b")}, has(c)=${lru.has("c")}`);

  // 演示语义缓存余弦行为 —— 同义改写。
  console.log("\n--- Semantic cache cosine threshold demo ---");
  const sc = new SemanticCache(0.5);
  sc.store("how do I reset my password", "Use forgot-password link.");
  const near = sc.lookup("how to reset password please");
  const far = sc.lookup("what is the capital of France");
  console.log(
    `near sim=${(near?.similarity ?? 0).toFixed(3)} response=${near?.response ?? "<miss>"}`,
  );
  console.log(
    `far  sim=${(far?.similarity ?? 0).toFixed(3)} response=${far?.response ?? "<miss>"}`,
  );

  console.log(
    "\n结论: 缓存是一个协议。构建你的提示和批处理以使其生效。",
  );
}

main();
