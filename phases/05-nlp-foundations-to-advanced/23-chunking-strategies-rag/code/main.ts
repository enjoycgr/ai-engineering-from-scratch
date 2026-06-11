// RAG 分块策略的 TypeScript 实现：固定分块、递归分块、语义分块、
// 句子分块、父文档分块。与 code/main.py 对应，并遵循
// LangChain.js 的分割器层级（RecursiveCharacterTextSplitter）。
// 来源：
//   https://docs.langchain.com/oss/javascript/integrations/splitters
//   https://philna.sh/blog/2024/09/18/how-to-chunk-text-in-javascript-for-rag-applications/
//   https://github.com/langchain-ai/langchainjs (textsplitters package)

import { createHash } from "node:crypto";

type Vec = readonly number[];

type ParentChildPair = {
  child: string;
  parentIdx: number;
  parent: string;
};

const TOKEN_RE = /[a-z0-9]+/g;

function tokenize(text: string): string[] {
  return text.toLowerCase().match(TOKEN_RE) ?? [];
}

function hashEmbed(text: string, dim = 256): Vec {
  if (dim <= 0) throw new Error("dim must be positive");
  // Hashing-trick 嵌入器：每个 token 向一个哈希维度贡献 +/-1。
  // 确定性、无需训练，可作为生产级嵌入器（BGE-M3、text-embedding-3-small、voyage-3）的替代品。
  const vec = new Array<number>(dim).fill(0);
  for (const tok of tokenize(text)) {
    const digest = createHash("md5").update(tok).digest();
    const idx = digest.readUInt32BE(0) % dim;
    const sign = digest[4] % 2 === 0 ? -1 : 1;
    vec[idx] += sign;
  }
  let norm = 0;
  for (const v of vec) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm === 0) return vec;
  return vec.map((v) => v / norm);
}

function cosine(a: Vec, b: Vec): number {
  let dot = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) dot += a[i] * b[i];
  return dot;
}

function chunkFixed(text: string, size: number, overlap = 0): string[] {
  if (size <= 0) throw new Error("size must be positive");
  const step = size - overlap;
  if (step <= 0) throw new Error("overlap must be less than size");
  const out: string[] = [];
  for (let i = 0; i < text.length; i += step) {
    const piece = text.slice(i, i + size);
    if (piece.trim().length > 0) out.push(piece);
  }
  return out;
}

function chunkRecursive(
  text: string,
  size: number,
  seps: readonly string[] = ["\n\n", "\n", ". ", " "],
): string[] {
  if (size <= 0) throw new Error("size must be positive");
  // 与 LangChain.js RecursiveCharacterTextSplitter 对应：先尝试最强的分隔符（段落），
  // 当当前轮次产生的 chunk 大于 `size` 时，降级到较弱的分隔符（句子、词）。
  if (text.length <= size) {
    const t = text.trim();
    return t.length > 0 ? [t] : [];
  }
  for (const sep of seps) {
    if (!text.includes(sep)) continue;
    const parts = text.split(sep);
    const chunks: string[] = [];
    let buf = "";
    for (const part of parts) {
      const candidate = buf.length === 0 ? part : buf + sep + part;
      if (candidate.length <= size) {
        buf = candidate;
      } else {
        if (buf.length > 0) chunks.push(buf.trim());
        buf = part;
      }
    }
    if (buf.length > 0) chunks.push(buf.trim());
    return chunks.filter((c) => c.length > 0);
  }
  return chunkFixed(text, size);
}

function splitSentences(text: string): string[] {
  return text
    .trim()
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
}

function chunkSemantic(text: string, threshold = 0.3, minChars = 40): string[] {
  // 语义分块：在相邻句子 embedding (嵌入) 相似度低于阈值处切分。
  // threshold 过高会产生碎片；过低会产生一个巨大 chunk。
  const sentences = splitSentences(text);
  if (sentences.length === 0) return [];
  const embs = sentences.map((s) => hashEmbed(s));
  const groups: string[][] = [[sentences[0]]];
  for (let i = 1; i < sentences.length; i += 1) {
    const sim = cosine(embs[i], embs[i - 1]);
    const current = groups[groups.length - 1];
    const joinedLen = current.join(" ").length;
    if (sim < threshold && joinedLen >= minChars) {
      groups.push([sentences[i]]);
    } else {
      current.push(sentences[i]);
    }
  }
  return groups.map((g) => g.join(" "));
}

function chunkSentence(text: string, sentencesPerChunk = 3): string[] {
  if (sentencesPerChunk <= 0) throw new Error("sentencesPerChunk must be positive");
  // 句子分块：每个 chunk N 个句子。成本仅为语义分块的一小部分。
  const sentences = splitSentences(text);
  const out: string[] = [];
  for (let i = 0; i < sentences.length; i += sentencesPerChunk) {
    out.push(sentences.slice(i, i + sentencesPerChunk).join(" "));
  }
  return out;
}

function chunkParentChild(text: string, parentSize = 800, childSize = 200): ParentChildPair[] {
  // 父文档分块：存储小的子 chunk 用于检索，大的父 chunk 用于上下文。
  const parents = chunkRecursive(text, parentSize);
  const pairs: ParentChildPair[] = [];
  parents.forEach((parent, parentIdx) => {
    const children = chunkRecursive(parent, childSize);
    for (const child of children) {
      pairs.push({ child, parentIdx, parent });
    }
  });
  return pairs;
}

function retrieveRecall(
  chunks: readonly string[],
  query: string,
  goldSubstrings: readonly string[],
  topK = 3,
): boolean {
  // 计算 recall@k：top-k chunk 中是否包含任一 gold 子串。
  const embs = chunks.map((c) => hashEmbed(c));
  const qEmb = hashEmbed(query);
  const scored = embs.map((e, i) => ({ score: cosine(e, qEmb), idx: i }));
  scored.sort((x, y) => y.score - x.score);
  const top = scored.slice(0, topK).map(({ idx }) => chunks[idx]);
  return top.some((c) => goldSubstrings.some((g) => c.toLowerCase().includes(g.toLowerCase())));
}

function main(): void {
  const doc = `Chapter 1. Introduction. This contract is between Acme Corp and Beta Inc. The parties agree to the following terms.

Chapter 2. Payment. Acme will pay Beta thirty thousand dollars on the first of each month. Late payments incur a five percent fee.

Chapter 3. Termination. Either party may terminate this agreement with ninety days written notice. Termination for cause requires only thirty days notice. Breach of payment constitutes cause.

Chapter 4. Confidentiality. Both parties agree to keep trade secrets confidential. This obligation survives termination of the agreement.

Chapter 5. Miscellaneous. This agreement is governed by the laws of the State of California. Disputes shall be resolved by arbitration.`;

  console.log("=== strategy comparison ===\n");

  const fixed = chunkFixed(doc, 300, 50);
  console.log("fixed (300 chars, 50 overlap):    " + fixed.length + " chunks");

  const rec = chunkRecursive(doc, 300);
  console.log("recursive (300 chars):            " + rec.length + " chunks");

  const sem = chunkSemantic(doc);
  console.log("semantic (hash-trick):            " + sem.length + " chunks");

  const sent = chunkSentence(doc, 3);
  console.log("sentence (3 per chunk):           " + sent.length + " chunks");

  const pc = chunkParentChild(doc, 800, 200);
  const parentSet = new Set(pc.map((m) => m.parentIdx));
  console.log("parent-child (800 / 200):         " + pc.length + " children, " + parentSet.size + " parents");

  const queries: ReadonlyArray<{ q: string; gold: readonly string[] }> = [
    { q: "When can either party terminate?", gold: ["ninety days", "thirty days"] },
    { q: "What is the late payment fee?", gold: ["five percent"] },
    { q: "Which state laws apply?", gold: ["California"] },
  ];

  console.log("\n=== recall@3 on 3 queries ===");
  const strategies: ReadonlyArray<{ name: string; chunks: readonly string[] }> = [
    { name: "fixed", chunks: fixed },
    { name: "recursive", chunks: rec },
    { name: "semantic", chunks: sem },
    { name: "sentence", chunks: sent },
    { name: "parent", chunks: Array.from(new Set(pc.map((m) => m.parent))) },
  ];
  for (const { name, chunks } of strategies) {
    const hits = queries.reduce((acc, { q, gold }) => acc + (retrieveRecall(chunks, q, gold) ? 1 : 0), 0);
    console.log("  " + name.padEnd(12) + ": " + hits + " / " + queries.length);
  }

  console.log("\nnote: hash-trick embedder is noisy.");
  console.log("production embedders (BGE, text-3) give 20-40 pp higher recall on the same chunks.");
}

main();
