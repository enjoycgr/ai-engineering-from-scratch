// Phase 0 · Lesson 04 — API 与密钥（TypeScript 版本）。
// 从环境变量读取 ANTHROPIC_API_KEY，解析最小化 .env 文件，然后使用
// 全局 fetch 发起一次 /v1/messages 调用。设置 MOCK=1 可跳过网络请求。
// Refs: https://docs.anthropic.com/en/api/messages
//       https://nodejs.org/api/process.html#processenv
//       https://nodejs.org/api/globals.html#fetch (Node 18+ 内置 fetch)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";

type MessagesRequest = {
  model: string;
  max_tokens: number;
  messages: { role: "user" | "assistant"; content: string }[];
};

type MessagesResponse = {
  content: { type: string; text: string }[];
  usage: { input_tokens: number; output_tokens: number };
};

// .env 加载器。与所有框架遵循的格式相同；我们跳过依赖以保持
// 可移植性。每行格式 KEY=VALUE，# 为注释，可选的引号包裹。
function loadDotenv(path: string): Record<string, string> {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    return {};
  }
  const out: Record<string, string> = {};
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

function mergeEnv(): NodeJS.ProcessEnv {
  // process.env 优先，用户可以不修改文件就覆盖配置。
  const fromFile = loadDotenv(resolve(process.cwd(), ".env"));
  return { ...fromFile, ...process.env };
}

// 模拟数据与真实 /v1/messages 响应格式一致，因此无论 MOCK=1 与否，
// 周围代码完全相同。
const MOCK_RESPONSE: MessagesResponse = {
  content: [
    {
      type: "text",
      text: "A neural network is a stack of differentiable functions that learns patterns by adjusting weights against a loss signal.",
    },
  ],
  usage: { input_tokens: 12, output_tokens: 28 },
};

async function callMessages(apiKey: string, request: MessagesRequest): Promise<MessagesResponse> {
  if (process.env.MOCK === "1" || apiKey === "mock") {
    return MOCK_RESPONSE;
  }

  const resp = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify(request),
  });

  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`anthropic ${resp.status}: ${body.slice(0, 200)}`);
  }
  return (await resp.json()) as MessagesResponse;
}

async function main(): Promise<number> {
  const env = mergeEnv();
  const apiKey = env.ANTHROPIC_API_KEY ?? "mock";
  const usingMock = process.env.MOCK === "1" || apiKey === "mock";

  process.stdout.write("=== API Calls ===\n\n");
  process.stdout.write(
    usingMock
      ? "Mode: MOCK (no network). Unset MOCK and export ANTHROPIC_API_KEY for a live call.\n\n"
      : "Mode: LIVE.\n\n",
  );

  const request: MessagesRequest = {
    model: "claude-sonnet-4-6",
    max_tokens: 256,
    messages: [{ role: "user", content: "What is a neural network in one sentence?" }],
  };

  try {
    const response = await callMessages(apiKey, request);
    const text = response.content[0]?.text ?? "";
    process.stdout.write(`response: ${text}\n`);
    process.stdout.write(
      `tokens: ${response.usage.input_tokens} in, ${response.usage.output_tokens} out\n`,
    );
    return 0;
  } catch (err) {
    process.stderr.write(`request failed: ${(err as Error).message}\n`);
    return 1;
  }
}

main().then((code) => process.exit(code));
