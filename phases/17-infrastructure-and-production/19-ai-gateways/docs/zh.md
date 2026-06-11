# AI Gateways（AI 网关）—— LiteLLM、Portkey、Kong AI Gateway、Bifrost

> Gateway（网关）位于应用与模型提供商之间。核心功能包括 provider routing（提供商路由）、fallback（故障转移）、retries（重试）、rate limiting（限流）、secret references（密钥引用）、observability（可观测性）、guardrails（护栏）。2026 年市场格局：**LiteLLM** 是 MIT 开源，支持 100+ 提供商，兼容 OpenAI，但在约 2000 RPS 时出现瓶颈（8 GB 内存，级联故障）；最适合 Python、<500 RPS、开发/原型阶段。**Portkey** 定位控制平面（guardrails、PII redaction、jailbreak detection、audit trails），2026 年 3 月转为 Apache 2.0 开源，延迟开销 20-40 ms，生产版 $49/月。**Kong AI Gateway** 基于 Kong Gateway——Kong 自家基准在相同 12 CPU 下：比 Portkey 快 228%，比 LiteLLM 快 859%；定价 $100/模型/月（Plus 版最多 5 个）；如果已经在用 Kong，适合企业。**Bifrost**（Maxim AI）——自动重试+可配置退避，OpenAI 429 时回退 Anthropic。**Cloudflare / Vercel AI Gateways**——托管、零运维、基础重试。Data residency（数据驻留）驱动自托管决策；Portkey 和 Kong 处于中间地带，开源+可选托管。

**Type:** Learn
**Languages:** Python（stdlib，toy gateway-routing simulator）
**Prerequisites:** Phase 17 · 01（Managed LLM Platforms）, Phase 17 · 16（Model Routing）
**Time:** ~60 minutes

## Learning Objectives

- 列举 gateway 的六项核心功能（routing、fallback、retries、rate limits、secrets、observability、guardrails）。
- 将 2026 年四个 gateway（LiteLLM、Portkey、Kong AI、Bifrost）映射到各自的 scale ceiling 和用例。
- 引用 Kong 基准（228% vs Portkey、859% vs LiteLLM）并解释为何对 >500 RPS 重要。
- 根据 data residency 和 ops budget 选择 self-hosted vs managed。

## The Problem

你的产品调用 OpenAI、Anthropic 和自托管 Llama。每个提供商有不同的 SDK、错误模型、rate limit 和 auth scheme。你想要 failover（OpenAI 429 时尝试 Anthropic）、单一凭证存储、统一 observability、以及按租户限流。

在应用层重复造轮子会将每个服务与每个提供商耦合。Gateway 层将其整合为一个进程、一个 API（通常是 OpenAI-compatible），再 fan out 到各提供商。

## The Concept

### Six core features

1. **Provider routing**——OpenAI、Anthropic、Gemini、自托管等，统一到一个 API 后。
2. **Fallback**——429、5xx 或质量失败时，换提供商重试。
3. **Retries**——exponential backoff（指数退避），有界尝试次数。
4. **Rate limits**——按租户、按 key、按模型限流。
5. **Secret references**——运行时从 vault 拉取凭证（绝不放在应用中）。
6. **Observability**——OTel + GenAI attributes（Phase 17 · 13）+ cost attribution。
7. **Guardrails**——PII redaction、jailbreak detection、allowed-topics filters。

### LiteLLM — MIT OSS, Python

- 100+ 提供商，OpenAI-compatible，router config、fallback、基础 observability。
- Kong 基准中约 2000 RPS 崩溃；8 GB 内存占用，持续负载下级联故障。
- 最佳场景：Python 应用、<500 RPS、dev/staging gateway、实验性 routing。
- 成本：OSS 免费；云版有免费 tier。

### Portkey — control plane positioning

- 2026 年 3 月起 Apache 2.0 OSS。Guardrails、PII redaction、jailbreak detection、audit trails。
- 每次请求延迟开销 20-40 ms。
- 生产版 $49/月，含 retention + SLA。
- 最佳场景：需要 guardrails + observability 打包的受监管行业。

### Kong AI Gateway — the scale play

- 基于 Kong Gateway（成熟 API gateway 产品，lua+OpenResty）。
- Kong 自家基准在 12-CPU 等效环境：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 版最多 5 个。
- 最佳场景：已在用 Kong；>1000 RPS；愿意购买授权。

### Bifrost（Maxim AI）

- 自动重试+可配置退避。
- OpenAI 429 时回退 Anthropic 是经典配方。
- 较新入局者；商业产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管、零运维。基础重试和 observability。
- 最佳场景：Cloudflare/Vercel 上的 edge-serving JavaScript 应用。
- 在 guardrails 和 rate limits 上不如 Kong/Portkey。

### Self-hosted vs managed

Data residency 是决定性因素。Healthcare 和 finance 默认自托管（LiteLLM 或 Portkey OSS 或 Kong）。Consumer 产品默认托管（Cloudflare AI Gateway）或中间层（Portkey managed）。Hybrid：受监管租户自托管，其他托管。

### Latency budget

- LiteLLM：典型开销 5-15 ms。
- Portkey：开销 20-40 ms。
- Kong：开销 3-8 ms。
- Cloudflare/Vercel：开销 1-3 ms（edge 优势）。

Gateway latency 直接叠加到 TTFT。TTFT P99 < 100 ms SLA 时选 Kong 或 Cloudflare。P99 < 500 ms 时任意均可。

### Rate-limit semantics matter

简单 token-bucket 在中等规模下有效。Multi-tenant 需要 sliding-window + burst allowance + per-tenant tiering。LiteLLM 提供 token-bucket；Kong 提供 sliding-window；Portkey 提供 tiered。

### Gateway + observability + routing compose

Phase 17 · 13（observability）+ 16（model routing）+ 19（gateways）在生产中是同一层。选一个能覆盖三者的工具，或小心拼接：大多数 2026 部署将 Helicone（observability）或 Portkey（guardrails）与 Kong（scale）组合，分角色使用。

### Numbers you should remember

- LiteLLM：约 2000 RPS 崩溃，8 GB 内存。
- Portkey：20-40 ms 开销；2026 年 3 月起 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 版最多 5 个。
- Cloudflare/Vercel：edge 处 1-3 ms 开销。

## Use It

`code/main.py` 模拟在 429/5xx 注入下跨 3 个提供商的 gateway routing with fallback。报告 latency、retry rate 和 fallback hit rate。

## Ship It

本课产出 `outputs/skill-gateway-picker.md`。根据 scale、ops posture、compliance、latency budget 选择 gateway。

## Exercises

1. 运行 `code/main.py`。配置 OpenAI→Anthropic→自托管的 fallback。5% 提供商错误率下预期 hit rate 是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms，基线 300 ms。哪些 gateway 在预算内？
3. Healthcare 客户需要 self-hosted + PII redaction + audit。选 Portkey OSS 还是 Kong。
4. 对比 LiteLLM vs Kong：团队应在什么 RPS 上限迁移？
5. 为 multi-tenant SaaS 设计 rate-limit 策略：free tier、trial tier、paid tier。Token-bucket 还是 sliding-window？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Gateway | "API broker" | Process sitting between apps and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, breaks at 2K RPS |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated log" | Immutable record of every LLM call |
| Token-bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding-window | "precise rate limit" | Time-windowed rate limiter; better fairness |

## Further Reading

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
