# Model Routing（模型路由）—— 成本削减原语

> 动态 broker（代理）评估每个请求（任务类型、token 长度、embedding 相似度、置信度），将简单查询发给廉价模型，复杂查询升级至 frontier model（前沿模型）。也称为 model cascading（模型级联）。生产案例研究表明，在美国/英国/欧盟部署中，iso-quality（同等质量）下可节省 20-60% 成本；高流量 SaaS 上 30% 的 routing efficiency（路由效率）提升可转化为每年六位数的节省。2026 年的背景是 LLM inference（推理）价格每年下降约 10 倍——GPT-4 级别的 token 从 2022 年底的 $20/M 降至 2026 年的约 $0.40/M。大部分降幅来自更好的 serving stacks（Phase 17 · 04-09），而非硬件。Routing 是将这种价格下降转化为利润、同时避免产品倒退的方法。失败模式是 cheap-model drift（廉价模型漂移）：路由将 40% 流量推向较弱模型，推理任务质量下降 3-5%，一个季度内无人察觉。用 online quality metrics（在线质量指标）而非仅离线评估集来把关路由。

**Type:** Learn
**Languages:** Python（stdlib，toy cascading router simulator）
**Prerequisites:** Phase 17 · 01（Managed LLM Platforms）, Phase 17 · 19（AI Gateways）
**Time:** ~60 minutes

## Learning Objectives

- 解释 model cascading：先走廉价模型+置信度检查，低置信度时升级。
- 列举四种 routing signals（路由信号）（任务分类、提示长度、与已知困难集的 embedding 相似度、首轮自置信度）。
- 在目标路由拆分和质量损失容忍度下计算 expected blended cost（预期混合成本）。
- 说出捕捉 cheap-model creep（廉价模型蔓延）的 drift-monitoring metric（漂移监控指标，即 online quality gate）。

## The Problem

你的服务每月在 GPT-5 上花费 $80k。分析显示 70% 的查询是简单的："巴黎现在几点？""改写这句话。"Haiku 级别的模型可以完美处理，成本仅 3%。30% 需要 GPT-5 的推理能力——编程、数学、多步规划。

如果将 70% 路由到廉价模型，30% 到昂贵模型，账单在同等产品质量下降约 65%。这就是 routing。诀窍是在不降低质量的前提下构建 broker。

## The Concept

### Four routing signals

1. **Task classification（任务分类）**：简单/复杂/代码生成/数学/聊天。可以是基于规则的分类器、小型 LLM（Haiku 级别，$0.25/M）或与标注桶的 embedding 相似度。输出：route = cheap / balanced / frontier。

2. **Prompt length**：>4K token 的提示通常需要 frontier 模型保持连贯性。<500 token 通常不需要。

3. **Embedding similarity to known-hard set**：如果查询与已知困难桶接近（cosine > 0.88），直接升级至 frontier。

4. **Self-confidence from first-pass**：先发往廉价模型；如果模型 log-probs 显示低置信度，或拒绝，或输出 hedging language（含糊措辞），则在 frontier 上重试。为约 10% 流量增加 P95 延迟，但节省其余 90% 的 50%+ 成本。

### Three patterns

**Pre-route**（前置分类器）：增加约 5-10ms 延迟；总体最快。

**Cascade**（廉价优先，低置信度升级）：中位延迟约 1.2 倍（廉价运行+验证），升级请求约 2 倍。最佳质量底线。

**Ensemble route**（并行运行廉价和 frontier 模型，reward-model 挑选）：最高质量，最高成本；仅用于关键 A/B testing（A/B 测试）。

### Implementation

AI gateways（Phase 17 · 19）暴露 routing 功能。LiteLLM 有 `router` 配置，支持 fallback 和 cost-routing。Portkey 有 guards + routing。Kong AI Gateway 有基于插件的 routing。OpenRouter 的模型市场提供推荐 API。

开源：RouteLLM（LMSYS）、Not Diamond（商业）、Prompt Mule。

### The 2026 price curve

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier（GPT-5, Claude 4） | — | ~$3-10/M | new tier |

大部分改进来自 serving efficiency——Phase 17 · 04-09 的核心课程转化为提供商端的成本下降。Routing 让你在应用层捕获这些收益，而无需等待所有用户迁移到廉价层级。

### Drift is the real risk

你的路由将 40% 流量发给廉价模型。六个月内，任务分布发生变化（用户变得更复杂，问题更长）。路由器没有察觉，因为其分类器是用 Q1 数据训练的。质量悄然下降。没有人大声抱怨。你在竞争对手的基准测试中发现输了。

用 online quality metrics 把关路由：

- 每条路由的用户 thumbs-up / thumbs-down。
- 对保留样本（5%）的自动化 LLM-judge。
- Escalation rate（升级率）：如果 cascade 升级率 >30%，说明廉价模型被过度路由。
- 每条路由的 refusal rate（拒绝率）。

### Numbers you should remember

- 2026 年 iso-quality 路由节省：案例研究 20-60%。
- LLM 价格 2022-2026 年降幅：每年约 10 倍。
- GPT-4 级别 2022 vs 2026：~$20/M → ~$0.40/M。
- Cascade 延迟影响：中位约 1.2 倍，升级约 2 倍（约 10% 流量）。

## Use It

`code/main.py` 在混合工作负载上模拟 pre-route、cascade 和 ensemble。报告 blended cost、quality loss 和 escalation rate。

## Ship It

本课产出 `outputs/skill-router-plan.md`。根据工作负载和质量预算，选择路由模式和信号。

## Exercises

1. 运行 `code/main.py`。在何种 accuracy floor 下 cascade 击败 pre-route？
2. 你的用户群 30% 企业（复杂查询），70% 免费（简单）。设计路由拆分。用什么 online metric 把关？
3. 某路由质量下降 2% 但节省 40%。是否上线？取决于产品——正反两方论证。
4. 使用 OpenAI / Anthropic API 的 logprobs 实现置信度检查。初始阈值是多少？
5. 六个月内 escalation rate 从 8% 攀升至 22%。诊断三个原因及各自修复方案。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Model routing | "cost broker" | Dynamic choice of model per request |
| Model cascade | "cheap-first escalate" | Run cheap, fall through to frontier on low confidence |
| Pre-route | "classify first" | Classifier up front; no re-run |
| Ensemble route | "parallel pick" | Run multiple, reward-model picks best |
| Escalation rate | "uprouted %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model-routing product |
| Drift | "cheap creep" | Distribution shift without router noticing |
| Online quality gate | "live check" | Automated LLM-judge sampling live traffic |

## Further Reading

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives.
