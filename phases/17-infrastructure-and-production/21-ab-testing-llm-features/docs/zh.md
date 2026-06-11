# A/B Testing（A/B 测试）LLM Features —— GrowthBook、Statsig 与 Vibes Problem

> 传统 A/B testing 并非为非确定性 LLM 设计。关键区别：evals（评估）回答"模型能否胜任？"；A/B tests 回答"用户是否在意？"两者缺一不可；仅凭 vibe check（感觉检查）就上线已成过去。2026 年应测试的内容：prompt engineering（提示工程，措辞）、model selection（模型选择，GPT-4 vs GPT-3.5 vs OSS；准确率 vs 成本 vs 延迟）、generation parameters（生成参数，temperature、top-p）。真实案例：某聊天机器人 reward-model 变体使对话长度提升 +70%、留存率提升 +30%；Nextdoor AI 主题行实验在 reward-function 优化后 CTR 提升 +1%；Khan Academy Khanmigo 在延迟 vs 数学准确率轴上迭代。平台分化：**Statsig**（2025 年 9 月被 OpenAI 以 $1.1B 收购）——sequential testing（序贯测试）、CUPED、all-in-one。**GrowthBook**——开源、warehouse-native（数仓原生）、Bayesian + Frequentist + Sequential 引擎、CUPED、SRM checks、Benjamini-Hochberg + Bonferroni corrections。根据 warehouse-SQL 偏好以及"被 OpenAI 收购"是否对你的组织重要来选择。

**Type:** Learn
**Languages:** Python（stdlib，toy sequential test simulator）
**Prerequisites:** Phase 17 · 13（Observability）, Phase 17 · 20（Progressive Deployment）
**Time:** ~60 minutes

## Learning Objectives

- 区分 evals（"模型能否胜任"）与 A/B tests（"用户是否在意"）。
- 列举三个可测试维度（prompt、model、parameters）并为每个选择指标。
- 解释 CUPED、sequential testing 和 Benjamini-Hochberg multiple-comparison corrections（多重比较校正）。
- 根据 warehouse-SQL 立场和企业收购态度选择 Statsig 或 GrowthBook。

## The Problem

你手工调优了一个 system prompt。感觉更好了。你直接上线。Conversion 随噪声波动。你怪指标。或者你上线了一个新模型，conversion 没变化——是模型退化了，还是变化太小检测不到？你不知道，因为你没有 A/B。

Evals 回答模型在标注集上能否完成任务。它们不回答用户是否更喜欢输出。只有受控在线实验能回答，且实验需要有足够 power、控制非确定性、校正多重比较。

## The Concept

### Evals vs A/B tests

**Evals** —— 离线、标注集、judge（rubric 或 LLM-as-judge 或人工）。回答："输出在固定分布上是否正确/有帮助/安全？"

**A/B test** —— 在线、真实用户、随机化。回答："新变体是否推动了重要的用户级指标？"

两者缺一不可。Evals 在暴露前捕获回退；A/B 在暴露后确认产品影响。

### What to test

1. **Prompt engineering** —— 措辞、system-prompt 结构、示例。指标：task success、user retention、cost/request。
2. **Model selection** —— GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：accuracy（任务）+ cost/request + latency P99。多目标。
3. **Generation parameters** —— temperature、top-p、max_tokens。指标：任务特定（输出多样性 vs 确定性）。

### CUPED —— variance reduction

Controlled-experiments Using Pre-Experiment Data。在比较后周期前回归掉前周期 variance。典型 variance reduction：30-70%。有效样本量免费提升。

实现：Statsig 和 GrowthBook 均支持。

### Sequential testing

经典 A/B 假设固定样本量。Sequential tests（"peek-and-decide"）在反复查看下控制 false-positive rate。Always-valid sequential procedures（mSPRT、Howard's confidence sequences）允许在明显赢家上提前停止。

### Multiple-comparison corrections

以 95% 置信度运行 20 个 A/B test 会随机产生一个 false positive。Bonferroni correction 收紧每测试的 α；Benjamini-Hochberg 控制 false-discovery rate。GrowthBook 两者都实现。

### SRM —— sample ratio mismatch

Assignment hash 将用户随机分配到变体。如果 50/50 拆分实际交付 47/53，说明有 bug——SRM check 标记它。两个平台都实现。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 $1.1B 收购。Hosted，SaaS。
- Sequential testing、CUPED、held-out populations。
- All-in-one：feature flags + experimentation + observability。
- 最佳场景：团队想要打包产品，不介意 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；warehouse-native（直接从 Snowflake/BigQuery/Redshift 读取）。
- 多引擎：Bayesian、Frequentist、Sequential。
- CUPED、SRM、Bonferroni、BH corrections。
- 自托管或托管云。
- 最佳场景：warehouse-SQL 团队，数据团队控制指标层，想要 OSS。

### Non-determinism complicates power

相同提示产生不同输出。传统 power calculation 假设 IID observations。LLM non-determinism 下，effective sample size 低于名义值。将所需样本量乘以约 1.3-1.5 倍作为安全余量。

### Real case outcomes

- 聊天机器人 reward model 变体：对话长度 +70%，留存率 +30%。
- Nextdoor 主题行：reward-function 优化后 CTR +1%。
- Khan Academy Khanmigo：延迟 vs 数学准确率权衡迭代。

### The anti-pattern: shipping on vibes

每位资深工程师都能说出一个因为"感觉更好"就上线、没有 A/B 的功能。大多数在数月内回退了团队未察觉的产品指标。A/B 是强制机制。

### Numbers you should remember

- Statsig 被 OpenAI 收购：$1.1B，2025 年 9 月。
- GrowthBook：开源 MIT；Bayesian + Frequentist + Sequential。
- CUPED variance reduction：30-70%。
- LLM non-determinism → +30-50% 样本量缓冲。

## Use It

`code/main.py` 模拟固定样本和 sequential boundary 的 sequential A/B test。展示 sequential 如何让你提前停止。

## Ship It

本课产出 `outputs/skill-ab-plan.md`。根据功能变更、工作负载、基线，选择平台、gates、样本量。

## Exercises

1. 运行 `code/main.py`。预期 5% 提升、基线 3% conversion，80% power 需要多少样本量？
2. 为受 healthcare 监管的内部部署客户选择 Statsig 或 GrowthBook。
3. 设计一个 A/B test，对比 GPT-4 vs GPT-3.5 在 cost-per-resolved-ticket 上的表现。Primary metric、guardrail metric、secondary metric 分别是什么？
4. Canary 通过但 A/B 显示 -1.2% conversion。你上线吗？写出升级标准。
5. 对前周期 variance 为后周期 60% 的数据应用 CUPED。计算 effective-sample-size 提升。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Eval | "offline test" | Labeled-set evaluation of model capability |
| A/B test | "experiment" | Live randomized comparison on users |
| CUPED | "variance reduction" | Pre-period regression to reduce variance |
| Sequential test | "peek-ok test" | Always-valid procedure allowing early stop |
| Multiple comparison | "the family error" | Running many tests inflates false positives |
| Bonferroni | "tight correction" | Divide α by number of tests |
| Benjamini-Hochberg | "BH FDR" | False-discovery-rate control, less conservative |
| SRM | "bad split" | Sample ratio mismatch; assignment bug |
| Statsig | "OpenAI owned" | Commercial all-in-one, acquired 2025 |
| GrowthBook | "the OSS one" | MIT warehouse-native platform |
| mSPRT | "sequential probability ratio test" | Classical sequential procedure |

## Further Reading

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
