# 提示缓存与语义缓存经济学

> **定价快照日期：2026-04。** 以下数值声明反映了本课发布时捕获的供应商价目表；在下游引用前，请对照链接文档核实。

> 缓存在两个层面发生。L2（供应商级）提示/前缀缓存为重复前缀复用注意力 KV —— Anthropic 的提示缓存文档宣传在长提示上最高可达 90% 成本降低和 85% 延迟降低；对于 Claude 3.5 Sonnet，缓存读取为 $0.30/M，而新鲜输入为 $3.00/M，TTL 为 5 分钟，1 小时 TTL 选项的写入溢价为 2 倍（docs.anthropic.com，2026-04）。OpenAI 提示缓存对 ≥1024 token 的提示自动生效，缓存输入定价约为新鲜的 90% 折扣（platform.openai.com，2026-04）；具体每模型缓存费率取决于实时价目表。L1（应用级）语义缓存在嵌入相似度命中时完全跳过 LLM。供应商"95% 准确率"指的是匹配正确性，而非命中率 —— 报告的生产命中率从 10%（开放式聊天）到 70%（结构化 FAQ）不等；没有任何供应商发布官方基线，因此将这些视为社区遥测而非保证。生产陷阱：并行化杀死缓存（在第一次缓存写入完成前发出 N 个并行请求可能使支出膨胀数倍），且前缀内的动态内容完全阻止缓存命中。ProjectDiscovery 报告通过将动态文本移出可缓存前缀，命中率从 7% 提升到 74%（2025-11）。

**类型:** 学习
**语言:** Python（标准库，玩具双层缓存模拟器）
**前置知识:** Phase 17 · 04（vLLM 服务内部原理），Phase 17 · 06（SGLang RadixAttention）
**时间:** 约 60 分钟

## 学习目标

- 区分 L2 提示/前缀缓存（供应商处 KV 复用）与 L1 语义缓存（相似提示时跳过 LLM）。
- 解释 Anthropic 的 `cache_control` 显式标记和两个 TTL 选项（5 分钟 vs 1 小时）及其价格倍数。
- 给定命中率、提示/响应混合和 token 价格，计算预期月度节省。
- 说出使账单膨胀 5-10 倍的并行化反模式，以及使命中率崩溃的动态内容反模式。

## 问题背景

你为 RAG 服务添加了提示缓存。账单持平。你测量命中率；只有 7%。你的提示看起来是静态的，但实际上不是 —— 系统提示包含精确到分钟的当前日期、请求 ID 和用于多样性的随机示例重排序。每个请求写入新缓存条目，读取为零。

另外，你的智能体每个用户问题并行运行十个工具调用。全部十个在第一次缓存写入完成前到达供应商。十次写入，零次读取。你的账单是"使用缓存"预期成本的 5-10 倍。

缓存是一个协议，不是一个标志。两个层面，两种不同的失效模式。

## 核心概念

### L2 —— 供应商提示/前缀缓存

供应商为可缓存前缀存储注意力 KV，并在下一个匹配前缀的请求上复用它。你支付一次写入成本，读取几乎免费。

**Anthropic（Claude 3.5 / 3.7 / 4 系列）**：请求中的显式 `cache_control` 标记。你标记哪些块是可缓存的。TTL：5 分钟（写入成本 1.25 倍基础）或 1 小时（写入成本 2 倍基础）。缓存读取：Claude 3.5 Sonnet 上 $0.30/M，新鲜输入 $3.00/M —— 便宜 10 倍（docs.anthropic.com，截至 2026-04）。不同模型费率不同（Opus/Haiku 单独发布）；始终交叉核对实时定价页面。

**OpenAI**：对 ≥1024 token 的提示自动缓存（platform.openai.com，2026-04）。无显式标志。当前 gpt-4o/gpt-5 价目表上缓存输入约为新鲜输入的 10%。文档和发布说明均未发布官方命中率基线；社区报告在仔细设计提示时集中在 30-60%。监控 `usage.cached_tokens` 来测量你自己的。

**Google（Gemini）**：通过显式 API 进行上下文缓存；1M token 上下文意味着缓存收益更大。

**自托管（vLLM、SGLang）**：Phase 17 · 06 涵盖 RadixAttention —— 相同模式在你自己的算力上。

### L1 —— 应用级语义缓存

在调用 LLM 之前，对提示做哈希、嵌入，并查找相似的缓存请求（余弦相似度高于阈值，通常 0.95+）。命中时返回缓存响应。未命中时调用 LLM 并缓存结果。

开源：Redis Vector Similarity、GPTCache、Qdrant。商业：Portkey Cache、Helicone Cache。

供应商准确率声明指的是返回的缓存响应在语义上适当的频率 —— 不是你命中的频率。生产命中率：

- 开放式聊天：10-15%。
- 结构化 FAQ / 支持：40-70%。
- 代码问题：20-30%（微小变体杀死命中）。
- 语音智能体重复提示：50-80%（语音归一化固定集）。

### 并行化反模式

你的智能体进行 10 个并行工具调用。全部 10 个具有相同的 4K token 系统提示。Anthropic 缓存写入是按请求的；第一次缓存写入在供应商看到提示后约 300 ms 完成。请求 2-10 在同一毫秒窗口到达，每个都看到缓存未命中。你支付 10 次写入溢价，0 次读取折扣。

修复：先串行后批处理 —— 先单独发起请求 1，然后在 1 的缓存填充后发起 2-10。为第一个工具调用增加 300 ms；节省 5-10 倍账单。

### 动态内容反模式

你的系统提示看起来像这样：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每个请求都是唯一的。每个请求都写入。零命中。

修复：将所有真正静态的内容移到可缓存前缀；在缓存边界后附加动态内容：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 通过这种方式将缓存命中率从 7% 提升到 74%，并发布了详细分析。

### 叠加批处理 + 缓存用于夜间工作负载

批处理 API（Phase 17 · 15）在 24 小时周转时提供 50% 折扣。叠加缓存输入再获得约 10 倍。夜间分类、标注和报告生成工作负载可以通过叠加降至同步未缓存成本的约 10%。

### 你应该记住的数字

定价点来自链接的供应商文档，每几个月漂移一次 —— 在依赖前重新核对。

- Anthropic 缓存读取：Claude 3.5 Sonnet 上 $0.30/M，约为新鲜输入的 10 倍便宜（docs.anthropic.com）。
- Anthropic 缓存写入溢价：1.25 倍（5 分钟 TTL）或 2 倍（1 小时 TTL）。
- OpenAI 自动缓存：适用于 ≥1024 token 的提示；当前价目表上缓存输入定价约为新鲜输入的 10%（platform.openai.com）。
- 语义缓存命中率（社区报告）：开放式聊天约 10%；结构化 FAQ 最高约 70%。非供应商文档基线。
- ProjectDiscovery：通过将动态内容移出前缀，命中率从 7% 提升到 74%（项目博客，2025-11）。
- 并行化反模式：当 N 个并行请求错过第一次缓存写入时，典型报告账单膨胀 5-10 倍。

## 动手实践

`code/main.py` 在混合工作负载上模拟 L1 + L2 缓存。报告命中率、账单，并展示并行化惩罚。

## 交付成果

本课产出 `outputs/skill-cache-auditor.md`。给定提示模板和流量，审计可缓存性并推荐重构。

## 练习

1. 运行 `code/main.py`。切换并行化标志。账单变化多少？
2. 你的系统提示包含日期。把它移出去。展示前后命中率计算。
3. 给定你的请求到达率，计算 1 小时 TTL（2 倍写入）与 5 分钟 TTL（1.25 倍写入）的盈亏平衡。
4. 语义缓存在 0.95 阈值时命中 20%。在 0.85 时命中 50%，但你看到错误的缓存响应。选择正确的阈值并论证。
5. 你每个用户问题批处理 10 个并行子查询。在不增加端到端延迟的情况下重写为缓存友好。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| L2 提示缓存 | "前缀缓存" | 供应商为重复前缀存储 KV |
| `cache_control` | "Anthropic 缓存标记" | 标记可缓存块的显式属性 |
| 缓存写入溢价 | "写入税" | 首次未命中到缓存的额外成本（1.25 倍或 2 倍） |
| L1 语义缓存 | "嵌入缓存" | 调用 LLM 前的应用级哈希和嵌入 |
| GPTCache | "LLM 缓存库" | 流行的开源 L1 缓存库 |
| 缓存命中率 | "命中/总计" | 从缓存服务的请求比例 |
| 并行化反模式 | "N 写入陷阱" | N 个并行请求错过缓存 N 次 |
| 动态内容陷阱 | "提示中的时间陷阱" | 前缀中的动态字节杀死命中率 |
| RadixAttention | "副本内缓存" | SGLang 的前缀缓存实现 |

## 延伸阅读

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— 官方 `cache_control` 语义和 TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) —— 自动缓存行为和资格。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
