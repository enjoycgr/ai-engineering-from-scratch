# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大云厂商，三种截然不同的策略。AWS Bedrock 是一个模型市场（marketplace）—— Claude、Llama、Titan、Stability、Cohere 统一在一个 API 后。Azure OpenAI 是与 OpenAI 的独家合作，加上 Provisioned Throughput Units (PTUs)（预置吞吐单元）提供专属容量。Vertex AI 以 Gemini 为核心，拥有最强的长上下文和多模态能力。2026 年 Artificial Analysis 的实测数据显示，在 Llama 3.1 405B 等效模型上，Azure OpenAI 的中位数延迟约为 50 ms，Bedrock 约为 75 ms —— 差距来自 PTU，因为专属容量优于共享按需实例。决策标准不是"哪个最快"，而是"哪个模型目录和 FinOps（云成本管理）界面匹配我的产品"。本课教你如何带着明确的权衡做选择，而不是凭感觉。

**类型：** 学习
**语言：** Python（标准库，简易成本与延迟比较器）
**前置知识：** 第 11 阶段（LLM 工程）、第 13 阶段（工具与协议）
**时间：** ~60 分钟

## 学习目标

- 说出三种平台策略（市场型 vs 独家合作型 vs Gemini 优先型），并将每种匹配到产品用例。
- 解释 Azure OpenAI 中的 Provisioned Throughput Units (PTUs)（预置吞吐单元）能带来什么，以及为什么 Bedrock 按需实例在 405B 规模上通常慢约 25 ms。
- 为每个平台绘制 FinOps（云成本管理）归因界面（Bedrock Application Inference Profiles vs Vertex 按项目分团队 vs Azure 作用域 + PTU 预留）。
- 写下"最少双供应商"策略，并解释为什么在 2026 年单一供应商锁定是代价高昂的错误。

## 问题

你为你的产品选择了 Claude 3.7 Sonnet。现在你需要提供服务。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过网关调用。直接 API 最简单；Bedrock 增加了 BAA、VPC 终端节点、IAM 和 CloudWatch 归因。网关增加了故障转移、统一计费和跨供应商速率限制。

更深的问题是目录。如果你需要在同一产品中使用 Claude、Llama 和 Gemini，你无法从一个地方全部买到，除非同时使用 Bedrock、Vertex 和 Azure OpenAI。云厂商之间不能互换 —— 它们在模型层 ownership 上下了不同的赌注。

本课将映射这三种赌注、延迟差距、FinOps 差距和锁定风险。

## 概念

### 三种策略

**AWS Bedrock** —— 市场型（marketplace）。Claude（Anthropic）、Llama（Meta）、Titan（AWS 自研）、Stability（图像）、Cohere（嵌入）、Mistral，以及图像和嵌入子目录。一个 API、一个 IAM 界面、一个 CloudWatch 导出。Bedrock 的赌注是客户更看重选择权，而非单一模型。

**Azure OpenAI** —— 独家合作型。你可以在 Azure 数据中心获得 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper 和 OpenAI 模型的微调（fine-tuning）。"Azure OpenAI Service" 目录中没有非 OpenAI 模型 —— 那些会放到 Azure AI Foundry（独立产品）。Azure 的赌注是 OpenAI 保持领先，客户希望在这种特定关系上拥有企业级控制。

**Vertex AI** —— Gemini 优先，其他次之。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方模型）。Vertex 的赌注是多模态长上下文 —— 100 万 token 的 Gemini 上下文是其差异化优势。

### 大规模延迟差距

Artificial Analysis 运行持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需实例）上，Azure OpenAI 的首 token 中位数延迟约为 50 ms；Bedrock 约为 75 ms。这个差距不是 AWS 的失败 —— 而是容量模式差异。Azure 出售 PTU（Provisioned Throughput Units）（预置吞吐单元），为你的租户预留 GPU 容量。Bedrock 的等效产品（Provisioned Throughput）也存在，但起价约每小时 21 美元/单位，大多数客户仍使用共享按需实例。

按需共享容量需要与其他客户的流量竞争。专属容量不需要。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你要么在 Azure 购买 PTU，要么购买 Bedrock Provisioned Throughput，要么接受默认的方差。

### 预置吞吐经济学

Azure PTU：预留的推理计算块。对于可预测的工作负载，比按需实例节省高达约 70%。无论流量如何，每小时固定成本 —— 空闲时也要为预留付费。盈亏平衡点通常在持续利用率 40-60% 左右。

Bedrock Provisioned Throughput：每小时 21-50 美元，取决于模型和区域。数学类似 —— 盈亏平衡点在峰值利用率的一半左右。需要月度承诺。

Vertex 的预留容量按 Gemini SKU 出售；定价因模型和区域而异，公开宣传较少。

### FinOps 界面 —— 真正的差异化因素

**Bedrock Application Inference Profiles** 是市场中最干净的归因方式。用 `team`、`product`、`feature` 标记一个 profile；将所有模型调用路由通过它；CloudWatch 无需后处理即可按 profile 拆分成本。2025 年新增，仍是云厂商原生粒度最细的。

**Vertex** 的归因方式是按项目分团队加无处不在的标签。你将每个团队建模为一个 GCP 项目，在每个资源上打标签，使用 BigQuery Billing Export + DataStudio 做汇总。工作量更大，但 BigQuery 让你可以对成本数据执行任意 SQL。

**Azure** 依赖订阅/资源组作用域加标签，PTU 预留作为一级成本对象。标签从资源组继承，而非从请求继承，因此按请求归因需要 Application Insights 自定义指标或一个在请求头中打戳的网关。

规律：Bedrock 原生最干净，Vertex 通过 BigQuery 最灵活，Azure 除非自行插桩否则最不透明。

### 锁定是 2026 年的风险

当单一模型主导时，单一云厂商承诺是可以接受的。2026 年，前沿模型每月更替 —— 本季度 Claude 3.7，下季度 Gemini 2.5，再下季度 GPT-5。锁定到一个平台意味着被排除在前沿模型的三分之二之外。

有效团队采用的模式：任何产品关键 LLM 调用至少双供应商。Bedrock 加 Azure OpenAI 是常见组合 —— 一个提供 Claude，一个提供 GPT，之间故障转移，同一网关。成本增加微不足道，因为网关会路由到最优选项；在 outage 期间（如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 故障）的可用性提升是决定性的。

### 数据驻留、BAA 和受监管行业

Bedrock：大多数区域提供 BAA；VPC 终端节点；guardrails。常见的金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管默认选择。
Vertex：HIPAA、GDPR、按区域数据驻留；Google Cloud 的合规体系。

三者都满足基本合规要求。差异在于数据保留策略、日志处理方式，以及滥用监控（abuse monitoring）是否会读取你的流量（大多数默认启用；企业可选择退出）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效模型上的中位数 TTFT（含 PTU）：~50 ms
- Bedrock 按需实例中位数 TTFT：~75 ms
- Bedrock Provisioned Throughput：$21-$50/小时/单位
- Azure PTU 盈亏平衡：~40-60% 持续利用率
- 高利用率下 PTU 比按需实例节省：高达 70%

## 使用它

`code/main.py` 在一个合成工作负载上比较三个平台 —— 它建模了按需实例 vs PTU 经济学、TTFT 方差和成本归因保真度。运行它看看 PTU 在哪里划算，以及市场型的模型广度在哪里超过了 TTFT 差距。

## 交付它

本课产出 `outputs/skill-managed-platform-picker.md`。给定一个工作负载画像（所需模型、TTFT SLA、日流量、合规要求），它推荐一个主平台、一个备用平台和一个 FinOps 插桩计划。

## 练习

1. 运行 `code/main.py`。对于 70B 级别模型，Azure PTU 在什么持续利用率下击败按需实例？计算盈亏平衡点并与宣传的 40-60% 区间比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署 —— 哪个放到哪个云厂商，前面放什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户需要 BAA、美国东部数据驻留和 P99 TTFT < 100 ms。选择一个平台并用三个具体功能说明理由。
4. 你发现本月 Bedrock 账单涨了 4 倍但流量没有变化。没有 Application Inference Profiles 时，你如何找到罪魁祸首？有了 profiles，需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜 —— 直接 Anthropic API、Bedrock 按需实例还是 Bedrock Provisioned Throughput？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场（marketplace） |
| Azure OpenAI | "Azure 的 ChatGPT" | Azure 数据中心中的独家 OpenAI 模型，带企业级控制 |
| Vertex AI | "Google 的 LLM" | 以 Gemini 为核心的平台，Model Garden 提供第三方模型 |
| PTU | "专属容量" | Provisioned Throughput Unit —— 预留的推理 GPU，按小时定价 |
| Application Inference Profile | "Bedrock 标签" | 带标签的按产品成本/用量 profile，CloudWatch 原生支持 |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型区，与 Gemini 分开 |
| 最少双供应商 | "LLM 冗余" | 在每个关键 LLM 路径上跨 >=2 个云厂商运行的策略 |
| BAA | "HIPAA 文书" | Business Associate Agreement；处理 PHI 所需；三家都提供 |
| 滥用监控 | "日志观察者" | 提供商对提示/输出的安全扫描；企业版可选择退出 |

## 延伸阅读

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) —— 权威费率表和 Provisioned Throughput 定价。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) —— PTU 经济学和费率表。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) —— Gemini 层级和 Model Garden 附加费。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) —— 跨供应商的持续延迟和吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) —— 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) —— 归因机制并排对比。
