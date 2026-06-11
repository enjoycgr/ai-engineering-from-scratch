# 多区域 LLM 服务与 KV Cache 局部性

> 轮询负载均衡对缓存型 LLM 推理是有害的。未命中持有其前缀的节点的请求需要支付完整的 prefill 成本 —— 长提示下 P50 约 800 ms，而缓存命中仅约 80 ms。2026 年的生产模式是缓存感知路由器（Rust 版 vLLM Router、llm-d router），它消费 KV-cache 事件并按前缀哈希匹配进行路由。近期研究（GORGO）将跨区域网络延迟作为路由目标中的显式项。商业"跨区域推理"产品（Bedrock cross-region inference、GKE multi-cluster gateways）将推理视为黑盒 —— 它们处理可用性，而非 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月进行了 us-east-1 故障转移，耗时约 22 分钟。DR 现实：32% 的 LLM DR 故障是因为团队备份了权重但忘记了 tokenizer 文件或量化配置。

**类型:** 学习
**语言:** Python（标准库，玩具前缀缓存感知路由器模拟器）
**前置知识:** Phase 17 · 04（vLLM 服务），Phase 17 · 06（SGLang Radix注意力）
**时间:** 约 60 分钟

## 学习目标

- 解释为什么轮询负载均衡会破坏缓存推理，并量化 TTFT 惩罚。
- 绘制缓存感知路由器架构：输入（KV-cache 事件）、算法（前缀哈希匹配）、决胜机制（GPU 利用率）。
- 说出 LLM DR 故障的 32% 驱动因素（缺少 tokenizer 文件 / 量化配置）并陈述三文件 DR 检查清单。
- 区分商业跨区域产品（Bedrock CRI、GKE Multi-Cluster Gateway）与 KV 感知路由。

## 问题背景

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个 ALB 并使用轮询。生产中的前缀缓存命中率降至 8%。TTFT P50 翻三倍。你的 vLLM 日志显示每个请求都在支付完整的 prefill 成本。

轮询对无状态服务是最优的。LLM 推理在设计上是有状态的 —— KV cache 编码了模型已见的一切。盲目路由就是路由到错误的缓存。

另外，你的团队有 DR 计划。你将模型权重备份到跨区域 S3。区域故障发生；你尝试故障转移；副本拒绝启动。你忘记了 tokenizer.json、量化配置和 RoPE 缩放配置在一个你未同步的独立存储桶中。

多区域 LLM 服务是一个缓存问题、一个路由问题和一个 DR 卫生问题 —— 不是负载均衡器问题。

## 核心概念

### 缓存感知路由

请求带着提示到达。路由器对前缀做哈希（比如前 512 个 token）；它询问每个副本"你有这个前缀缓存吗？"。副本在分配和驱逐块时通过 pub/sub 通道发布 KV-cache 事件。路由器选择匹配的副本，如果没有匹配则回退到基于 GPU 利用率的决胜机制。

**vLLM Router**（Rust，2026 生产栈）：订阅 `kv.cache.block_added` 事件，维护前缀哈希 → 副本索引，以 O(1) 查找路由。无匹配时回退到最小队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内等效方案。跨副本路由严格在上游进行。

### 数据

2K token 提示在 H100 上运行 Llama 3.3 70B FP8 的 TTFT P50：
- 缓存命中（同一副本，前缀驻留）：约 80 ms。
- 缓存未命中（冷 prefill）：约 800 ms。

10 倍差距。如果你的路由器在副本间达到 60-80% 的前缀缓存命中率，你就以 N 副本容量近似单副本性能。如果只有 10%，你就近似朴素扩展。

### 跨区域有一个新约束 —— 网络延迟

跨区域 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果将请求从 us-east-1 路由到 ap-southeast-1 的热前缀，节省的 prefill（800 → 80 ms）会被 440 ms 往返所淹没。GORGO（2026 年研究）使这一点显式化 —— 联合最小化 `prefill_time + network_latency`，而非仅 prefill。通常答案是将路由保持在区域内，除非是多 MB 前缀且 prefill 占主导。

### 商业"跨区域推理"对此无帮助

AWS Bedrock cross-region inference 在容量压力下自动将请求路由到其他区域。它优化可用性，而非 TTFT，并将推理视为黑盒。GKE Multi-Cluster Gateway 相同 —— 服务级故障转移，无 KV cache 感知。

即使使用这些产品，你仍然需要一个应用层缓存感知路由器。它们处理"us-east-1 着火"的情况。缓存感知路由处理 TTFT 的情况。

### DR 卫生 —— 32% 缺失文件问题

广泛引用的 2026 年数据：32% 的 LLM DR 故障是因为团队备份了权重但忘记了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE 缩放、attention masks、chat templates）
- 引擎配置（`vllm_config.yaml`、sampling defaults、LoRA adapter manifests）

修复方案是三文件最小 DR 清单：

1. HF 模型仓库下的所有文件（权重 + 配置 + tokenizer）。
2. 引擎特定的服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁定）。

另外：每季度运行一次 DR 演练。JPMorgan 的 us-east-1 演练在 2024 年 11 月达到 22 分钟恢复，仅因为手册经过排练。

### 数据驻留是正交的

欧盟客户的 PHI 不能离开欧盟。如果你的缓存感知路由器将来自巴黎的请求发送到 us-east-1 进行前缀匹配，无论 TTFT 收益如何，你都违反了 GDPR。在优化缓存之前，按驻留边界分区路由器。

### 你应该记住的数字

- 缓存命中 vs 未命中 TTFT 差距：约 10 倍（2K 提示下 80 ms vs 800 ms）。
- 跨区域 RTT 美欧：约 75 ms。
- DR 故障：32% 缺失 tokenizer/量化配置。
- JPMorgan us-east-1 故障转移 2024 年 11 月：22 分钟（30 分钟 SLA）。

## 动手实践

`code/main.py` 模拟多区域工作负载上的三种路由策略（轮询、缓存感知区域、缓存感知全局）。报告缓存命中率、TTFT P50/P99 和跨区域费用。

## 交付成果

本课产出 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计路由计划。

## 练习

1. 运行 `code/main.py`。给定 75 ms RTT，跨区域路由何时击败仅本地路由？
2. 你的缓存命中率从 70% 降至 12%。诊断三个可能原因及可确认每个原因的观测指标。
3. 为 vLLM 中服务的 70B AWQ 量化模型设计 DR 清单，含 5 个 LoRA adapter。列出每个文件和配置。
4. 论证 Bedrock cross-region inference 对具有严格 TTFT SLA 的金融科技公司是否"足够"。引用具体行为。
5. 一个来自巴黎的请求匹配了 us-east-1 的前缀。你路由它吗？写出策略。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| 缓存感知路由 | "智能 LB" | 按前缀哈希匹配路由到持有 KV cache 的副本 |
| KV-cache 事件 | "缓存 pub-sub" | 副本发布块添加/驱逐；路由器索引 |
| 前缀哈希 | "缓存键" | 前 N 个 token 的哈希，用作路由器查找 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式项 |
| 跨区域推理 | "Bedrock CRI" | AWS 产品；可用性故障转移，非 TTFT 感知 |
| DR 清单 | "备份列表" | 恢复所需的每个文件 —— 不只是权重 |
| 数据驻留 | "GDPR 边界" | 限制哪个区域能看到用户数据的法律约束 |
| RTT | "往返时间" | 网络延迟；美欧 75 ms，美亚太 220 ms |
| LLM 感知 LB | "缓存命中 LB" | 缓存感知路由器作为产品类别 |

## 延伸阅读

- [BentoML — 多云和跨区域推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV-cache 复用。
- [TianPan — 多区域 LLM 服务缓存局部性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — 缓存感知路由器源码。
