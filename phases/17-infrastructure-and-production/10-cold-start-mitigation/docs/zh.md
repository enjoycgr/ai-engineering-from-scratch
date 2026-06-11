# Serverless LLM 冷启动缓解

> 一个 20 GB 的模型镜像从冷状态到可服务需要 5-10 分钟（7B 模型）到 20 分钟以上（70B 模型）。在真正的 serverless 世界中，这不是预热 —— 这是故障。缓解措施在五个层面运作：预置节点镜像（AWS 上的 Bottlerocket、双卷架构）、模型流式加载（NVIDIA Run:ai Model Streamer，vLLM 原生支持）、GPU 内存快照（Modal checkpoints，重启速度提升高达 10 倍）、温池（`min_workers=1`）、分层加载（ServerlessLLM 的 NVMe→DRAM→HBM 流水线，延迟降低 10-200 倍），以及实时迁移（传输输入 token（KB）而非 KV cache（GB））。Modal 公布的冷启动下限为 2-4 秒；Baseten 默认 5-10 秒，预预热可达亚秒级。本课教你如何测量、预算和叠加这五个层面。

**类型:** 学习
**语言:** Python（标准库，玩具冷启动路径模拟器）
**前置知识:** Phase 17 · 02（推理平台经济学），Phase 17 · 03（GPU 自动扩缩容）
**时间:** 约 60 分钟

## 学习目标

- 列举冷启动缓解的五个层面，并在每个层面说出一个工具或模式。
- 计算 70B 模型的总冷启动时间，分解为（节点配置）+（权重下载）+（权重加载到 HBM）+（引擎初始化）。
- 解释为什么实时迁移传输输入 token（KB）而非 KV cache（GB），以及代价是什么（重新计算）。
- 说出温池的权衡（为空闲 GPU 付费或接受冷启动尾部）以及 `min_workers > 0` 成为强制要求的 SLA 阈值。

## 问题背景

你的 serverless LLM 端点夜间缩容到零。早上 8 点流量激增。第一个请求等待：

1. Karpenter 配置 GPU 节点：45-60 秒。
2. 容器拉取 30 GB 含权重的镜像：120-300 秒。
3. 引擎将权重加载到 HBM：45-120 秒，取决于模型大小和存储速度。
4. vLLM 或 TRT-LLM 初始化 CUDA graph、KV cache 池、tokenizer：10-30 秒。

总计：220-510 秒（约 3-8 分钟）才返回第一个 token。你的 SLA 是 2 秒。你部署了温池（`min_workers=1`），问题似乎消失了 —— 但现在你 24×7 为一台空闲 GPU 付费。如果你的服务有 5 个产品，每个有一个温副本，那就是 5 × 24 × 30 = 3,600 GPU 小时/月，无论是否有用户调用。

冷启动缓解是如何在保持 serverless 经济性的同时，近似实现始终在线的延迟。

## 核心概念

### 第一层 —— 预置节点镜像（Bottlerocket）

在 AWS 上，Bottlerocket 的双卷架构将操作系统与数据分离。对你的容器镜像预拉取后的数据卷做快照；在 `EC2NodeClass` 中引用快照 ID。新节点启动时权重已在本地 NVMe 上 —— 第 2 步和第 3 步的部分消失。原生支持 Karpenter。典型节省：大模型每次冷启动 2-4 分钟。

GCP 上的等效方案：预烘焙容器层的自定义 VM 镜像。Azure 上：具有相同模式的托管磁盘快照。

### 第二层 —— 模型流式加载（Run:ai Model Streamer）

不是在回答第一个请求前加载完整文件，而是将权重逐层流式加载到 GPU 内存，并在第一个 transformer block 驻留后立即开始处理。NVIDIA Run:ai Model Streamer 在 2026 年 vLLM 中原生提供。支持 S3、GCS 和本地 NVMe。通过将 I/O 与计算设置重叠，将大模型的权重加载时间大致减半。

### 第三层 —— GPU 内存快照（Modal）

Modal 在首次加载后对 GPU 状态（权重、CUDA graph、KV cache 区域）做 checkpoint。后续重启直接反序列化到 HBM —— 比重新初始化快 10 倍。这最接近"在 2 秒内启动一个温 GPU"。权衡：快照是每 GPU 拓扑的，所以如果 Karpenter 将你迁移到不同的 SKU，你需要重新做 checkpoint。

### 第四层 —— 温池（min_workers=1）

最简单的缓解：保持一个副本始终就绪。成本是一台 GPU 的每小时费率 × 24×7。对于小模型，计算是残酷的（你支付 $0.85-$1.50/小时来避免 30 秒冷启动）；对于大模型则温和（支付 $4/小时来避免 5 分钟冷启动）。温池成为强制要求的 SLA 阈值：70B+ 模型通常 TTFT P99 < 60 秒。

### 第五层 —— 分层加载（ServerlessLLM）

ServerlessLLM 将存储视为层级：NVMe（快但大）、DRAM（中等但分层）、HBM（小但即时）。权重预加载到 DRAM；按需加载到 HBM。论文报告与朴素磁盘到 HBM 相比，冷加载延迟降低 10-200 倍。生产采用尚处早期，但已有与 vLLM 的集成。

### 第六层 —— 实时迁移（ bonus 模式）

当节点不可用（spot 驱逐、节点排空）时，传统模式是冷启动另一个副本并排空请求队列。实时迁移将输入 token（千字节）移动到已加载模型的目标节点，并在目标节点上重新计算 KV cache。重新计算比通过网络传输 GB 级 KV cache 更便宜。适用于分离式部署。

### 温池计算

对于 P99 TTFT SLA 为 2 秒的服务，问题不是"温池是/否"，而是"多少个温副本，以及哪些路径获得它们"。

- 高价值交互路径（实时聊天、语音智能体）：`min_workers=1-2`。
- 后台批处理路径（夜间分类）：接受缩容到零，5-10 分钟冷启动可容忍。
- 高级层：每个租户 `min_workers`，专用容量。

### 优化前测量

70B 模型在全新节点上的冷启动解剖（示意）：

| 阶段 | 时间 | 缓解措施 |
|------|------|---------|
| 节点配置 | 50 秒 | Bottlerocket + 预置镜像、温池 |
| 镜像拉取 | 180 秒 | 预置数据卷（消除） |
| 权重到 HBM | 75 秒 | 模型流式加载（减半）；GPU 快照（消除） |
| 引擎初始化 | 20 秒 | 持久 CUDA graph cache |
| 首次前向 | 3 秒 | 最小固有延迟 |
| **冷启动总计** | **328 秒** | |
| **缓解后总计** | **约 15 秒** | 22 倍降低 |

### 你应该记住的数字

- Modal 冷启动：2-4 秒（使用 GPU 快照）。
- Baseten 默认冷启动：5-10 秒；预预热可达亚秒级。
- 原始 70B 冷启动：3-8 分钟。
- Run:ai Model Streamer：权重加载速度提升约 2 倍。
- ServerlessLLM 分层加载：延迟降低 10-200 倍（论文数据）。

## 动手实践

`code/main.py` 模拟有和没有每种缓解措施的冷启动路径。报告总冷启动时间、温池成本，以及温池自盈亏的请求率临界点。

## 交付成果

本课产出 `outputs/skill-cold-start-planner.md`。给定 SLA、模型大小和流量形态，选择要叠加哪些缓解措施。

## 练习

1. 运行 `code/main.py`。计算温副本比支付冷启动税（通过 SLA 外的额外请求丢弃）更划算的盈亏请求率。
2. 你部署一个 13B 模型，P99 TTFT SLA 为 3 秒。选择实现它的最小缓解层数。
3. Bottlerocket 预置消除了镜像拉取，但权重仍从快照加载到 HBM。如果快照支持的 NVMe 读取速度为 7 GB/s，计算 70B 模型的墙钟时间。
4. 你的 serverless 提供商提供 GPU 快照（Modal），你的团队拒绝因为"快照泄露 PII"。论证双方 —— 现实风险是什么，缓解措施是什么（临时快照、加密、命名空间隔离）？
5. 设计分层温池策略：付费用户、试用用户和批处理工作负载各需要多少个温副本？展示计算过程。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| 冷启动 | "大停顿" | 从请求到新鲜副本首 token 的时间 |
| 温池 | "始终在线最小值" | `min_workers >= 1` 保持至少一个副本就绪 |
| 预置镜像 | "烘焙 AMI" | 容器权重预驻留的节点镜像 |
| Bottlerocket | "AWS 节点 OS" | 支持双卷快照的 AWS 容器优化操作系统 |
| 模型流式加载 | "流式加载" | 将权重 I/O 与计算设置重叠 |
| GPU 快照 | "checkpoint 到 HBM" | 序列化加载后的 GPU 状态；重启时反序列化 |
| 分层加载 | "NVMe + DRAM + HBM" | 存储层级；按需加载 |
| 实时迁移 | "移动 token" | 传输输入（KB），在目标节点重新计算 KV |
| `min_workers` | "温副本" | Serverless 最小保活数量 |
| 缩容到零 | "完全 serverless" | 空闲时无成本；接受完整冷启动税 |

## 延伸阅读

- [Modal — 冷启动性能](https://modal.com/docs/guide/cold-start) — Modal 发布的基准和 checkpoint 架构。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — 预置数据卷快照模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — 将权重加载与计算设置重叠。
- [Baseten — 冷启动缓解](https://www.baseten.co/blog/cold-start-mitigation/) — 预预热手册。
- [ServerlessLLM 论文 (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — 分层加载设计。
- [NVIDIA — Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — 分离式部署的实时迁移。
