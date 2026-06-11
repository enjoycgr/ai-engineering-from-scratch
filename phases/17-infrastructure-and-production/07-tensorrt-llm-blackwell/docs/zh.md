# TensorRT-LLM on Blackwell with FP8 and NVFP4

> TensorRT-LLM is NVIDIA-only but it wins on Blackwell. On GB200 NVL72 with Dynamo orchestration, SemiAnalysis InferenceX measured $0.012 per million tokens on a 120B model in Q1-Q2 2026, against $0.09/M on H100 + vLLM — a 7x economic gap. The stack is three floating-point regimes compounded: FP8 stays critical for KV cache and attention kernels because it has the dynamic range they need; NVFP4 (4-bit microscaling) handles weights and activations; multi-token prediction (MTP) and disaggregated prefill/decode add another 2-3x on top. Day-0 model support loads FP4 weights directly without post-training conversion. The catch for 2026 engineering teams: TRT-LLM is a closed NVIDIA stack, so adopting it trades portability for throughput. Run the math on your mix of models and hardware before committing.

**类型：** 学习
**语言：** Python（标准库，玩具级 FP8/NVFP4 内存与成本计算器）
**前置知识：** 第 17 阶段 · 04（vLLM 推理服务内部原理）、第 10 阶段 · 13（量化）
**时间：** ~75 分钟

## 学习目标

- 解释为什么即使权重使用 NVFP4，KV cache 和 attention 仍需要 FP8。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省来自哪里。
- 说出 TRT-LLM 利用的 Blackwell 特有特性（day-0 FP4、MTP、分离式服务、all-to-all 原语）。
- 判断何时 TRT-LLM 的 NVIDIA 锁定值得 7 倍成本差距 vs Hopper 上的 vLLM。

## 问题

2026 年推理经济学的前沿是"每美元多少 token"。答案取决于四个叠加选择：硬件代际（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、推理引擎（vLLM vs SGLang vs TRT-LLM）、编排（纯推理 vs 分离式 vs Dynamo）。

在 Hopper 配 vLLM 上，120B MoE 约 $0.09/百万 token。在 Blackwell 配 TRT-LLM + Dynamo 上，同模型约 $0.012 —— 便宜 7 倍。部分差距来自硬件（Blackwell 每 GPU LLM 吞吐量是 Hopper 的 11-15 倍）。部分来自栈：FP4 权重、MTP 草稿、分离式 prefill/decode、以及用于 MoE 专家通信的 NVLink 5 all-to-all。

你无法在 NVIDIA 栈之外复现这一点。这就是权衡 —— 可移植性换经济学。理解哪些栈选择贡献了差距的哪一部分，是本课的重点。

## 概念

### 为什么 FP8 仍是 KV cache 的底线

2026 年的常见错误：假设 NVFP4 适用于所有地方。并非如此。KV cache 需要 FP8（8 位浮点），因为它存储的 attention key 和 value 跨越宽动态范围。将 KV 量化到 FP4 会导致灾难性精度损失 —— 分布尾部衰减，attention score 崩塌。FP8 的指数位给 KV cache 提供了所需的范围。

NVFP4（2025-2026）适用于权重和激活。微缩放：每块权重有自己的缩放因子，因此小块可以跨越不同动态范围而不损失每张量缩放。对于激活，FP4 可以胜任，因为激活在层内范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV cache：FP8。
- Attention 累加器：FP32（softmax 稳定性）。

### TRT-LLM 利用的 Blackwell 特有原语

- **Day-0 FP4 权重**：模型提供商直接发布 FP4 权重；TRT-LLM 无需训练后转换即可加载。FP4 无需 AWQ / GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE（第 17 阶段 · 05）相同理念，但集成到 TRT-LLM 构建中。
- **分离式服务**：prefill 和 decode 在独立 GPU 池上，KV cache 通过 NVLink 或 InfiniBand 传输。与 Dynamo（第 17 阶段 · 20）理念相同。
- **All-to-all 通信原语**：NVLink 5 将 MoE 专家通信延迟比 Hopper 降低 3 倍。TRT-LLM 的 MoE kernel 为此调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Core 上的硬件加速缩放因子处理。

### 你应该记住的数字

- HGX B200 通过 TRT-LLM 在 GPT-OSS-120B 上达到 $0.02/M token。
- GB200 NVL72 通过 Dynamo（编排 TRT-LLM）达到 $0.012/M token。
- H100 + vLLM 在可比工作负载上约 $0.09/M token。
- TRT-LLM 三个月更新带来 2.8 倍吞吐量提升（2026 年）。
- Blackwell vs Hopper 每 GPU LLM 吞吐量 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 主导每个提交任务。

### FP4 在质量上的实际代价

NVFP4 很激进。在推理密集型工作负载（思维链、数学、长上下文代码生成）上，FP4 权重明显退化。逐块校准可缓解但无法消除。部署推理模型的团队通常使用 FP8 权重 + FP4 激活作为折中，或全程使用 H200 配 FP8。

规则：在承诺 NVFP4 权重前，始终在你的评估集上验证任务质量。

### 为什么这是 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源 kernel。模型需要为特定 GPU SKU 编译。不支持 AMD、Intel、ARM。如果你的基础设施策略是多供应商，TRT-LLM 对 TRT-LLM 服务层来说是死路 —— 你仍可在混合硬件上用 vLLM 服务。如果你是纯 NVIDIA，7 倍差距足以支付锁定的代价。

### 2026 年实用配方

对于年推理账单超过 $1 亿，在 Hopper + vLLM 上运行会留下 7-10 倍未挖掘潜力。将成本主导工作负载迁移到 Blackwell + TRT-LLM + Dynamo。在 H100 + vLLM 上保留实验层以加速模型迭代。在生产前验证每个 NVFP4 转换模型的质量。

### 分离式奖励

TRT-LLM 的分离式服务（独立的 prefill 和 decode 池）在第 17 阶段 · 20 中深入覆盖。在 Blackwell 上，乘数叠加：FP4 权重 × MTP 加速 × 分离式放置 × 缓存感知路由。7 倍数字假设完整栈。

## 使用它

`code/main.py` 计算模型在三个栈上的 HBM 占用、decode 吞吐量（内存受限 regime）和 $/M-token：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它以查看复合效应和每项改变贡献的差距份额。

## 交付它

本课产出 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年 token 量，决定 Blackwell + TRT-LLM 栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。在 120B MoE 上，30% 活跃参数，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上的内存带宽受限 decode 吞吐量。最大跳跃来自哪里？
2. 客户在 H100 + vLLM 上年花费 $200 万。给定 7 倍经济差距，他们需要购买多少 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. NVFP4 权重转换后 MATH 精度下降 3 个百分点。说出两条恢复路径：一条质量优先（保留 FP8 权重），一条成本优先（用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell-over-Hopper 差距最小，为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV cache、128k 上下文下的 HBM 需求。它能放入单个 GB200 NVL72 节点吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| FP8 | "八位浮点" | 8 位浮点；因动态范围用于 KV cache 和 attention |
| NVFP4 | "四位微缩放" | NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上的权重和激活 |
| MXFP8 | "MX 八位" | 微缩放 FP8 变体；Blackwell Tensor Core 硬件加速 |
| Day-0 FP4 | "发布 FP4 权重" | 模型提供商直接发布 FP4 权重；无需训练后转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 集成的投机解码草稿（第 17 阶段 · 05） |
| 分离式服务 | "拆分 prefill/decode" | Prefill 和 decode 在独立 GPU 池上；KV 通过 NVLink/IB 传输 |
| All-to-all | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年行业接受的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) —— 2026 年 4 月 MLPerf 结果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) —— NVLink 5 all-to-all 和 MoE kernel。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) —— 官方引擎文档。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) —— TRT-LLM 之上的分离式编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) —— 发布 Blackwell 数字的基准套件。
