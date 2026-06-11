# EAGLE-3 生产环境投机解码

> 投机解码（speculative decoding）将快速草稿模型与目标模型配对。草稿提出 K 个 token；目标模型在一次前向传播中验证；被接受的 token 是免费的。2026 年，EAGLE-3 是生产级变体——它在目标模型的隐藏状态上训练草稿头而非原始 token，将接受率 alpha 推至通用对话的 0.6-0.8 区间。正确的问题不是"草稿有多快"，而是"我的流量上 alpha 是多少？"如果 alpha 低于约 0.55，在高并发下投机解码会是净负收益，因为每次被拒绝的草稿都要付出第二次目标前向传播的代价。本课教你先测量 alpha，再开启 flag。

**类型：** 学习
**语言：** Python（标准库，玩具级接受率模拟器）
**前置知识：** 第 17 阶段 · 04（vLLM 推理服务内部原理）、第 10 阶段 · 18（多 token 预测）
**时间：** ~60 分钟

## 学习目标

- 说出投机解码的三代演进，并解释 EAGLE-3 相比 EAGLE-2 和经典草稿模型改变了什么。
- 定义接受率 alpha，从 alpha 和 K（草稿长度）计算预期加速比，并识别目标并发下的盈亏平衡 alpha。
- 解释为什么投机解码在 2026 年 vLLM 中是可选开启（非默认），以及不测量 alpha 就开启为什么是生产反模式。
- 撰写测量计划：哪个基准、哪个 prompt 分布、哪个并发点、哪个指标作为门槛。

## 问题

解码（decode）是内存受限的。H100 上运行 Llama 3.3 70B FP8 时，每个解码 token 读取约 140 GB/s 的权重并输出一个 token。GPU 计算在解码期间几乎空闲——瓶颈是 HBM 带宽，而非矩阵乘法吞吐量。

投机解码利用了这个差距。用便宜的草稿模型生成 K 个候选 token，然后让目标模型在一次前向传播中验证全部 K 个。每个被验证的 token 实际上是免费的（摊销到目标模型本来就要做的 K 个 batch 前向中）。

经典草稿模型方法使用同一家族的小模型（Llama 3.2 1B 为 Llama 3.3 70B 打草稿）。它能工作，但接受率一般——小模型的分布与目标模型偏离。EAGLE、EAGLE-2、EAGLE-3 直接在目标模型的内部状态上训练轻量草稿头，因此草稿的分布更紧密地跟踪目标模型。这就是 alpha 从草稿模型的 0.4 提升到 EAGLE-3 的 0.6-0.8 的原因。

陷阱：EAGLE-3 在 2026 年 vLLM 中是可选开启。必须显式设置 `speculative_config`。没有 flag，就没有加速。不测量真实流量 alpha 就开启的团队往往看到尾部延迟变差，而非变好。

## 概念

### 投机解码实际带来什么

没有投机解码时，每 token 成本是一次目标前向。有投机解码时，草稿长度 K、接受率 alpha，预期每目标前向的 token 数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证开销。K=5、alpha=0.7 时：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。实际数字集中在 2-3x，因为生产流量上 alpha 很少那么高，且 epsilon 在大 batch size 下增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会消失——它们强制对第一个被拒绝的 token 进行第二次目标前向。在 alpha 降到 0.4 的工作负载上，你要支付草稿开销加验证加重掷。在高并发（比如 256 并发）下，解码 batch 已经足够大，"目标模型单独"和"目标模型加验证"之间的内存带宽差距缩小。在大多数 2026 年硬件上，alpha 低于 0.55 时投机解码是净负收益。

Alpha 因工作负载而异。ShareGPT 风格通用对话上，在 ShareGPT 上训练的 EAGLE-3 达到 0.6-0.8。在领域特定流量（代码、医疗、法律）上，在通用数据上训练的草稿头降到 0.4-0.6。训练领域特定草稿头可以恢复 alpha——与目标微调相比，这是轻量、快速的训练任务。

### EAGLE 三代一览

- **经典草稿模型**：同一家族的小模型。Alpha 0.3-0.5。基础设施简单——加载两个模型，草稿每目标前向运行 K 次前向。
- **EAGLE-1 (2024)**：在目标隐藏状态（最后一层）上训练的单草稿头。Alpha ~0.5-0.6。在目标之上的少量参数开销。
- **EAGLE-2 (2025)**：自适应草稿长度和树形草稿（一次目标传播验证多个分支）。Alpha ~0.6-0.7。更复杂的草稿调度器。
- **EAGLE-3 (2025-2026)**：在多个目标层（不仅是最后一层）上训练草稿头，对齐更好。通用对话上 alpha ~0.6-0.8。

### 2026 年生产配方

1. 先部署纯目标模型。测量目标并发下的基线 TTFT、ITL、吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重新运行基准测试。
3. 记录接受率 alpha。vLLM V1 将其报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的草稿长度得到 alpha。
4. 如果生产流量分布上 alpha < 0.55，禁用投机解码或训练领域特定 EAGLE-3 草稿。
5. 在生产并发下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

平均 ITL 随投机解码下降。如果不调优，P99 可能变差。被拒绝的草稿触发两阶段序列（草稿 + 验证失败 + 重掷）。在满 batch 下，这两阶段串行执行。关注 P99 ITL，而非 P50。

### EAGLE-3 已部署场景

Google 在 2025 年 AI Overviews 中部署了投机解码（相同质量，更快响应）。vLLM V1 将 `speculative_config` 作为文档接口；V1 中的 N-gram GPU 投机解码是与 chunked prefill 兼容的变体。SGLang 支持 EAGLE-3 作为前缀重负载工作负载的推荐草稿路径。

### 盈亏平衡数学（一行）

预期加速比：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。设 `S = 1` 解 alpha：`alpha_breakeven = verify_overhead / K`。典型 verify_overhead ~0.15、K=5 时：`alpha_breakeven = 0.03`。但这是原始解码数学。高并发下验证开销上升，且解码 batch 已经跨序列摊销内存读取，因此有效 alpha_breakeven 在实践中升至 ~0.45-0.55。

### 何时不使用投机解码

- 延迟不重要的 batch-1 离线生成。使用纯目标模型。
- 非常短的输出（低于 50 token）。草稿开销和验证成本占主导。
- 没有领域训练草稿头的专业领域。Alpha 太低。
- vLLM v0.18.0 加草稿模型投机解码加 `--enable-chunked-prefill`。此组合无法编译。文档记录的例外是 V1 中的 N-gram GPU 投机解码。

## 使用它

`code/main.py` 模拟了解码循环，对比有无投机解码在多种 alpha 值和草稿长度 K 下的表现。打印盈亏平衡 alpha、测量加速比和尾部行为。在多个 (alpha, K) 组合上运行，看看投机解码何时停止收益。

## 交付它

本课产出 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，产出分阶段 EAGLE-3 上线计划——基线基准测试、启用配置、测量 alpha、以 alpha >= 0.55 为门槛、关注 P99 ITL。

## 练习

1. 运行 `code/main.py`。K=5 时，2x 加速需要什么 alpha？3x 呢？对 verify_overhead 有多敏感？
2. 假设生产流量 70% 通用对话、30% 代码。通用对话在 ShareGPT 训练的 EAGLE-3 上 alpha 0.7；代码上 alpha 0.4。混合 alpha 是多少？投机解码净正收益吗？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram）以及哪种与 chunked prefill 兼容。
4. 启用 EAGLE-3 后平均 ITL 下降 25% 但 P99 ITL 上升 15%。诊断并提出缓解措施。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头内存成本。与运行 Llama 3.2 1B 作为经典草稿相比如何？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 投机解码 | "草稿加验证" | 用便宜模型提出 K 个 token，目标模型一次前向验证全部 K 个 |
| 接受率 alpha | "spec 接受率" | 草稿 token 被目标接受的比例；唯一重要的指标 |
| 草稿长度 K | "spec k" | 草稿每目标前向提出的 token 数；典型 4-8 |
| 验证开销 epsilon | "spec 开销" | 验证加重掷相比纯目标前向的额外成本；随 batch 增长 |
| EAGLE-3 | "最新 EAGLE" | 2025-2026 变体；在多个目标层上训练草稿头；通用对话 alpha 0.6-0.8 |
| `speculative_config` | "vLLM spec 配置" | vLLM V1 中的显式 opt-in；没有默认配置意味着没有加速 |
| N-gram 投机解码 | "N-gram 草稿" | GPU 端使用 prompt 中 N-gram 查找的草稿；与 chunked prefill 兼容 |
| 盈亏平衡 alpha | "无操作 alpha" | 投机解码零加速时的 alpha；在生产并发下关注此值 |
| 被拒绝草稿两阶段 | "重掷成本" | 草稿拒绝时的两次目标前向；驱动 P99 尾部 |

## 延伸阅读

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) —— `speculative_config` 和 V1 中 chunked-prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) —— 精确字段集。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) —— 原始 EAGLE 草稿头公式。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) —— 自适应草稿和树。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) —— 带投机解码的高效 LLM 系统。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) —— 生产上线检查清单。
