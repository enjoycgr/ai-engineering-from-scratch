---
name: nsa-integrator
description: 长上下文预训练运行中 Native Sparse Attention 的集成计划。
version: 1.0.0
phase: 10
lesson: 17
tags: [nsa, sparse-attention, long-context, pre-training, kernel-aligned, deepseek]
---

给定长上下文预训练运行规范（目标上下文、基础架构、可用训练 token、GPU 拓扑、部署目标），生成 NSA 集成计划。

产出：

1. 压缩块大小 `l`。选择 32、64 或 128。根据目标上下文证明合理性：`l = 32` 用于 16k-32k，`l = 64` 用于 64k-128k，`l = 128` 用于 256k 以上。更大的 `l` 意味着更少的压缩键但更粗的路由信号。
2. Top-k 选择计数。在 8 到 32 之间选择。论文默认是 16。根据目标任务混合证明合理性：推理密集型任务（数学、代码）受益于更高的 `k`，因为选择精度更重要。检索密集型任务在较低的 `k` 下工作。
3. 滑动窗口 `W`。选择 256、512 或 1024。默认 512。对于结构化内容（代码）本地上下文足够时更短；对于散文更长。
4. 门控 MLP。指定宽度和初始化。默认：从 `hidden` 到 3 的线性层，使用 `sigmoid` 或 `softplus` activation（激活函数）。如果门控权重崩溃为偏向一个分支则警告——这表明 `l`、`k` 或 `W` 调谐不当。
5. Kernel 选择。确认目标加速器的 Triton 或 CUDA kernel 可用性。拒绝在推理时回退到密集 attention（NSA 的全部意义在于节省解码计算）。如果只有前向 kernel 而没有后向 kernel，拒绝预训练并推荐在现有密集检查点上继续训练。

硬性拒绝：
- 在没有持续预训练的情况下对密集 attention 预训练的模型使用 NSA。无法在推理时 bolt on。
- 目标上下文低于 16k。三分支开销主导节省。
- 在没有 NSA kernel 支持的栈上进行仅推理部署。推荐 MLA 或滑动窗口 attention 替代。

拒绝规则：
- 如果没有长上下文评估数据（RULER、LongBench、needle-in-haystack），拒绝并首先请求校准数据。
- 如果训练数据上下文分布以短序列为主，拒绝并推荐在集成 NSA 之前重新加权数据。
- 如果加速器比 A100 旧，拒绝——NSA 的 kernel 优势假设 H100/H200/MI300 内存层次结构。

输出：一页集成计划，列出 `l`、`k`、`W`、门控配置、kernel 路径和目标上下文下的预期计算节省。最后以"成功标准"段落结束：证明保留 NSA 的特定 RULER 或 LongBench 数字（与匹配的密集 attention 基线相比的百分点）。包括回退触发器——架构应回退到 MLA 或密集 GQA 的指标阈值。
