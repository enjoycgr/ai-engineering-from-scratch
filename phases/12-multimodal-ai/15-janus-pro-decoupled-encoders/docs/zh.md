# Janus-Pro：统一多模态模型的解耦编码器

> 统一多模态模型有不可避免的张力。理解需要语义特征——SigLIP 或 DINOv2 输出富含概念级信息的向量。生成需要重建友好代码——组合回清晰像素的 VQ token。两个目标在单个编码器中不兼容。Janus (DeepSeek, 2024 年 10 月) 和 Janus-Pro (DeepSeek, 2025 年 1 月) 论证修复是停止尝试：解耦两个编码器。在任务间共享 transformer body，但通过 SigLIP 路由理解、通过 VQ tokenizer 路由生成。7B 下，Janus-Pro 在 GenEval 上击败 DALL-E 3 同时在 MMMU 上匹配 LLaVA。本课阅读为什么两个编码器在单个失败的地方工作。

**类型：** Build
**语言：** Python (stdlib, dual-encoder routing + shared-body signal)
**前置知识：** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**时间：** ~120 分钟

## 学习目标

- 解释为什么单个共享编码器在理解或生成质量上妥协。
- 描述 Janus-Pro 的路由：理解输入侧用 SigLIP 特征，生成输入输出都用 VQ token。
- 追踪让 Janus-Pro 在 Janus 未成功处成功的数据规模扩展。
- 对比解耦（Janus-Pro）、耦合-连续（Transfusion）和耦合-离散（Show-o）架构。

## 问题

统一模型跨理解和生成共享 transformer body。先前尝试（Chameleon、Show-o、Transfusion）都使用一个视觉 tokenizer 做双向。Tokenizer 是妥协：

- 为重建优化（生成）：VQ-VAE 捕捉细粒度像素细节但产生语义连贯弱的 token。
- 为语义优化（理解）：SigLIP embedding 将"猫"图像分组在"猫"token 附近但不允许良好重建。

Show-o 和 Transfusion 在一个方向上支付可见的质量税。Janus-Pro 问道：当任务有不同需求时，为什么要求一个 tokenizer？

## 概念

### 解耦视觉编码

Janus-Pro 架构分离两个编码器：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → transformer body。
- 生成路径。输入图像（如果条件于现有图像）→ VQ tokenizer → token IDs → transformer body。
- 输出生成。Transformer body 预测的图像 token → VQ decoder → 像素。

Transformer body 共享。Body 上游和下游的一切都是任务特定的。

输入由 prompt 格式消歧：`<understand>` 标签通过 SigLIP 路由；`<generate>` 通过 VQ。或路由从任务隐式进行。

### 为什么有效

理解 loss 获得 SigLIP 特征，CLIP 风格预训练已调优语义相似性。模型感知基准优于 Show-o / Transfusion，因为输入特征对任务更好。

生成 loss 获得 VQ token，tokenizer 已调优重建。图像质量优于 Show-o，因为 VQ 代码干净组合回像素。

共享 transformer body 看到两种输入分布（SigLIP 和 VQ）并学习与两者工作。声明：足够数据 + 足够参数，body 吸收切换。

### 数据扩展——Janus vs Janus-Pro

Janus（原始版, arXiv 2410.13848）引入解耦但小规模（1.3B 参数，有限数据）。Janus-Pro (arXiv 2501.17811) 扩展：

- 7B 参数（vs 1.3B）。
- 阶段 1（对齐）9000 万图像-文本对（vs 7200 万）。
- 阶段 2（统一）7200 万（vs 2600 万）。
- 阶段 3 添加 20 万图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs ~58）并在 GenEval 上击败 DALL-E 3（0.80 vs 0.67）。一个开放模型，在统一光谱两侧竞争。

### JanusFlow——rectified flow 变体

JanusFlow (arXiv 2411.07975) 用 rectified-flow 生成路径（连续）交换 VQ 生成路径。分裂变成理解用 SigLIP + 生成用 rectified flow。质量天花板进一步提升。架构保持解耦编码器-共享 body。

### 共享 body 的工作

Transformer body 处理统一序列但有两种输入分布。它的工作是：

- 理解：消费 SigLIP 特征 + 文本 token → 自回归发出文本。
- 生成：消费文本 token +（可选图像 VQ token）→ 自回归发出图像 VQ token。

Body 每 block 无模态特定权重。它是你期望在 Qwen 或 Llama 内部找到的文本风格 transformer，加两个输入 adapter。

有趣的是，这意味着 Janus-Pro 的 body 可从预训练 LLM 初始化。Janus-Pro 从 DeepSeek-MoE-7B 初始化。该选择重要：LLM 贡献纯 from-scratch 统一模型难以达到的推理能力。

### 与 InternVL-U 对比

InternVL-U (Lesson 12.10) 是 2026 后续。它结合：

- 原生多模态预训练（InternVL3 主干）。
- 解耦编码器路由（SigLIP 进，VQ + diffusion head 出）。
- 统一理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入更大框架。解耦编码器想法现在是大规模统一模型的默认。

### 限制

解耦编码器增加架构复杂度。两个 tokenizer 训练、两个输入路径维护、两组失败模式。不需要生成的产品，Janus-Pro 过度工程——选 LLaVA 家族理解模型。

不需要理解的，Janus-Pro 过度胜任——选 Stable Diffusion 3 / Flux 模型。

两者都需要，Janus-Pro 现在是参考开放架构。

## 使用它

`code/main.py` 模拟 Janus-Pro 路由：

- 两个 mock 编码器：SigLIP-like（产生 256-dim 语义向量）和 VQ-like（产生整数代码）。
- 基于任务标签挑选编码器的 prompt router。
- 共享 body（替身），处理无论哪个编码器产生的 token 序列。
- 从阶段 1（对齐）到阶段 3（指令微调）的加权样本 schedule 切换。

为 3 个示例打印路由路径：图像 QA、T2I、图像编辑。

## 交付它

本课产生 `outputs/skill-decoupled-encoder-picker.md`。给定想要前沿级质量统一生成 + 理解的产品，它在 Janus-Pro、JanusFlow 或 InternVL-U 之间挑选并附具体数据规模推荐。

## 练习

1. Janus-Pro-7B 在 GenEval 上击败 DALL-E 3。解释为什么 7B 开放模型能在生成上匹配前沿专有模型但不在理解上。

2. 实现 router 函数：给定 prompt 文本，分类为 `understand` 或 `generate`。如何处理模糊 prompt 如 "describe and then sketch"？

3. JanusFlow 用 rectified flow 替换 VQ 路径。Transformer body 现在输出什么，loss 中什么改变？

4. 提出 Janus-Pro 架构可用一个额外解耦编码器处理的第四个任务。例子：图像分割（DINO 风格）、深度（MiDaS 风格）。

5. 阅读 Janus-Pro Section 4.2 关于数据扩展。哪个数据阶段对 T2I 质量增益 vs Janus 贡献最多？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Decoupled encoding | "Two visual encoders" | 每方向分离的 tokenizer 或编码器：理解用语义，生成用重建 |
| Shared body | "One transformer" | 单个 transformer 处理任一编码器输出；无模态特定权重 |
| SigLIP for understanding | "Semantic features" | CLIP 家族视觉塔提供丰富概念特征但重建差 |
| VQ for generation | "Reconstruction codes" | 向量量化 token 干净解码回像素 |
| JanusFlow | "Rectified-flow variant" | Janus-Pro 用连续 flow-matching 生成 head 替代 VQ |
| Routing tag | "Task tag" | 挑选输入编码器的 prompt 标记（`<understand>` / `<generate>`） |

## 延伸阅读

- [Wu 等人 — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen 等人 — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma 等人 — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong 等人 — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
