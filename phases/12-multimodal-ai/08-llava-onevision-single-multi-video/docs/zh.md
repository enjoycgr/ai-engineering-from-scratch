# LLaVA-OneVision：单图像、多图像、视频合一模型

> 在 LLaVA-OneVision (Li 等人, 2024 年 8 月) 之前，开源 VLM 世界有独立谱系：LLaVA-1.5 用于单图像，Mantis 和 VILA 等多图像模型，Video-LLaVA 和 Video-LLaMA 等视频模型。每个在自己的基准上获胜，在其他上失败。LLaVA-OneVision 论证单个课程可以训练一个模型主导所有三种场景，且涌现的任务迁移效应（单图像技能导出到视频，多图像推理导出到单图像）击败专家之和。配方惊人地简单：一个跨场景恒定的视觉 token 预算，加一个从单图像到 OneVision（多图像）再到视频的显式课程。本课阅读预算、课程和涌现行为。

**类型：** Build
**语言：** Python (stdlib, token budget solver + curriculum planner)
**前置知识：** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**时间：** ~180 分钟

## 学习目标

- 设计一个跨单图像、多图像和视频输入保持恒定的视觉 token 预算。
- 排序一个从单图像到视频的技能迁移训练课程，无灾难性遗忘。
- 解释为什么课程做对时，单模型在相同参数量下击败专家。
- 说出 LLaVA-OneVision 报告的三个涌现能力：多摄像头推理、set-of-mark prompting、iPhone-screenshot agent。

## 问题

图像、多图像和视频各自以不同方式压力模型。

单图像想要高分辨率 token（AnyRes，约 2880 视觉 token）来捕捉 OCR 和细粒度细节。每样本预算：一张图像，2880 token。

多图像想要几张中等分辨率图像（每张约 576 token），使跨图像推理能放入上下文。每样本预算：4-8 张图像，每张 576，2300-4600 token。

视频想要多帧低分辨率（pooling 后每帧约 196 token）来捕捉时序动态。每样本预算：8-32 帧，每帧 196，1600-6200 token。

如果你训练单独模型，你选一个预算。如果你训练一个模型，你需要预算跨场景合理扩展而不爆炸上下文。

OneVision 之前，默认答案是"训练一个场景，忽略其他。"Video-LLaVA 用额外训练阶段把视频改装到图像模型上。LLaVA-NeXT 用 tiling 添加多图像支持。没有一个干净处理所有三个。

## 概念

### OneVision token 预算

LLaVA-OneVision 挑选约 3000-4000 token 每样本的统一视觉 token 预算，按场景不同分配：

- 单图像：AnyRes-9 (3x3 瓦片 + 缩略图)，每瓦片 384 配 729 patch，激进双线性 pooling 2x2 → 每瓦片 182。总计：9 * 182 + 182 = 1820 token。或 AnyRes-4 每瓦片 729 = 2916 + 729。
- 多图像：每张图像中等分辨率（384，无 tiling），无 pooling 的 729 token。预算 6 张图像 → 4374 token。
- 视频：32 帧在 384 分辨率，激进 3x3 双线性 pool → 每帧 81 token。总计：32 * 81 = 2592 token。

分配保持大致恒定的总 token。LLM 从不会看到爆炸其上下文的 batch。编码器每场景产生不同几何，但 LLM 消费相同预算。

### 三阶段课程

LLaVA-OneVision 训练分三阶段：

1. 单图像 SFT（阶段 SI）。所有数据是单图像+文本。在 AnyRes 高分辨率输入上训练。这教授感知、OCR 和细粒度理解。使用 LLaVA-NeXT 数据加 OneVision 特定单图像数据。
2. OneVision SFT（阶段 OV）。混合单图像 + 多图像 + 视频（均匀采样帧）。在统一 token 预算上训练。这教授模型处理异构 batch 形状。无权重重置——从阶段 SI 继续。
3. 任务迁移（阶段 TT）。继续目标任务混合，通常按产品侧重多图像或视频。可选部署微调。

关键：课程顺序重要。先训练视频或多图像比先训练单图像产生更差的图像性能，即使数据相同。论文显式 ablated 这一点。

### 为什么课程有效

单图像训练构建感知基础。Patch token 携带细粒度视觉特征；LLM 学习整合它们与文本。多图像和视频引入结构挑战（哪张图像是哪个、先发生什么），没有强感知基础很难学习。

如果你从零开始一起训练所有场景，模型在感知上欠拟合（每 batch 单图像数据有限）并在结构上过拟合（大量多图像/视频数据）。结果：一个模型遵循跨图像推理模式但视觉上浅。

课程排序给你阶段 SI 的感知强度，然后阶段 OV 的组合/时序推理，不丢失任何一方。

### 跨场景涌现技能

LLaVA-OneVision 论文报告三个涌现能力：

1. 多摄像头推理。在 multi-image + 视频上分别训练；推理时，要求推理多摄像头驾驶场景。模型正确整合视图，尽管训练中从未见过该确切格式。
2. Set-of-mark prompting。用户在图像中给物体编号标注；模型推理"标记 3 相对于标记 7 在做什么"。训练中既没标记也没标注；从空间定位 + multi-image reference 的组合学习。
3. iPhone-screenshot agent。用户提供 iPhone 屏幕截屏并要求规划下一次点击。在 UI 截屏、用户工作流视频和 multi-image 前后对上训练。泛化到 agent 用例。

这些不是训练任务；它们从课程的组合结构中涌现。

### 视觉 token pooling

Token 预算需要 pooling。OneVision 在 2D patch 网格上使用双线性插值：24x24 = 576 patch 变成 12x12 = 144（2x 因子）或 8x8 = 64（3x 因子）。Pooling 在 patch-grid 空间完成，而非 token 空间，以保留局部性。

每场景 pooling 因子选择本身是一个超参数。更少 pooling = 更多 token = 更丰富表示。更多 pooling = 更少 token = 更多帧/图像能放入。

### LLaVA-OneVision-1.5

2025 后续（LLaVA-OneVision-1.5, arXiv 2509.23661）在训练数据、模型权重和代码上"完全开放"。在某些基准上匹配专有 gap 并民主化配方。相同课程，更多数据，更好基础 LLM。无架构改变。

### 与 Qwen2.5-VL 对比

Qwen2.5-VL (Lesson 12.09) 做不同选择。它使用 M-RoPE 和动态 FPS 替代固定 pooling。其预算随输入缩放——1 分钟视频比 5 秒视频用更多 token。LLaVA-OneVision 固定预算并缩放 pooling。两者都有效；它们交易可配置性换可预测性。

## 使用它

`code/main.py` 是一个 OneVision 风格 VLM 的课程和预算规划器。给定每样本 token 预算和目标场景混合（如 40% 单图像、30% 多图像、30% 视频），它：

- 分配每场景的分辨率、pooling 因子和帧数。
- 检查每场景是否适配共享预算。
- 报告预期 token 计数、LLM FLOPs 和哪些场景 token 不足。
- 打印逐阶段训练计划。

用它来规划 OneVision 微调或 sanity-check VLM 部署的每请求成本。

## 交付它

本课产生 `outputs/skill-onevision-budget-planner.md`。给定目标任务分布和每样本预算，它发出 AnyRes 因子、每帧 pooling、视频帧数和课程阶段权重。每当你训练或微调统一场景 VLM 时使用。

## 练习

1. 你的产品支持 80% 单图像、10% 多图像（2-4 张）、10% 视频（8-16 帧）。设计 token 预算。你不做 heavy multi-image 节省的额外预算放在哪里？

2. 阅读 LLaVA-OneVision Section 4.3（涌现能力）。提出课程可能解锁但论文未报告的第四个涌现技能。

3. 交换课程顺序——先训练 multi-image，然后单图像，然后视频。预测哪些基准退化及为什么。

4. 论文报告仅在每样本 8 帧上训练的视频基准。推理时泛化到 30 秒视频吗？什么先崩溃——token 预算还是时序推理？

5. 24x24 patch 双线性 pooling 到 12x12 是每维 4x 缩减。在 stdlib Python 中实现 pooling 并验证每个 2x2 block 的均值匹配双线性输出。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| OneVision scenario | "单图像、多图像或视频" | 统一 VLM 处理的三种输入形状之一；预算跨场景保持恒定 |
| Token budget | "每样本多少 token" | LLM 每训练/推理样本看到的总视觉 token，通常 3000-4000 |
| Curriculum | "训练顺序" | 阶段排序（单图像 → 多图像 → 视频）选择用于涌现迁移 |
| Bilinear pooling | "Token shrink" | 对 patch 网格（2D）应用双线性插值以减少 token 数同时保留局部性 |
| Emergent skill | "未训练仍有效" | 推理时出现的能力，没有匹配训练数据，由于课程组合 |
| AnyRes-k | "k-tile 设置" | k 个固定分辨率子瓦片加一个缩略图，典型 k ∈ {4, 9} |
| Task transfer | "跨场景泛化" | 单图像上学习的技能应用到视频（反之亦然），通过共享主干 |

## 延伸阅读

- [Li 等人 — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin 等人 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin 等人 — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang 等人 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
