# 视频-语言模型：时间 Token 与时间定位

> 视频不是照片堆叠。5 秒片段有因果顺序、动作动词和事件时间，图像模型无法表示。Video-LLaMA (Zhang 等人, 2023 年 6 月) 发货了第一个带音频-视觉定位的开放视频-LLM。VideoChat 和 Video-LLaVA 扩展了模式。到 2025 年 Qwen2.5-VL 的 TMRoPE 用前沿专有模型关闭了差距。每个系统不同地解决了时间 token——每片段 Q-former、每帧 concat-pool、每 token TMRoPE。本课阅读模式，构建统一 vs 动态帧采样器，并在时间定位任务上评估。

**类型：** Build
**语言：** Python (stdlib, frame sampler + temporal-grounding evaluator)
**前置知识：** Phase 12 · 08 (LLaVA-OneVision)
**时间：** ~180 分钟

## 学习目标

- 解释为什么时间位置编码独立于视觉编码器改变视频 VLM 性能。
- 对比统一、动态-FPS 和事件驱动帧采样在每秒 token 数 vs 定位准确率上。
- 描述每片段 Q-former (Video-LLaMA) vs 每帧 pooled (Video-LLaVA) vs 每 token M-RoPE (Qwen2.5-VL) 设计。
- 说出四个视频基准：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题

1 分钟视频在 30 FPS 是 1800 帧。ViT-B 在 224 下每帧 196 视觉 token，即 352k token——大于任何 2024 年 LLM 上下文。

三种缩减策略存在：

1. 子采样帧（1-8 FPS 取决于内容）。
2. 激进 pool 每帧的 patch token（3x3 或 4x4 双线性 pool）。
3. 通过 Q-former 压缩，取 16 帧片段输出 64 token。

每个权衡不同。子采样丢失时间细节。Pooling 丢失空间细节。Q-former 两者都丢一点但节省 token。

时间位置编码是另一个轴：模型如何知道帧 5 在帧 6 之前？选项包括简单 1D 时间 RoPE (Video-LLaMA)、学习时间 embedding (Video-LLaVA) 和 TMRoPE (Qwen2.5-VL, 完整 3D)。

## 概念

### Video-LLaMA：每片段 Q-former + 音频分支

Video-LLaMA (2023) 是第一个开放视频-LLM。架构：

- 16 帧片段在 2 FPS（所以 8 秒）。
- 每帧 ViT 特征 -> Video Q-former 跨所有 16 帧 cross-attend -> 32 学习 query -> LLM。
- 并行音频分支：波形 -> ImageBind 音频编码器 -> Audio Q-former -> 32 query -> LLM。

优势：音频-视觉联合推理。劣势：固定片段长度，无任意时间定位。

### VideoChat 和 Video-LLaVA

VideoChat 保留 Video-LLaMA 想法但去掉音频并简化。Video-LLaVA (Lin 等人, 2023) 在图像和 video frame 上训练单个视觉编码器（"投影前对齐"），给出统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

都不处理长视频。两者都是 8-16 帧系统。

### Qwen2.5-VL 和 TMRoPE

Qwen2.5-VL 引入 TMRoPE——Temporal-Modality Rotary Position Embedding。每个 patch token 携带 (t, h, w) 位置，其中 t 是实际时间戳（不是帧索引）。

与简单时间 embedding 的关键差异：

- 绝对时间，不是索引。模型看到"在 4.2 秒"不是"在帧 15。"
- 每 token 旋转，不是每片段。每个视觉 token 独立按其时间戳旋转。
- 与动态 FPS 兼容。如果你在这里 2 FPS 采样、那里 4 FPS，TMRoPE 原生处理不均匀间隔。

TMRoPE 启用"猫在何时跳？"查询。模型可以输出"在 4.2 秒。"Video-LLaMA 只能说"片段早期。"

### 帧采样策略

统一：在持续时间上均匀采样 N 帧。简单，丢失运动峰值。

动态 FPS：基于运动强度自适应采样。光流或帧差分挑选高运动段进行更密采样。Qwen2.5-VL 在这个上训练。

事件驱动：运行轻量检测器，在动作发生处采样更多。VideoAgent 使用。

关键帧 + 上下文：在镜头边界采样 + 几个相邻帧。用于电影内容。

### 每帧 pooling

1 FPS 和每帧 576 token 下，5 分钟片段是 172,800 token。Qwen2.5-VL-72B 的 128k 上下文可做，紧张。

3x3 双线性 pool 缩减到每帧 64 token -> 5 分钟 19,200 token。大多数任务的最佳点。

更激进 pool（6x6 -> 每帧 16 token）用于 agent 工作流，其中空间细节较不重要。

### 四个视频基准

- VideoMME：综合视频理解，短 + 中 + 长。
- TempCompass：细粒度时间推理，"之前"/"之后"问题。
- EgoSchema：长程第一人称视频。
- Video-MMMU：多模态多学科视频问题。

完整视频-VLM 评估命中所有四个。它们压力不同轴——TempCompass 全是关于排序，EgoSchema 关于 3+ 分钟推理，VideoMME 跨持续时间。

### 定位输出格式

时间定位的输出格式：

- 自由文本："猫在大约 4 秒标记处跳。"易于解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练这个。
- Token-based：特殊 `<time>4.1</time>` token 与答案交错。Qwen2.5-VL 的内部格式。

Token-based 对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式直接解析。

### 2026 年最佳实践

2026 年视频 VLM：

- 编码器：SigLIP 2 带 M-RoPE 或 TMRoPE (Qwen2.5-VL)。
- 帧采样：动态 FPS（1-4 取决于运动）带最大帧上限。
- 每帧 pooling：3x3 双线性。
- 输出：带时间 + 事件字段的结构化 JSON。
- 基准：VideoMME + TempCompass 用于通用；EgoSchema 用于长程。

## 使用它

`code/main.py` 包括：

- 统一和动态-FPS 帧采样器。
- 玩具时间定位评估器：给定"地面真实"事件在时间 T 和模型输出，用容差评分准确率。
- 跨 Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）的对比。

## 交付它

本课产生 `outputs/skill-video-vlm-frame-planner.md`。给定视频任务（监控、动作识别、时间定位、摘要），它挑选帧采样器、pooling 因子、输出格式和预期准确率层级。

## 练习

1. 对于 3 分钟烹饪演示，挑选统一 vs 动态 FPS。用 token 计数证明。

2. TMRoPE 添加了简单时间 embedding 表无法做的什么具体东西？

3. 为 VLM 可以学习发出的时间定位写 JSON schema。包含错误案例。

4. 阅读 Video-LLaVA 的 Section 3 关于"Alignment Before Projection。"为什么这比训练单独的图像和视频编码器更好？

5. 给定 VideoMME 排行榜，2026 年顶级开放模型与顶级专有模型之间的差距是多少？多少差距可归因于时间编码 vs 基础 LLM 规模？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Temporal grounding | "Time-localized answers" | VLM 输出事件发生的时间戳范围 |
| TMRoPE | "Time-Multimodal RoPE" | 带绝对时间戳的 3D 旋转位置，由 Qwen2.5-VL 使用 |
| Dynamic FPS | "Motion-aware sampling" | 在高运动段采样更多帧，在静态处更少 |
| Frame pooling | "Spatial compress per frame" | 在 LLM 前用双线性插值减少每帧 patch |
| Video Q-former | "Clip compressor" | Cross-attention 瓶颈，将 N 帧映射到 K 学习 query |
| VideoMME | "Video bench" | 综合短/中/长视频基准，2500+ 样本 |

## 延伸阅读

- [Zhang 等人 — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li 等人 — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin 等人 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin 等人 — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
