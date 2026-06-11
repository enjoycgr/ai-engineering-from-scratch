# Qwen-VL 家族与动态 FPS 视频

> Qwen-VL 家族——Qwen-VL (2023)、Qwen2-VL (2024)、Qwen2.5-VL (2025)、Qwen3-VL (2025)——是 2026 年最具影响力的开源视觉-语言模型谱系。每一代做了一个决定性的架构赌注，开源生态在十二个月内复制：M-RoPE 原生动态分辨率、带绝对时间对齐的动态 FPS 采样、ViT 中的窗口 attention、结构化 agent 输出格式。到 Qwen3-VL，配方已稳定：2D-RoPE-ViT 编码器带原生宽高比输入、MLP projector 进入大型 Qwen3 语言基础、强调 OCR、定位和 agent 行为作为一等目标的训练阶段。本课按时间顺序阅读家族，让你理解每个旋钮为何在那里。

**类型：** Learn
**语言：** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**前置知识：** Phase 12 · 06 (patch-n'-pack)
**时间：** ~120 分钟

## 学习目标

- 计算 M-RoPE 的三轴旋转（时间、高度、宽度）并解释为什么三者都需要。
- 为视频挑选动态 FPS 采样策略，并推理每秒 token 数与事件检测准确率。
- 按顺序说出四代 Qwen-VL 升级及每代解锁了什么。
- 连接 Qwen2.5-VL 风格 JSON agent 输出格式并从 VLM 响应解析结构化 tool call。

## 问题

Qwen-VL 在 2023 年 8 月作为 LLaVA-1.5 和 BLIP-2 的直接回应发货。Qwen 团队瞄准的 gap 是三重的：分辨率、视频和结构化输出。

分辨率：LLaVA-1.5 跑在 336x336。照片足够，中文发票或密集电子表格截屏无用。Qwen-VL 的第一个创新是 448x448 和 grounded bounding-box 输出，让模型指向东西。

视频：Video-LLaMA 堆叠每帧编码器并喂给 LLM。对短片段有效，对多分钟以上视频（时序轴是信号）无效。Qwen 团队想要一个理解时间的单个编码器。

结构化输出：LLaVA 发出自由形式文本。Agent 需要 JSON。Qwen-VL 在显式 JSON 输出格式上训练，包括坐标作为文本的 bounding-box。

每代 Qwen-VL 扩展这三个轴之一。

## 概念

### Qwen-VL (2023 年 8 月)

第一代：OpenCLIP ViT-bigG/14 作为编码器（2.5B 参数），LLama 兼容 Q-Former（1 步 256 query），Qwen-7B 基础。贡献：

- 448x448 分辨率（当时开放 VLM 的 SOTA）。
- Grounding：在带显式坐标 token 输出的图像-文本对上训练。"The cat is at <box>(112, 204), (280, 344)</box>"。
- 中英双语训练从一开始。

当时基准：英语上与 GPT-4V 竞争，中文上主导。Grounding 监督是真正的头条。

### Qwen2-VL (2024 年 9 月) — M-RoPE 和原生分辨率

Qwen2-VL 用原生动态分辨率 ViT 编码器取代固定分辨率 + Q-Former 栈。关键改变：

- 原生动态分辨率。ViT 接受任何能被 28 整除的 HxW（patch 14 带 2x 空间合并）。1120x672 图像（40x24 合并 patch）产生 960 视觉 token。无 resize，无 tiling，无缩略图。
- M-RoPE (Multimodal RoPE)。每个 token 携带 3D 位置 (t, h, w) 替代 1D。图像 t=0，视频 t = frame_index。RoPE 用每轴频率旋转 query/key 向量。无位置 embedding 表。
- MLP projector。丢弃 Q-Former；在合并 patch token 上使用 2 层 MLP。
- 视频带动态 FPS。默认 1-2 FPS 采样，但模型接受任意帧数。

结果：Qwen2-VL-7B 在多个多模态基准上匹配 GPT-4o 并在 DocVQA 上击败它（94.5 vs 88.4）。架构改变是决定性举措。

### Qwen2.5-VL (2025 年 2 月) — 动态 FPS + 绝对时间

Qwen2.5-VL 的大转变是视频。动态 FPS 不只是"需要时采样更多帧。"论文形式化：

- 绝对时间 token。替代位置索引（frame 0, 1, 2...），使用实际时间戳。"At 0:04, the cat jumps."模型看到序列中穿插的 `<time>0.04</time>` token。
- 动态 FPS。慢镜头 1 FPS 采样，动作 4+ FPS。用户或训练者选择；M-RoPE 适应。
- ViT 中的窗口 attention。空间 attention 是窗口化的（块内局部）以提升吞吐；每隔几层全局 attention。
- 显式 JSON 输出格式。在 tool-call 数据上训练：`{"tool": "click", "coords": [380, 220]}`。开箱即用的 agent-ready。
- MRoPE-v2 缩放。位置随最大输入大小缩放，10 分钟视频不会耗尽频率范围。

基准：Qwen2.5-VL-72B 在大多数视频基准上击败 GPT-4o，在文档上匹配 Gemini 2.0，并在 GUI grounding 上设置开放模型 SOTA（ScreenSpot：84% 准确率 vs GPT-4o 的 38%）。

### Qwen3-VL (2025 年 11 月)

Qwen3-VL 是巩固而非革新的增量升级：更大 LLM 主干（Qwen3-72B）、扩展训练数据、改进 OCR、通过 Qwen3 "thinking mode" 的更强推理。ViT 和 M-RoPE 保留。论文聚焦数据和训练改进而非架构。

谱系要点：到 2025 年 Qwen-VL 架构已稳定。额外世代扩展计算和数据，而非原语。

### M-RoPE 数学

经典 RoPE 用位置 `m` 通过成对坐标旋转 query `q`：

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE 把 hidden dim 分成三个频带。如 `d = 96`。32 dim 给时间，32 给高度，32 给宽度。每频带按自己的轴位置旋转。在 (t=5, h=10, w=20) 的 patch 得到三个频带的 `R_t(5)`、`R_h(10)`、`R_w(20)` 旋转。

文本 token 使用 `t = text_index, h = 0, w = 0`（或归一化选择），保持兼容。视频帧使用 `t = frame_time, h = row, w = col`。单图像使用 `t = 0`。

好处：一个位置编码处理文本、图像和视频，无需分支代码或不同位置表。

### 动态 FPS 采样逻辑

给定持续时间 `T` 秒的视频和目标 token 预算 `B`：

1. 计算能负担的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 从 `{1, 2, 4, 8}` 挑选满足 `fps <= fps_max` 的目标 FPS。
3. 如果运动高（光流启发式或显式用户请求），选更高 FPS。如果运动低，选更低。
4. 以选定 FPS 均匀采样；帧间插入 `<time>t</time>` token。

Qwen2.5-VL 隐式训练此逻辑；推理时用户通过 `fps` 参数控制。60 秒动作序列在 4 FPS、每帧 81 token = 19440 token，在 32k 上下文中可管理。

### 结构化 agent 输出

Qwen2.5-VL 的 agent 训练明确针对结构化 tool call：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是确定性的：JSON.parse over 模型输出。对比自由形式 "click at (1024, 512)" 需要 regex 和歧义处理。这就是 Qwen2.5-VL 的 ScreenSpot 分数从 Qwen2-VL 的 55% 跳到 84% 的转变。

## 使用它

`code/main.py` 实现：

- M-RoPE 位置计算，对混合文本、图像 patch 和视频帧的打包序列。
- 动态 FPS 采样器：给定（持续时间、预算、运动级别），挑选 FPS 并发出帧时间戳。
- 玩具 Qwen2.5-VL JSON 输出解析器，处理带坐标字段的 tool-call 响应。

运行它，然后感受在 5 分钟视频上交换固定 FPS 为动态 FPS 的差异。

## 交付它

本课产生 `outputs/skill-qwen-vl-pipeline-designer.md`。给定视频任务（监控、agent、动作识别、无障碍），它发出 Qwen2.5-VL 配置（帧预算、FPS 策略、窗口 attention 标志、agent-output 模式）和延迟估计。每当你为视频产品部署 Qwen-VL 家族模型时使用。

## 练习

1. 计算 hidden 48（每频带 16，base theta 10000）时 patch (t=3, h=5, w=7) 的 M-RoPE 旋转。展示每频带前三对的旋转角度。

2. 10 分钟安防录像在 1 FPS 产生多少帧？384 分辨率 3x pool 下多少总 token？Qwen2.5-VL 默认 32k 上下文能否处理？

3. 为 30 秒网球回合 vs 30 秒食谱演示 vs 30 秒 UI-agent 录制挑选 FPS。用动态 FPS 逻辑证明每个。

4. Qwen2.5-VL 完全丢弃 Q-Former。为什么简单 MLP 在 2025 年有效但在 2023 年无效？（提示：数据规模和编码器质量。）

5. 解析三个 Qwen2.5-VL JSON tool-call 输出为 Python dict。畸形 JSON 出什么错，Qwen cookbook 推荐什么恢复策略？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| M-RoPE | "Multimodal RoPE" | Hidden dim 中带时间、高度、宽度频带的 3D 旋转位置编码 |
| Dynamic FPS | "Smart sampling" | 基于运动、持续时间和 token 预算为每视频选择的帧采样率 |
| Absolute time token | "Timestamp token" | 序列中穿插的 `<time>t</time>`，让模型看到实际秒数而非帧索引 |
| Window attention | "Local attention" | 为速度限制在小窗口内的空间 self-attention；定期添加全局 attention |
| Structured agent output | "JSON mode" | 训练数据监督教 VLM 发出带坐标和工具名的可解析 JSON |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL 每请求控制限制总像素数从而限制 token 数的旋钮 |
| Grounding | "Point-at-it" | 作为文本 token 输出 bounding-box 坐标；自 Qwen-VL v1 使用 |

## 延伸阅读

- [Bai 等人 — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang 等人 — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu 等人 — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
