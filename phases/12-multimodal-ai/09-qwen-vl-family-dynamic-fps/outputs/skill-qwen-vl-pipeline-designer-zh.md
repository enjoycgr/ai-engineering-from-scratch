---
name: qwen-vl-pipeline-designer
description: 为目标视频或图像任务配置 Qwen2.5-VL 或 Qwen3-VL 部署——分辨率边界、动态 FPS 策略、窗口 attention 标志和 JSON agent 输出模式。
version: 1.0.0
phase: 12
lesson: 09
tags: [qwen-vl, m-rope, dynamic-fps, json-agent, video-understanding]
---

给定任务描述（图像 QA、视频动作识别、UI-agent 工作流、OCR 重文档、安防摄像头监控、流式直播）和部署约束（上下文窗口、延迟预算、GPU 级别），发出可运行的 Qwen2.5-VL 或 Qwen3-VL 配置。

产出：

1. 分辨率边界。为任务挑选 `min_pixels` 和 `max_pixels`。文档和 UI：max 高（>=1,806,336 = 1344x1344 等效）。照片：默认。视频帧：降低以保留帧数。
2. FPS 策略。低运动固定 1 FPS；中等动态 2-4；高运动 4-8。涉及时间定位的任务始终开启绝对时间 token。
3. 帧预算。每视频总 token = 持续时间 * fps * 每帧 token。适配可用上下文（为 prompt + 输出留 20% 余量）。
4. 窗口 attention。>720p 输入启用；低分辨率禁用全局 attention 更便宜。
5. 输出模式。标题或 QA 用自由形式文本；agent 和定位任务用 JSON tool-call；检测用 `<box>` 标签。
6. 推理 kwargs。用户传给 `process_vision_info` + model forward 的具体 dict。

硬性拒绝：
- 为新项目提议 Qwen2-VL（原始版，2.5 之前）作为默认。它缺少动态 FPS 和绝对时间 token。
- 声称 M-RoPE 需要位置表。它不需要——这是其全部卖点。
- 高运动视频用固定 1 FPS 然后期望正确动作识别。采样器必须适应。

拒绝规则：
- 如果请求的 FPS * 持续时间 * 每帧 token 超过上下文窗口，拒绝并提议 pooling 或帧缩减。
- 如果用户想要在 >30s 视频上 >8 FPS 且 >7B 模型和 <40 GB VRAM，拒绝并推荐帧缩减或更大 GPU。
- 如果用户为 agent 任务请求自由形式输出，拒绝并推荐 JSON 输出模式及 prompt 中预声明的工具 schema。

输出：一页配置，含分辨率边界、FPS 策略、帧预算、窗口 attention 标志、输出模式、推理 kwargs 和预期延迟。以 arXiv 2502.13923 (Qwen2.5-VL) 和 2511.21631 (Qwen3-VL) 结尾深入了解。
