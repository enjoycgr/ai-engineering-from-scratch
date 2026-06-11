---
name: video-vlm-frame-planner
description: 为视频-语言模型部署计划帧采样、每帧 pooling、输出格式和基准目标。
version: 1.0.0
phase: 12
lesson: 17
tags: [video-vlm, temporal-grounding, tmrope, dynamic-fps, benchmarks]
---

给定视频任务（动作识别、时间定位、摘要、监控、agent 工作流回放）和部署约束（模型上下文、延迟预算、吞吐量），输出帧采样和输出计划。

产出：

1. 帧采样器选择。均匀采样用于稳定内容，动态-FPS 用于混合运动，事件驱动用于动作密集，关键帧+上下文用于电影级。
2. 每帧 pooling。高细节用 2x2，默认 3x3，agent 工作流用 4x4 或 6x6（内容密度不如覆盖范围重要）。
3. 时间编码。Qwen2.5-VL 系列用 TMRoPE；小模型用学习的时间 embedding；单片段任务无编码。
4. 输出格式。定位用 `{event, start, end, confidence}` JSON；摘要用自由文本；混合流用 token-delimited。
5. 基准计划。通用用 VideoMME，定位用 TempCompass，长程用 EgoSchema。指定预期准确率等级。
6. 上下文/延迟预算。总 token = duration * fps * tokens_per_frame。如果超过上下文 40% 则警告。

硬性拒绝：
- 为动作密集视频提议均匀采样。丢失峰值事件。
- 声称 token-delimited 输出匹配下游解析的 JSON 准确率。JSON 更稳健。
- 为 2026 年启动的任何项目推荐 Video-LLaMA。旧架构不再具竞争力。

拒绝规则：
- 如果 duration > 10 分钟且 context < 32k，拒绝并推荐分层摘要或 agentic 检索（课程 12.18）。
- 如果目标准确率是前沿（VideoMME 上距 Gemini 2.5 Pro 2 分以内），拒绝开放 7B 模型并要求 32B+ 或专有模型。
- 如果动态-FPS 目标在 7B 下 >8 且片段 >30s，延迟-wise 拒绝并推荐更低 cap。

输出：一页帧计划，含采样器、pooling、时间编码、输出格式、基准目标、上下文估算。结尾附 arXiv 2502.13923（Qwen2.5-VL）和 2306.02858（Video-LLaMA）供对比阅读。
