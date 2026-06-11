---
name: duplex-pipeline
description: 为语音智能体工作负载选择全双工（Moshi）vs 流程（VAD + STT + LLM + TTS）架构。
version: 1.0.0
phase: 6
lesson: 15
tags: [moshi, hibiki, full-duplex, voice-agent, streaming]
---

给定工作负载（延迟目标、工具调用需求、语言覆盖、硬件预算、云端 vs 边缘），输出：

1. 架构。全双工（Moshi / GPT-4o Realtime / Gemini Live）vs 流程（LiveKit + STT + LLM + TTS，Lesson 12）。一句话原因。
2. 模型。Moshi · Hibiki · Hibiki-Zero · Sesame CSM · GPT-4o Realtime · Gemini 2.5 Live · 传统流程。原因。
3. 规模。每会话 GPU 成本（Moshi 占用一个槽位）、最大并发会话数、冷启动影响。
4. 工具调用路径。如果需要——混合流程（全双工 + 外部 LLM 用于工具调用）或纯流程。解释权衡。
5. 语言覆盖。全双工模型语言支持窄；流程继承 LLM 的多语言能力。

拒绝为需要工具调用 / 检索的企业智能体使用纯全双工架构 —— Moshi 是对话模型，不是智能体框架。拒绝为亚 250 ms 对话智能体使用纯流程架构 —— 阶段累加。拒绝在单 GPU 上为 Moshi 分配 %3E 4 并发会话 —— 会碰到争用。

示例输入："语言学习语音伴侣 —— 对话流利度练习。英语 + 法语。%3C 250 ms 响应。1 万日活。"

示例输出：
- 架构：全双工（Moshi）。亚 250 ms 延迟需求 + 对话流利度契合 Moshi 的优势。
- 模型：Moshi。EN + FR 都支持良好。CC-BY 4.0 许可证。
- 规模：每 4-6 并发会话一个 L4 GPU → 峰值约 1500 GPU 支撑 1 万 DAU 按 10% 并发。规划安静路径使用 Kyutai Pocket TTS + 本地 Whisper 的设备端轻量模式。
- 工具调用：最少 —— "揭示语法提示"和"翻译这个短语"可通过小型 LLM sidecar 路由；大部分交互是 Moshi 擅长的开放式对话。
- 语言覆盖：EN + FR（原生）；ES / DE / JP 通过 Hibiki-Zero 适配（每新语言需 1000 小时音频）。
