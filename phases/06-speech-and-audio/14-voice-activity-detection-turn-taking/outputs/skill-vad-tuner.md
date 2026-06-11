---
name: vad-tuner
description: 为语音助手选择 VAD 模型、阈值、静音保持、预卷和轮替检测策略。
version: 1.0.0
phase: 6
lesson: 14
tags: [vad, silero, cobra, turn-detection, flush-trick]
---

给定工作负载（消费者 / 呼叫中心 / 边缘 / 无障碍；噪声特征；语言混合；延迟），输出：

1. VAD。Silero VAD（默认）· Cobra（商业精度）· pyannote segmentation（说话人分割级）· WebRTC VAD（遗留 / 极小）。一句话原因。
2. 参数。阈值（0.3-0.5）、最短语音（200-300 ms）、静音保持（400-800 ms）、预卷（250-500 ms）。
3. 语义轮替检测。启用（LiveKit turn-detector 或自定义 MLP）或否。原因与预期用户语音模式挂钩。
4. Flush trick。启用（如果 STT 支持 —— Kyutai / Deepgram）或否。预期延迟节省。
5. 保护机制。拒绝短于最短时长的语音；始终保留预卷；限制每用户静音保持覆盖；VAD 服务宕机时 fail-open（将所有内容视为语音）。

拒绝生产环境使用纯能量 VAD —— 噪声太大。拒绝零静音保持 —— 会打断用户。拒绝在有专用 Silero 时使用基于 Whisper 的 VAD（更慢、更不准）。

示例输入："呼叫中心 IVR，航空公司改签。嘈杂背景（机场）。英语 + 西班牙语。< 500 ms 轮替检测。"

示例输出：
- VAD：Cobra（商业），因为噪声抗性优势。成本过高时回退到 Silero。
- 参数：阈值 0.4（机场噪声底较高）；最短语音 300 ms；静音保持 600 ms（用户在 IVR 中常暂停阅读航班号）；预卷 400 ms。
- 语义轮替：启用 LiveKit turn-detector —— 句子中间停顿常见（"我要改签航班...到明天"）。
- Flush trick：在 Deepgram streaming 上启用。预期节省：400 ms → 150 ms 轮替结束延迟。
- 保护机制：Cobra/Deepgram 不可达时 fail-open；审计记录每次 VAD 触发事件用于调优。
