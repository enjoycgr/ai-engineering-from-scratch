---
name: voice-pipeline
description: 搭建 Pipecat 风格的 voice pipeline（语音流水线）（VAD + STT + LLM + TTS + transport），包含 barge-in（打断）、confidence gating（置信度门控）和 latency budget enforcement（延迟预算强制）。
version: 1.0.0
phase: 14
lesson: 22
tags: [voice, pipecat, livekit, webrtc, latency]
---

给定一个语音产品 spec（语言、transport 传输协议、providers 提供商），搭建一个 frame-based pipeline（基于帧的流水线）。

生成：

1. `Frame` 类型，包含 `kind`、`payload`、`direction`（downstream 下游 / upstream 上游）。
2. Processors（处理器）：`VAD`、`STT`、`LLM`、`TTS`、`Transport`。每个都有 `process(frame)`。
3. `link()` helper 将 processors 正向和反向链接。
4. Cancel frame handling（取消帧处理）：UPSTREAM 路径从 transport 到 TTS 到 LLM 到 STT，在每个阶段丢弃 pending work（挂起工作）。
5. Observers（观察者）：per-stage latency metrics（每阶段延迟指标）；每个 frame 跨越 processor 时发射 OTel span（Lesson 23）。
6. Confidence gate（置信度门控）on STT：低于 threshold（阈值）时，发射 "please repeat"（"请重复"）text frame 而非 transcript（转录）。

Hard rejects（硬性拒绝）：

- 没有 UPSTREAM handling（上游处理）的 pipeline。Barge-in（打断）对语音不是可选的。
- 没有 streaming（流式传输）的 LLM calls。First-token latency（首个 token 延迟）占主导；必须 streaming。
- Confidence-blind（无视置信度）的 STT。将错误 transcripts（转录）输入 LLM 会产生错误回复。

Refusal rules（拒绝规则）：

- 如果 end-to-end latency（端到端延迟）在 cold run（冷启动）时超过 1500ms，拒绝发布。优化链条或使用 MultimodalAgent（LiveKit 直接音频）。
- 如果产品是 telephony-first（电话优先）且 pipeline 没有 SIP adapter，拒绝。通过 LiveKit SIP 或平台（Vapi/Retell）路由。
- 如果产品在传输中承载 PII audio 且无加密，拒绝。

输出：`frames.py`、`processors.py`、`pipeline.py`、`observers.py`、`README.md`，解释 latency budget（延迟预算）、barge-in design（打断设计）和 transport choice（传输协议选择）。最后以 "what to read next" 指向 Lesson 23（OTel）、Lesson 24（observability backends 可观测性后端）或 LiveKit docs 了解 WebRTC 细节。
