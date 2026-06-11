---
name: realtime-voice-pipeline
description: 为目标端到端延迟选择传输、VAD、流式 STT、LLM、流式 TTS 和编排。
version: 1.0.0
phase: 6
lesson: 11
tags: [voice-agent, livekit, pipecat, silero, streaming, latency]
---

给定目标（延迟 P50/P95、语言、信道、离线 vs 云端、通话量），输出：

1. 传输。WebRTC (LiveKit / Daily) · WebSocket · SIP 中继 (Twilio / Telnyx)。原因与抖动容忍度 + 用例挂钩。
2. VAD + 轮替检测。Silero VAD（开源，99.5% TPR）· Cobra（商业）· LiveKit turn-detector。阈值、最短语音时长、静音保持时间。
3. 流式 STT。Parakeet TDT（最快开源）· Kyutai STT（带 flush 技巧）· Deepgram Nova-3（API，~150 ms）· Whisper-streaming。原因。
4. LLM + 流式。在 TTS 启动前固定前 20 个 token。模型 + 流式配置 + 针对提示注入的防护措施。
5. 流式 TTS。Kokoro-82M（~100 ms TTFA）· Orpheus · Cartesia Sonic · ElevenLabs Turbo。声音包或克隆保护（Lesson 8）。
6. 编排。LiveKit Agents · Pipecat · Vapi · Retell · 自定义 Rust。原因与团队技能 + 规模挂钩。
7. 可观测性。每阶段 P50/P95/P99 直方图；误报打断率；掉话率；通话样本上的 WER。

拒绝缓冲整段话语后才做 STT 的部署。拒绝不流式的 TTS。拒绝用平均延迟评估——要求 P95。拒绝管理型平台（Vapi / Retell）用于 > 10 万分钟/月而不与自建做成本比较。

示例输入："汽车保险报价语音智能体。< 500 ms P95。英语，美国。5 万分钟/周。合规：HIPAA 相关（日志中无 PII）。"

示例输出：
- 传输：LiveKit Agents + Twilio SIP。经呼叫中心规模验证，支持 HIPAA 模式选择。
- VAD：Silero VAD @ 阈值 0.45，最短语音 220 ms，静音保持 400 ms。LiveKit turn-detector 覆盖层。
- STT：Deepgram Nova-3 英语（~150 ms P95）；如需本地审计则回退到 Parakeet-TDT。
- LLM：通过 OpenAI realtime API 的 GPT-4o 流式；用后置过滤器防护提示注入；固定前 20 个 token 给 TTS。
- TTS：Cartesia Sonic 2（~150 ms TTFA，不使用声音克隆——预定义声音）。
- 编排：LiveKit Agents。通过 Hamming AI 做生产可观测性。
- 日志：持久化前用正则 + NER  pass 去除 CVV / SSN / DOB。保留 30 天。
