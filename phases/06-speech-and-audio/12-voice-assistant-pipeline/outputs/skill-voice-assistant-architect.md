---
name: voice-assistant-architect
description: 给定工作负载，产出全栈语音助手规格——组件、延迟预算、可观测性、合规。
version: 1.0.0
phase: 6
lesson: 12
tags: [voice-assistant, architecture, livekit, pipecat, compliance]
---

给定用例（消费者 / 客户支持 / 无障碍 / 边缘）、预期规模（并发会话、分钟/月）、语言、延迟目标、合规（HIPAA、PCI、欧盟 AI 法案、加州 SB 942），输出：

1. 组件（7 层）。麦克风 + 分块 · VAD · 流式 STT · LLM + 工具 · 流式 TTS · 播放 · 打断处理器。为每层命名具体提供商/模型。
2. 延迟预算。每阶段 P50 / P95 / P99 目标，累加到端到端目标。标记哪些阶段是并行 vs 串行。
3. 工具调用模式。每个工具的 JSON 规范 + 错误处理 + 回退文本。始终包含一个"无法帮助"路径，LLM 在失败两次时必须走这条路。
4. 安全。提示注入防护、声音克隆锁定（如果 TTS 支持克隆）、唤醒词门控（针对始终开启）、日志中的 PII 脱敏、30 天保留。
5. 可观测性。每阶段 P50/P95/P99 · 误打断率 · 工具调用成功率 · 每 100 通电话的 WER · 每分钟成本 · 放弃率。
6. 合规。披露音频（"This is an AI assistant"）、区域固定（欧盟数据留在欧盟）、审计日志保留、退出路径。

拒绝没有唤醒词的始终开启部署。拒绝不流式的 TTS（增加话语长度延迟）。拒绝不用 P95 的平均延迟——尾部是用户流失的地方。拒绝未经法律审查的原始音频保留 > 30 天。

示例输入："低视力用户的无障碍助手：面向消费者邮件应用的纯语音界面。英语。P95 < 600 ms。~1 万并发用户。"

示例输出：
- 组件：sounddevice（通过 LiveKit Agents 的 WebRTC）· Silero VAD · Deepgram Nova-3（英语）· GPT-4o 配合邮件工具（read_message、compose_reply、mark_read）· Cartesia Sonic 2 流式 · WebRTC 输出 · 打断 = VAD 触发时取消 LLM 和 TTS。
- 预算：捕获 120 ms + VAD 40 + STT 150 + LLM TTFT 100 + TTS TTFA 150 = 560 ms P95。
- 工具：read_message({id})、compose_reply({message_id, body})、mark_read({id})、search({query})。全部返回 JSON；LLM 每个工具最多重试 2 次，然后回退"I couldn't do that — try rephrasing"。
- 安全：提示注入防护（检测 `ignore previous instructions`）；唤醒词"Hey Mail"；无声音克隆（固定 Cartesia 声音）；日志中脱敏邮件正文。
- 可观测性：Hamming AI 生产监控；每阶段 Prometheus 直方图；误打断 > 5% 或 p95 > 800 ms 时告警。
- 合规：首次使用时 AI 披露；仅针对医疗消息的 HIPAA 选择加入；欧盟用户接入欧盟托管的 Cartesia + GPT-4o Ireland。
