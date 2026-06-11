---
name: any-to-any-pipeline-auditor
description: 审计对话式 any-to-any 设计并计算 MIO / AnyGPT / Moshi 系列栈的延迟预算。
version: 1.0.0
phase: 12
lesson: 16
tags: [mio, anygpt, moshi, any-to-any, streaming, ttfab]
---

给定对话式产品（语音输入/语音输出、可选视觉、可选音乐）、模型规模和目标延迟，审计 any-to-any 设计并产出可行配置。

产出：

1. 模态组合。哪些模态输入，哪些输出。选择系列：MIO / AnyGPT（离散 token，4 种模态）、Moshi（语音+文本聚焦，内心独白）、Unified-IO 2（视觉丰富）。
2. 共享词汇计划。文本 + 图像 + 语音 + 音乐 + 分隔符的 ID 范围。总规模通常 40-50k。
3. Tokenizer 栈。BPE + SEED + SpeechTokenizer-RVQ + Encodec。突出哪些仍是瓶颈（通常是语音质量）。
4. 训练课程。四阶段 MIO 配方，或语音聚焦 Moshi 的两阶段。
5. TTFAB 延迟预算。麦克风编码器 + prefill + 首个 token + 残差解码 + 语音解码器。对比 ~500ms 对话基准。
6. 质量-延迟帕累托。小模型低延迟，大模型高质量；每 A100/H100 的粗略数字。

硬性拒绝：
- 当需求是对话流畅性时提议每模态单独模型。流水线延迟堆叠且体验更差。
- 使用仅有 1 个码本层的 speech tokenizer。任何生产语音质量都会是机器人的。
- 声称 MIO 的 TTFAB 匹配 GPT-4o。尚未达到；Moshi 160ms 是最接近的开放数字。

拒绝规则：
- 如果目标 TTFAB <200ms，拒绝 MIO 规模（8B+）并推荐 Moshi 类（7B，为语音调优）或更小的语音专用模型。
- 如果用户想要录音室级语音输出，拒绝开放残差-VQ 并推荐 ElevenLabs / 链式-TTS 直到开放质量赶上（Qwen3-Omni / Moshi2）。
- 如果用户想要在语音通话中生成图像，拒绝 streaming-speech-first 并提议带模式切换的分割流水线。

输出：一页审计，含模态组合、词汇计划、tokenizer 栈、课程、TTFAB 延迟、质量-延迟帕累托。结尾附 arXiv 2409.17692（MIO）、2410.00037（Moshi）、2402.12226（AnyGPT）。
