---
name: audio-brief
description: 将音频需求简报翻译为 TTS、音乐和音效 (SFX) 的模型 + 提示词 + 评估计划。
version: 1.0.0
phase: 8
lesson: 11
tags: [audio, tts, music, sfx, codec]
---

给定一个音频需求简报（任务：TTS / 音乐 / 音效 / 语音克隆，时长、风格、声音或曲风、许可证约束、实时或离线、质量标准），输出：

1. 模型 + 托管方案。ElevenLabs V3、OpenAI TTS、XTTS v2、Suno v4、Udio、Stable Audio 2.5、MusicGen 3.3B、AudioCraft 2 或 GPT-4o realtime。一句话说明理由。
2. 提示词格式 (prompt format)。TTS：文本 + 语音提示（3–10 秒样本或声音 ID）+ 情感 / 语速标签。音乐：曲风 + 乐器 + 情绪 + BPM + 结构标记。音效：拟声词 + 声源 + 时长提示。
3. 编解码器 (codec) + 生成器 + 声码器 (vocoder) 链路。明确编解码器名称（EnCodec 32 kHz、DAC 44 kHz、自定义）和生成器选择（token-自回归 vs 流匹配 flow-matching）。
4. 种子 + 可复现性。种子固定、版本固定、提示词哈希。
5. 评估 (eval)。TTS 用 MOS（平均意见分）或 A/B，音乐用 CLAP 分数，TTS 转写用 CER，音效用用户试听测试。
6. 护栏 (guardrails)。语音克隆需获得所有者明确同意 + 水印（PerTh / SynthID-audio），音乐输出需版权扫描，训练数据政策检查。

拒绝在无验证同意的情况下克隆任何声音（卡带时代的"3 秒提示"不等于同意）。拒绝使用未授权参考素材生成音乐。标记任何低于 200 毫秒的实时目标，除非使用流式 token-自回归 (streaming token-AR) 模型——基于扩散 (diffusion) 的音频在 2026 年无法满足低于 300 毫秒的 TTFB。
