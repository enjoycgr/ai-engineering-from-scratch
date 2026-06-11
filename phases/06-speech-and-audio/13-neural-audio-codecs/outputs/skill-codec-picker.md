---
name: codec-picker
description: 为给定生成或压缩任务选择神经音频编解码器（EnCodec / DAC / SNAC / Mimi）。
version: 1.0.0
phase: 6
lesson: 13
tags: [codec, encodec, dac, snac, mimi, rvq, semantic-tokens]
---

给定任务（生成式 LM、压缩、全双工对话、音乐编辑、保真度目标），输出：

1. 编解码器。EnCodec-24k · EnCodec-48k · DAC-44.1k · SNAC-24k · Mimi ·（回退：Opus 用于非神经压缩）。一句话原因。
2. 帧率 + 码本。比特率预算、码本数量（通常 4-12）、目标片段时长对应的序列长度。
3. 分词方案。Flat vs 分层（SNAC）vs 语义+声学（Mimi）。LM 如何消费 token。
4. 解码器。编解码器内建解码器 · 外部声码器（HiFi-GAN）· 仅 LM（无声码器，直接预测 codec token）。解释原因。
5. 训练影响。需要训练编码器/解码器？在领域音频上微调（仅语音 → 领域特定音乐）？冻结现成？

拒绝将 DAC 用于紧延迟预算的 AR-LM 工作负载——86 Hz 帧率 × 8 码本 = 每秒 5504 个 token，对快速生成来说太长。拒绝将 Mimi 用于音乐——它是语音调优的。拒绝将 EnCodec 用于语义条件生成——没有语义码本，从文本生成的语音模糊。

示例输入："构建用于文本到语音 TTS 的 AR LM。目标 TTFA 200 ms。仅英语。"

示例输出：
- 编解码器：Mimi。语义+声学分离支持文本 → 码本 0 → 码本 1-7 的分解，既快又支持声音克隆。
- 帧率 + 码本：12.5 Hz · 8 码本 · 4.4 kbps。10 秒 = 1000 个 token。
- 分词：先从文本 + 说话人参考预测码本 0；然后在码本 0 + 说话人参考的条件下预测码本 1-7（depth-transformer 模式）。
- 解码器：Mimi 内建解码器，无需外部声码器。
- 训练：训练 text-to-codec LM；冻结 Mimi。
