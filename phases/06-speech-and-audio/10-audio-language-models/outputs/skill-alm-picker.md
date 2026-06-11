---
name: alm-picker
description: 为音频理解任务选择音频-语言模型、基准子集、输出模态（文本 vs 语音）和防护措施。
version: 1.0.0
phase: 6
lesson: 10
tags: [alm, lalm, qwen-omni, audio-flamingo, gemini-audio, mmau]
---

给定任务（语音 / 声音 / 音乐 / 多音频 / 长音频、输出模态、延迟、许可证），输出：

1. 模型。Qwen2.5-Omni-7B · Qwen3-Omni · SALMONN · Audio Flamingo 3 · AF-Next · LTU · GAMA · Gemini 2.5 Pro (API) · GPT-4o Audio (API)。一句话原因。
2. 验证用基准子集。MMAU-Pro 语音 / 声音 / 音乐 / 多音频 · LongAudioBench · AudioCaps · ClothoAQA。选择与用户任务匹配的维度。
3. 输出模态。仅文本 · 文本 + 语音（Qwen-Omni、GPT-4o Audio）。如需额外语音解码器，计入预算。
4. 防护措施。当模型的多音频得分 < 30%（接近随机）时，拒绝需要多音频比较的提示。对 > 10 分钟的输入，在 LALM 之前先做说话人分离。
5. 升级路径。何时该任务应回退到专用模型——Whisper 用于转录，BEATs 用于分类，pyannote 用于说话人分离。LALM 不是每个领域的最佳。

拒绝交付多音频比较任务而不验证模型在 MMAU-Pro 多音频子集上得分 > 40%。拒绝长音频（> 10 分钟）而不做上游说话人分离。标记任何使用厂商报告数字而不独立重新验证的部署。

示例输入："合规审计：转录 10 分钟银行通话录音 + 检测座席是否阅读了强制披露。"

示例输出：
- 模型：Whisper-large-v3-turbo 用于转录 + Gemini 2.5 Pro（通过 API）用于转录文本上的披露检查 QA。直接在原始音频上使用 LALM 很诱人，但长音频 LALM 准确率超过 10 分钟会下降。
- 基准子集：MMAU-Pro 语音子集（Gemini 2.5 Pro = 73.4%）——覆盖语音推理维度。也在你自己的 50 通黄金通话集上抽查。
- 输出模态：仅文本。审计报告不需要语音输出。
- 防护措施：先用 pyannote 3.1 做说话人分离；分别发送每说话人片段；记录每通电话的置信度分数。
- 升级路径：如果一通电话未通过披露检查，路由到人工审核员而不是自动标记。
