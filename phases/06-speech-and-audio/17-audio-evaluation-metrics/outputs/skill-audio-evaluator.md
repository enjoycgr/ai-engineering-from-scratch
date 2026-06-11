---
name: audio-evaluator
description: 为任何音频模型发布选择指标、基准、归一化规则和报告格式。
version: 1.0.0
phase: 6
lesson: 17
tags: [evaluation, wer, mos, utmos, eer, der, fad, mmau, leaderboard]
---

给定任务（ASR / TTS / 克隆 / 说话人验证 / 说话人分割 / 分类 / 音乐 / LALM / 流式 S2S），输出：

1. 主要指标。WER · MOS · UTMOS · SECS · EER · DER · mAP · FAD · MMAU-Pro 准确率 · 延迟 P95。选择一个。
2. 次要指标。1-3 个额外维度（速度、多样性、鲁棒性）及原因。
3. 归一化规则。小写、去除标点、数字展开、空白折叠。使用 Whisper-normalizer 或自定义，并记录。
4. 公共基准。报告所对照的标准排行榜（Open ASR、TTS Arena、MMAU-Pro、VoxCeleb1-O、AudioSet、LongAudioBench 等）。
5. 内部集。N 个样本的留出领域数据；人口统计 / 声学切片细分。
6. 报告格式。分布（延迟的 P50/P95/P99；分类的每类召回率；MMAU 的每类别）。发布说明模板。

拒绝延迟的单数字评估（报告百分位数）。拒绝分类的仅总体（报告每类）。拒绝没有 MOS/UTMOS 和 SECS（克隆时）的 TTS 发布。拒绝没有 WER 归一化规范的 ASR 发布。拒绝只有 FAD 的音乐发布——始终与人类 MOS panel 配对。

示例输入："发布新的英西对话 TTS。需要说服团队它比现有 Cartesia-Sonic 基线更好。"

示例输出：
- 主要：UTMOS（每种语言 50 个提示的成对音频样本）+ 人工 panel MOS（每种语言 20 名听众，与基线盲测 A/B）。
- 次要：TTFA 中位数 & P95（必须匹配基线）；SECS > 0.80 vs 固定声音参考（无说话人回归）；往返 ASR（Whisper-large-v3-turbo）CER < 2%。
- 归一化：英语用 Whisper-normalizer + 西班牙语用 Hugging Face multilingual-normalizer 用于往返 WER。
- 公共基准：TTS Arena（英语）和 Artificial Analysis Speech 用于相对 ELO 定位。目标：在最近的竞争对手 50 ELO 以内。
- 内部集：200 个留出提示（每种语言 100 个），涵盖金额、日期、产品名、两句叙述、情感朗读、代码切换。10 个人口统计声音。
- 报告：发布说明带标题（UTMOS + MOS）、P50/P95 TTFA 直方图、SECS CDF、CER 每类别细分、故障模式标注（代码切换提示在 X% 处失败）。
