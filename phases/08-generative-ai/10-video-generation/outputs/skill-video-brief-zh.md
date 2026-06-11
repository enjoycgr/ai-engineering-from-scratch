---
name: video-brief
description: 将视频需求简报翻译为 2026 年视频生成器的模型 + 提示词 + 镜头计划。
version: 1.0.0
phase: 8
lesson: 10
tags: [video, diffusion, sora, veo, kling]
---

给定一个视频需求简报（时长、宽高比、风格、主体、镜头计划、音频需求、画质标准、预算），输出：

1. 模型 + 托管方案。Sora、Veo 3、Kling 2.1、Runway Gen-3、Pika 2.0、CogVideoX、HunyuanVideo、WAN 2.2 或 Mochi-1。一句话说明理由，关联到时长 / 画质 / 许可证。
2. 提示词脚手架 (prompt scaffolding)。（a）镜头语言（establishing, tracking, dolly, crane, handheld），（b）主体 + 动作，（c）光照 + 风格，（d）负面提示词 (negative prompt) 或风格开关。Sora 目标 50–150 个 token，Runway 20–60 个。
3. 镜头计划 (shot plan)。单剪辑 vs 拼接多镜头，关键帧 (keyframe) 或首帧锚定 (first-frame anchors)，每镜头用 I2V 还是 T2V。
4. 种子 + 可复现性 (reproducibility)。每镜头种子、版本固定、工具仓库。
5. QA 检查清单 (checklist)。逐帧检查闪烁 (flicker)、身份一致性 (identity consistency)、物理违反、水印合规。
6. 音频 (audio)。Veo 3 原生支持，否则外挂（ElevenLabs、Suno 或授权音轨 + 口型同步 (lip-sync) 通道）。

拒绝承诺在免费 tier 上生成超过 10 秒的连续 1080p 运动（Pika / Kling / Runway 上限为 10 秒；更长需要拼接）。拒绝在无授权的情况下生成真实人物的肖像。标记任何暗示 2026 年实时 4K 生成的需求简报——当前最佳水平是在托管端点上半分钟生成一段 6 秒 1080p 剪辑。
