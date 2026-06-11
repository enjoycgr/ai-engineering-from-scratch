---
name: omni-streaming-budget
description: 为 Thinker-Talker streaming 语音流水线（Qwen-Omni / Moshi / Mini-Omni）针对目标 TTFAB 和功能集做大小规划。
version: 1.0.0
phase: 12
lesson: 20
tags: [qwen-omni, moshi, mini-omni, streaming, ttfab, thinker-talker]
---

给定语音优先产品规格（目标 TTFAB、麦克风采样率、视觉 yes/no、双语、全双工）和计算约束（GPU 等级、预算），为 Thinker-Talker 流水线做大小规划。

产出：

1. 模型系列选择。Moshi（最佳延迟）、Qwen2.5-Omni（最佳开放功能）、Qwen3-Omni（前沿质量）、Mini-Omni（最简单）。
2. Thinker 和 Talker 大小。<400ms TTFAB 用 7B Thinker + 200-300M Talker。质量用 70B+ Thinker，接受更高 TTFAB。
3. TTFAB 拆解。逐组件延迟估算。
4. 双工模式。默认用 VAD turn-taking 的半双工；如果产品需要 backchannel 则用全双工。
5. 视觉集成。TMRoPE 带交错视频帧的绝对时间戳。
6. 部署形态。基于吞吐量需求的单 GPU vs 分割（Thinker 在 A，Talker 在 B）。

硬性拒绝：
- 提议 70B Talker。Talker 必须小以跟上语音 token 速率。
- 使用非 streaming speech decoder。TTFAB 爆炸。
- 声称全双工即插即用。它需要专用训练数据。

拒绝规则：
- 如果目标 TTFAB <200ms，拒绝单 A100 上任何大于 Moshi 类（7B fused）的方案。
- 如果产品需要流内音乐生成，拒绝此架构并推荐单独音乐流水线。
- 如果麦克风采样率是 48kHz 且质量严格，标记需要更强的语音编码器；不要盲目降采样。

输出：一页 streaming 计划，含模型选择、大小、TTFAB 拆解、双工模式、视觉策略、部署。结尾附 arXiv 2503.20215（Qwen2.5-Omni）、2410.00037（Moshi）。
