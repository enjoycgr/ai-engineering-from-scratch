---
name: speaker-verifier
description: 为给定任务选择模型、注册协议、阈值调参计划和欺诈防护措施，设计说话人验证或分离流程。
version: 1.0.0
phase: 6
lesson: 06
tags: [audio, speaker, verification, diarization]
---

给定一个目标（验证 vs 识别 vs 分离、领域、信道、威胁模型）和数据（阈值调参小时数、说话人数、注册片段预算），输出：

1. 嵌入器。ECAPA-TDNN / WavLM-SV / ReDimNet / x-vector。原因。
2. 注册协议。片段数、最短时长、噪声门、信道匹配。
3. 打分。余弦 / PLDA；是否使用 AS-norm；队列大小。
4. 阈值。目标 FAR（欺诈风险）或 EER；调参集大小。
5. 欺骗防御。反欺骗模型（AASIST、RawNet2）、活体挑战或重放检测。

拒绝任何没有反欺骗前端的欺诈级部署。拒绝不报告评估集、其信道和片段长度分布的 EER 发布。标记任何跨领域未重新调参的固定余弦阈值。
