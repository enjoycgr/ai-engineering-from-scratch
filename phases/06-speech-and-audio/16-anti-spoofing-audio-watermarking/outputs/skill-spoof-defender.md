---
name: spoof-defender
description: 为语音生成 / 语音认证部署选择检测模型、水印、溯源清单和操作手册。
version: 1.0.0
phase: 6
lesson: 16
tags: [anti-spoofing, watermark, audioseal, asvspoof, c2pa, voice-fraud]
---

给定工作负载（语音生成 vs 语音认证、部署规模、合规区域、对抗者画像），输出：

1. 检测（CM）。AASIST · RawNet2 · NeXt-TDNN + WavLM · 商业（Pindrop、Validsoft）。训练数据：ASVspoof 2019 / ASVspoof 5 / 领域特定。目标 EER。
2. 水印（出站生成）。AudioSeal 16 位有效载荷编码 `(model_id, user_id, generation_ts)` · WaveVerify（替代）· 无（需论证）。检测器在每次输出交付前的 CI 中运行。
3. 溯源。用部署者密钥签名的 C2PA 清单 · IPTC 元数据 · 无（针对非消费者音频）。
4. 语音认证防护（如适用）。活体挑战（随机短语 TTS + 转录）、重放攻击检测（AASIST + PA 模型）、每信道生物识别阈值校准。
5. 运营。审计日志保留、同意凭证保留（7+ 年）、滥用检测信号（突然流量激增、命名实体提示）、kill-switch 流程。

拒绝没有 AudioSeal（或等效水印）的语音生成部署。拒绝没有反欺骗检测的语音生物识别部署 —— 语音克隆使得仅 cosine 认证可轻易绕过。拒绝仅依赖溯源清单的部署（可剥离）。拒绝未经过信道校准扫描的、在 ASVspoof 2019 上训练的检测阈值的实际部署。

示例输入："银行客服 IVR。语音生物识别解锁 + AI 生成语音智能体。每月 1000 万通电话。美国 + 欧盟。"

示例输出：
- 检测：首选 Pindrop 商业方案，或 NeXt-TDNN + WavLM 开源。在 ASVspoof 5 + 10 万银行特定通话样本上训练。领域内目标 EER %3C 0.5%。
- 水印：每次出站 TTS 话语嵌入 AudioSeal 16 位有效载荷；有效载荷编码 bank_id + session_id + timestamp。传输前检测器验证。
- 溯源：面向客户的音频导出工作流使用 C2PA 清单；内部通话跳过。
- 语音认证：每次认证进行活体挑战（TTS 随机 4 位数字短语；用户复述 + 检测器 + 转录器）。每次入站认证尝试运行反欺骗。生物识别阈值 FAR 0.1%、FRR 1%。
- 运营：同意 + 审计日志在区域内保留 7 年（欧盟数据驻欧盟）。克隆请求流量突然 %3E 2σ 时告警；滥用检测时触发 kill-switch。
