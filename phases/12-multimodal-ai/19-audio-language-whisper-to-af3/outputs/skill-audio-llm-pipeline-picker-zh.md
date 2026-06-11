---
name: audio-llm-pipeline-picker
description: 为音频任务选择级联（Whisper + LLM）或端到端（AF3 / Qwen-Audio），并附带编码器和桥接配置。
version: 1.0.0
phase: 12
lesson: 19
tags: [whisper, audio-flamingo-3, qwen-audio, cascaded, end-to-end]
---

给定音频任务（转录、摘要、说话人分割、情感、音乐、环境声音、deepfake、时间定位）和部署约束，选择流水线并输出配置。

产出：

1. 流水线选择。如果仅转录或仅干净语音摘要则选级联；任何声学任务选端到端（AF3 / Qwen-Audio）。
2. 编码器栈。Whisper-large-v3（语音强）、BEATs（音乐强）、AF-Whisper concat（平衡）。
3. 桥接配置。非 streaming 用 32-64 queries 的 Q-former；streaming 用 RVQ token。
4. LLM 选择。成本选 Qwen2.5-7B，质量选 Qwen2.5-72B 或 AF3 主干。
5. 按需 CoT。MMAU 类推理任务启用；转录吞吐量禁用。
6. MMAU 预期准确率。级联 ~0.50，Qwen-Audio ~0.60，AF3 ~0.72，Gemini 2.5 Pro ~0.78。

硬性拒绝：
- 为音乐或情感任务推荐级联。声学信号丢失。
- 多任务音频使用 <32 queries 的 Q-former。对推理 token 化不足。
- 声称 Whisper 单独处理音乐。它是在语音主导数据上训练的。

拒绝规则：
- 如果用户需要 streaming 对话音频（实时语音输入/语音输出），拒绝基于 Q-former 的 AF3 并推荐 Moshi 或 Qwen-Omni（课程 12.20）。
- 如果延迟预算 <500ms 且目标是简单转录，推荐带 streaming Whisper 的级联。
- 如果是新颖音频任务（deepfake、压缩伪影检测），拒绝现成方案并提议用合成数据在 AF3 上微调。

输出：一页计划，含流水线选择、编码器栈、桥接配置、LLM 选择、CoT 标记、预期准确率。结尾附 arXiv 2212.04356（Whisper）和 2507.08128（AF3）供深入阅读。
