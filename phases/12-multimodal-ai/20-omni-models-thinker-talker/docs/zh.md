# Omni 模型：Qwen2.5-Omni 与 Thinker-Talker 拆分

> GPT-4o 在 2024 年 5 月的产品演示之所以具有颠覆性，不是因为底层模型，而是因为产品形态——一个语音界面，你说话，模型看到摄像头所见，并在 250ms 内回话。开放生态在 2024 和 2025 年剩余时间竞相达到那个产品表面。Qwen2.5-Omni (2025 年 3 月) 是参考开放设计：Thinker（大型文本生成 transformer）加 Talker（并行语音生成 transformer），由流式语音 token 连接。Mini-Omni 简化了它，Moshi 匹配了其延迟，GLM-4-Voice 将其扩展到中文。本课阅读 Thinker-Talker 架构和让流式实时对话工作的延迟预算。

**类型：** Build
**语言：** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**前置知识：** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**时间：** ~180 分钟

## 学习目标

- 将推理流水线拆分为 Thinker（文本推理）和 Talker（语音合成）并解释为什么并行流式工作。
- 计算对话交互的 time-to-first-audio-byte (TTFAB) 预算，逐组件。
- 描述 TMRoPE 在 Thinker 内跨视觉、音频和文本的时间对齐位置编码。
- 说出三种实时对话模式：half-duplex、turn-taking、full-duplex。

## 问题

实时语音助手需要做很多，很快：

1. 听到用户。实时语音 tokenization、voice activity detection (VAD) 知道用户何时说完。
2. 可选看到。摄像头输入以 2-4 FPS，与音频一起流入 Thinker。
3. 思考。基于对话历史组成响应。
4. 说话。合成音频 token、解码为波形、流式传输到用户扬声器。

每步增加延迟。对话感需要总往返 < 500ms——低于此，用户停止注意到滞后。GPT-4o 声称 ~250ms。Moshi ~160ms。Qwen2.5-Omni ~350-500ms。

每个组件需要流式。没有什么可以"batch 一切然后解码。"

## 概念

### Thinker 和 Talker

Qwen2.5-Omni 的分解：

- Thinker：7B-80B 文本生成 transformer。消费交错的文本 + 图像 + 音频 token。输出发出要说什么的文本 token。
- Talker：更小的语音生成 transformer（200M-1B）。消费 Thinker 的文本输出 token 加近期语音上下文 token。输出离散语音 token（residual-VQ 索引）。
- Speech decoder：流式波形 decoder（SNAC、MoVQGAN 家族），将语音 token 带到音频样本实时。

分离重要。Thinker 必须大才能良好推理。Talker 可以小因为其工作是局部的——将文本转换为语音 token。更大的 Talker 不是更具表现力；它更慢。

两者并行运行：

1. Thinker 发出文本 token t_i。
2. Talker 消费 t_i（通过流式）并发出语音 token s_i, s_{i+1}, ..., s_{i+k}。
3. Speech decoder 随到来消费语音 token 并发出音频样本。
4. Thinker 到达文本 token t_{i+3} 时，Talker 已为 t_0..t_{i+2} 流式传输音频。

### TMRoPE——时间对齐多模态位置

Thinker 需要整合以 4 FPS 到达的图像帧、以 50 帧/秒到达的音频帧、以及对话历史中的文本。朴素序列顺序（所有图像，然后所有音频，然后文本）丢失时间对齐。

TMRoPE 给每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户"stop"的文本 token 在 t=2.35s。RoPE 按时间戳旋转 attention；模型将它们视为时间上同时发生。

这是"他挥手时说你好"工作的基础设施——模型在同一概念时刻看到视频帧和音频。

### 流式语音合成

语音 token 必须流式传输。Mini-Omni (Xie & Wu, 2024) 引入"语言模型可以边思考边听和说"：Thinker 输出 token 和 Talker 输出 token 在同一序列中交错。Talker 在 Thinker 提交下一个文本 token 时立即触发。无 batch 边界。

Moshi (Défossez 等人, 2024 年 10 月) 是最快的开放实现。单 A100 上 160ms TTFAB。架构：单个 7B transformer 在交替位置发出文本和语音 token，带"inner monologue"将思考流与说话流分离。这实际上是 Thinker + Talker 融合到一个模型中，带仔细训练。

### VAD 和 turn-taking

语音活动检测在输入侧运行。两种模式：

- Half-duplex：用户说，模型听。模型说，用户听。通过 VAD 静默检测（~200ms）清晰切换。
- Full-duplex：两者可同时说话。模型可以 backchannel（"uh-huh"）或打断。难得多。Moshi 支持这个。

Qwen2.5-Omni 默认支持 half-duplex，通过静默阈值 turn-taking。Full-duplex 需要应用层处理。

### Qwen3-Omni (2025 年 11 月)

后继者。Qwen3-80B Thinker、更大的 Talker、改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开放权重。OmniBench 基准上与 Gemini 2.0 Live 竞争。

### 生产延迟预算

典型流式交互：

- Mic -> 音频 token：40-80ms。
- Prefill（prompt + 历史）：7B 时 100-200ms，70B 时多得多。
- 第一个 Thinker 文本 token：40ms。
- Talker 处理第一个文本 token：20ms。
- 第一个语音 token 提交：40ms。
- Residual-VQ 解码：30ms。
- Speech waveform 解码：50-80ms。

总 TTFAB：7B 时 320-510ms，70B 时 600-900ms。前沿质量通常意味着 70B+；因此前沿延迟差距。

### Token-rate 数学

16kHz 语音带 50 Hz 基础语音 token，每秒输出需要 50 语音 token。Talker 必须发出 ≥50 tok/s 才能跟上。典型 H100 上 LLM 吞吐 30-80 tok/s，小（200-300M）Talker 足够快；7B Talker 会落后。

这就是存在小专用 Talker 模型而不是"只使用主模型"的原因。

## 使用它

`code/main.py`：

- 用 mock token 发射率模拟 Thinker-Talker 流水线。
- 计算可配置模型大小和 mic 采样率下的 TTFAB。
- 用 VAD 静默阈值演示 half-duplex turn-taking。

## 交付它

本课产生 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉进、双语、full-duplex），在 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 之间挑选并调整 Thinker/Talker 大小。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述模型在用户于 t=1s 开始说话且摄像头在 t=1.2s 捕捉手势的 prompt 中看到什么。

3. Full-duplex 支持需要模型在听的同时发出音频。提出教这个的训练数据格式。

4. 阅读 Moshi 论文 Section 4。描述"inner monologue"分离及为什么它避免 Thinker-Talker 拆分。

5. 计算吞吐预算：Talker 必须以多快发出 token 才能跟上 16kHz 语音在 50 基础层 token/秒？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Thinker | "Reasoning brain" | 大型文本生成 transformer 产生要说什么 |
| Talker | "Speech-generating mouth" | 小型 transformer 从 Thinker 文本产生离散语音 token |
| TTFAB | "Latency budget" | Time-to-first-audio-byte：从用户语音结束到第一个音频样本输出 |
| TMRoPE | "Time-aligned RoPE" | 跨视觉、音频、文本使用绝对时间戳的位置编码 |
| Half-duplex | "Turn-taking" | 用户和模型交替；VAD 静默检测用户说完 |
| Full-duplex | "Simultaneous" | 模型可以同时说和听；能 backchannel |
| Inner monologue | "Moshi separation" | 单模型设计，思考流和说话流交错 |

## 延伸阅读

- [Xu 等人 — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez 等人 — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng 等人 — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
