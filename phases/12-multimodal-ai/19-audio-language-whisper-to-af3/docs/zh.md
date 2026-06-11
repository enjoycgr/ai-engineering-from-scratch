# 音频-语言模型：从 Whisper 到 Audio Flamingo 3

> Whisper (Radford 等人, 2022 年 12 月) 解决了语音识别——68 万小时弱监督多语言语音，简单编码器-解码器 transformer，一个让后续每次 ASR 发布都引用它的基准。但识别不是推理。问"这段录音里有什么乐器"或"说话者在表达什么情绪"或"第 3 分钟发生了什么"需要音频理解，不是转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3 (AF3, 2025 年 7 月) 逐步构建了那个栈：保留 Whisper 类编码器，装上 Q-former，在音频-文本指令数据上训练，添加 chain-of-thought 推理。本课走过这个弧。

**类型：** Build
**语言：** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**前置知识：** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**时间：** ~180 分钟

## 学习目标

- 从波形计算 log-Mel spectrogram：窗、FFT、filter banks、log 变换。
- 对比编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合。各何时获胜。
- 构建音频 Q-former：N 个可学习 query cross-attending 到 spectrogram patch。
- 解释级联 (Whisper-then-LLM) vs 端到端音频-LLM 训练：为什么端到端对推理扩展更好。

## 问题

语音识别被 Whisper 解决。音频 OCR 是商品。但"商品"停在转录。如果模型无法推理它听到的——时间、说话者、情绪、音乐结构、环境声音——转录 alone 无法驱动产品功能。

三条明显路线：

1. 级联：Whisper 转录，LLM 在 transcript 上推理。对纯语音场景有效。对音乐、环境音频、多说话者重叠、情绪失败。

2. 端到端音频-LLM：音频编码器直接将音频 token 喂入 LLM，跳过转录。保留声学信息（情绪、说话者、环境）。需要新训练数据。

3. 混合：音频编码器 + 文本解码器既能转录又能推理。Qwen-Audio 和 Audio Flamingo 走这条路线。

## 概念

### Log-Mel spectrogram：输入特征

每个音频编码器从相同特征开始：log-Mel spectrogram。

1. 重采样到 16 kHz。
2. 25ms 窗口、10ms hop 的短时傅里叶变换。
3. 取 FFT 结果的幅度。
4. 应用 Mel filter banks（通常 80 个 filter，对数间距 0-8000 Hz）warp 到感知频率。
5. Log 压缩 (log(1 + x)) 用于动态范围。

结果：形状 (T, 80) 的 2D 数组，T 是时间帧数。30 秒片段在 100 Hz 帧率：(3000, 80)。

### Whisper 编码器

Whisper 编码器是处理 log-Mel spectrogram 作为时间帧序列的 12 层 ViT 风格 transformer。输出：每时间帧一个 hidden-state 向量。

对 ASR，Whisper 解码器是条件于编码器输出的 cross-attention transformer，生成文本 token。标准编码器-解码器。

对 ALM（音频-LLM），你想要编码器输出作为不同 LLM 的输入。模式：Whisper 编码器冻结，Q-former 可训练，LLM 冻结或调优。

### BEATs 和音频特定编码器

Whisper 在语音主导数据上训练。对音乐和环境音频较弱。

BEATs (Chen 等人, 2022) 是在 AudioSet 上训练的自监督 transformer。在相同参数量下比 Whisper 更好地捕捉音乐和环境声音。

AF-Whisper (Audio Flamingo 3 的混合)：拼接 Whisper + BEATs 特征作为音频输入。Whisper 携带语言信号，BEATs 携带声学信号。

### 音频 Q-former

与 BLIP-2 视觉 Q-former 相同模式。固定数量的可学习 query（通常 32 或 64）cross-attend 音频编码器输出帧。Query 成为 LLM 消费的音频 token。

训练对齐阶段：仅 Q-former，在音频-文本对上的对比 + 标题 loss（AudioCaps、Clotho）。指令阶段：端到端，解冻 LLM，在指令数据上训练。

### 弧——SALMONN、Qwen-Audio、AF3

SALMONN (Tang 等人, 2023)：Whisper + BEATs + Q-former + LLaMA。第一个带严肃推理能力的开放音频-LLM。MMAU 上 ~0.55 综合。

Qwen-Audio (Chu 等人, 2023)：类似架构，在更丰富数据集上训练，调优多轮对话。MMAU ~0.60。

LTU——Listen, Think, Understand (Gong 等人, 2023)：显式推理数据，聚焦音频片段上的 chain-of-thought。更小但更聚焦。

Audio Flamingo 3 (Goel 等人, 2025 年 7 月)：当前开放 SOTA。8B LLM 主干 (Qwen2 7B)、Whisper-large 编码器拼接 BEATs、64-query Q-former、在 100 万+ 音频-文本指令对上训练。MMAU 0.72，在一些子任务上匹配前沿专有模型。

AF3 还引入音频按需 chain-of-thought：模型可选发出思考 token（"让我先识别乐器：...") 在最终答案前。复杂推理任务上启用思考时准确率提升 3-5 分。

### 级联 vs 端到端

级联流水线：

1. Whisper 将音频转录为文本。
2. LLM 在文本上推理。

对"总结这个播客"完美工作。对以下失败：
- "这首歌的情绪是什么？"——情绪在声音中，不在文字里。
- "Alice 还是 Bob 在说话？"——需要说话者识别。
- "爆炸发生在第几秒？"——时间定位在文本中丢失。
- "这是真实还是生成的音频？"——deepfake 检测需要声学特征。

端到端保留声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026 年生产配方

对新音频理解产品：

- 级联如果：目标是转录，无音乐，无情绪推理。
- AF3 / Qwen-Audio 家族如果：音乐、情绪、多说话者或复杂音频推理。

级联更便宜更简单。端到端更有能力。

### MMAU——音频推理基准

MMAU (Massive Multimodal Audio Understanding) 是 2024-2025 音频推理基准：

- 10,000 跨语音、音乐、环境声音的音频-文本 QA 对。
- 覆盖分类、时间推理、因果推理、开放式 QA。
- 测试级联流水线系统性错过什么。

开放 SOTA (AF3) 在 0.72；前沿专有 ~0.78 (Gemini 2.5 Pro, Claude Opus 4.7)。差距比 VideoMME 的开放-vs-封闭差距小，表明音频-LLM 正在成熟。

## 使用它

`code/main.py`：

- 在 stdlib 中实现 log-Mel spectrogram 计算：窗、朴素 DFT、Mel filter-bank。
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、attention 并发出 N token。
- 玩具任务上级联-vs-端到端对比。

## 交付它

本课产生 `outputs/skill-audio-llm-pipeline-picker.md`。给定音频任务（转录、音乐标签、情绪推理、多说话者 diarization、环境分类），它在级联、端到端 AF3 或混合之间挑选。

## 练习

1. 计算 30 秒片段在 16kHz、25ms 窗口、10ms hop、80 Mel bin 下的 log-Mel spectrogram 维度。48kHz 下如何变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕捉 Whisper 不捕捉的什么音频特征？

3. 64 query 的音频 Q-former vs 32：在什么任务复杂度下 64 回本？32 为哪些节省计算？

4. 阅读 AF3 Section 4 关于按需思考。提出 chain-of-thought 帮助最大的三个音频任务。

5. 使用 AF3 输出实现最小 diarization 流水线。你如何信号说话者变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Log-Mel spectrogram | "Mel features" | Mel filter banks 后 log-magnitude 值的 2D (时间, 频率) 数组 |
| Audio Q-former | "Audio Perceiver" | 从音频编码器输出到固定长度 query 的 cross-attention 瓶颈，喂入 LLM |
| Cascaded | "ASR-then-LLM" | Whisper 转录然后文本 LLM 推理的流水线；丢失声学信息 |
| End-to-end | "Audio-LLM" | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | "Audio AudioSet encoder" | 在 AudioSet 上训练的 SSL transformer；音乐 + 环境声音强 |
| MMAU | "Audio reasoning bench" | 10k 跨语音、音乐、环境的 QA 对；2024 评估标准 |
| On-demand thinking | "Audio CoT" | 模型可在最终答案前可选发出推理 token，提升准确率 3-5 分 |

## 延伸阅读

- [Radford 等人 — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu 等人 — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel 等人 — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang 等人 — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong 等人 — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
