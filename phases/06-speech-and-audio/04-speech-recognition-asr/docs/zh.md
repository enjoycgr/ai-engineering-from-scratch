# 语音识别（ASR）—— CTC、RNN-T、Attention

> 语音识别是在每个时间步做音频分类，再由一个懂英语和沉默的序列模型粘合起来。CTC、RNN-T 和 attention 是做这件事的三种方式。选一种并理解为什么。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图与 Mel），Phase 5 · 08（文本的 CNN 与 RNN），Phase 5 · 10（Attention）
**时间：** 约 45 分钟

## 问题

你有一段 10 秒、16 kHz 的音频片段。你想要一个字符串："turn on the kitchen lights"。挑战在于结构：音频帧与字符不是一一对应的。单词"okay"可能需要 200 ms 或 1200 ms。沉默穿插在语句中。有些音素比其他音素长。输出 token 的数量事先未知。

三种形式化方法解决了这个问题：

1. **CTC（Connectionist Temporal Classification，连接时序分类）。** 输出每帧的 token 概率，包括一个特殊的 *blank（空白）*。解码时折叠重复项并去掉 blank。非自回归，速度快。wav2vec 2.0、MMS 使用。
2. **RNN-T（Recurrent Neural Network Transducer，循环神经网络转导器）。** 联合网络在给定编码器帧和先前 token 的情况下预测下一个 token。可流式处理。Google 的端侧 ASR、NVIDIA Parakeet 使用。
3. **Attention 编码器-解码器。** 编码器将音频压缩为隐藏状态，解码器通过 cross-attention 自回归地生成 token。Whisper、SeamlessM4T 使用。

2026 年，LibriSpeech test-clean 上的 SOTA WER 是 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差异很小；部署差异很大。

## 概念

![三种 ASR 形式化方法：CTC、RNN-T、attention 编码器-解码器](../assets/asr-formulations.svg)

**CTC 直觉。** 让编码器输出 `T` 个帧级别的 `V+1` 个 token 分布（V 个字符 + blank）。对于长度为 `U < T` 的目标字符串 `y`，任何折叠后为 `y` 的帧对齐都计入。CTC 损失对所有此类对齐求和。推理：每帧 argmax，折叠重复项，移除 blank。

优点：非自回归、可流式处理、零前瞻。缺点：*条件独立性假设*——每帧预测独立于其他帧，因此没有内部语言模型。通过 beam search 或浅融合（shallow fusion）引入外部 LM 来弥补。

**RNN-T 直觉。** 添加一个 *predictor（预测器）* 网络来嵌入 token 历史，以及一个 *joiner（联合器）* 将预测器状态与编码器帧组合成 `V+1` 的联合分布（`+1` 是 null / no-emit）。显式建模了 CTC 忽略的条件依赖。可流式处理，因为每一步只依赖过去的帧和过去的 token。

优点：可流式处理 + 内部 LM。缺点：训练更复杂且内存密集（3D 损失晶格）；RNN-T 损失核函数本身就是一个完整的库类别。

**Attention 编码器-解码器。** 编码器（6-32 层 transformer）作用于 log-mel 帧。解码器（6-32 层 transformer）通过 cross-attention 到编码器输出来自回归地生成 token。没有对齐约束——attention 可以看向音频的任何位置。非流式，除非你限制 attention（分块 Whisper-Streaming，2024）。

优点：离线 ASR 最高质量，使用标准 seq2seq 工具易于训练。缺点：自回归延迟与输出长度成正比；没有工程优化无法流式处理。

### WER：唯一的数字

**Word Error Rate（词错误率）** = `(S + D + I) / N`，其中 S=替换数，D=删除数，I=插入数，N=参考词数。匹配词级别的 Levenshtein 编辑距离。越低越好。WER 高于 20% 通常不可用；低于 5% 对朗读语音达到人类水平。2026 年标准基准上的数字：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 大小 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

这些都是基于编码器-解码器或 RNN-T 的。纯 CTC 系统（wav2vec 2.0）在 test-clean 上约为 1.8–2.1%。

## 动手实现

### 步骤 1：greedy CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 每帧概率向量的列表
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：折叠连续重复项，去掉 blank。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境使用带 LM 融合的前缀树 beam search；这是概念骨架。

### 步骤 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤 4：Whisper 推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026 年最强通用 ASR 的一行代码。在 24 GB GPU 上以约 20 倍实时速度运行。

### 步骤 5：Parakeet 或 wav2vec 2.0 的流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式 ASR 需要分块编码器 attention 和传递状态；使用支持它的库（NeMo 用于 Parakeet，带 `chunk_length_s` 的 `transformers` pipeline）。

## 实际应用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 英语、离线、最高质量 | Whisper-large-v3-turbo |
| 多语言、鲁棒 | SeamlessM4T v2 |
| 流式、低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘、移动端、<500 ms 延迟 | Whisper-Tiny 量化或 Moonshine (2024) |
| 长音频 | 带 VAD 分块的 Whisper（WhisperX） |
| 领域特定（医疗、法律） | 微调 wav2vec 2.0 + 领域 LM 融合 |

## 2026 年仍会出现的陷阱

- **没有 VAD。** 在静音上运行 Whisper 会产生幻觉（"Thanks for watching!"）。始终用 VAD 把关。
- **字符 vs 词 vs 子词 WER。** 报告归一化后（小写、去掉标点）的词级别 WER。
- **语言识别漂移。** Whisper 的自动 LID 会将噪声片段误路由到日语或威尔士语；当你知道语言时强制 `language="en"`。
- **长片段不分块。** Whisper 有 30 秒窗口。对任何更长的内容使用 `chunk_length_s=30, stride=5`。

## 交付产物

保存为 `outputs/skill-asr-picker.md`。为给定部署目标选择模型、解码策略、分块和 LM 融合。

## 练习

1. **简单。** 运行 `code/main.py`。它对人工构造的 CTC 输出做 greedy 解码并计算与参考的 WER。
2. **中等。** 正确实现步骤 2 中的前缀树 beam search（考虑 blank 合并规则）。在 10 个示例的合成数据集上与 greedy 比较。
3. **困难。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。计算前 100 段的 WER。与已发表数字比较。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| CTC | 带 blank token 的损失 | 对所有帧到 token 对齐求边际；非自回归。 |
| RNN-T | 流式损失 | CTC + 下一个 token 预测器；处理词序。 |
| Attention 编码-解码 | Whisper 风格 | 编码器 + cross-attention 解码器；离线最佳质量。 |
| WER | 你报告的那个数字 | 词级别的 `(S+D+I)/N`。 |
| Blank | 空白 | CTC 中标记"此帧无输出"的特殊 token。 |
| LM fusion | 外部语言模型 | beam search 中加入加权 LM 对数概率。 |
| VAD | 静音门 | 语音活动检测器；切除非语音部分。 |

## 延伸阅读

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 论文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) —— RNN-T 论文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) —— 2022 年经典论文；v3-turbo 扩展于 2024 年。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) —— 2026 Open ASR 排行榜领先者。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) —— 25+ 模型的实时基准。
