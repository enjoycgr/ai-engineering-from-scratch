# 音频评估 —— WER、MOS、UTMOS、MMAU、FAD 与开放排行榜

> 你无法交付无法衡量的东西。本课程列出 2026 年每项音频任务的指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、往返 ASR-WER）、音频语言模型（MMAU、LongAudioBench）、音乐（FAD、CLAP）和说话人（EER）。以及你对照的排行榜。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 04、06、07、09、10；Phase 2 · 09（模型评估）
**时间：** 约 60 分钟

## 问题

每项音频任务都有多个指标，每个衡量不同维度。使用错误指标是你交付一个在仪表盘上看起来很好、在生产中很糟糕的模型的方式。2026 年的标准清单：

| 任务 | 主要指标 | 次要指标 |
|------|---------|---------|
| ASR | WER | CER · RTFx · 首 token 延迟 |
| TTS | MOS / UTMOS | SECS · 往返 ASR-WER · CER · TTFA |
| 语音克隆 | SECS（ECAPA cosine）| MOS · CER |
| 说话人验证 | EER | minDCF · 工作点 FAR / FRR |
| 说话人分割 | DER | JER · 说话人混淆 |
| 音频分类 | top-1 · mAP | macro F1 · 每类召回率 |
| 音乐生成 | FAD | CLAP · 听众 panel MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| 流式 S2S | 延迟 P50/P95 | WER · MOS |

## 概念

![音频评估矩阵 —— 指标 vs 任务 vs 2026 排行榜](../assets/eval-landscape.svg)

### ASR 指标

**WER（Word Error Rate，词错误率）。** `(S + D + I) / N`。评分前小写、去除标点、归一化数字。使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。`% 3C 5%` = 人类级朗读语音。

**CER（Character Error Rate，字错误率）。** 相同公式，字符级。用于声调语言（中文、粤语），其中词切分有歧义。

**RTFx（逆实时因子）。** 每墙钟秒处理的音频秒数。越高越好。Parakeet-TDT 达到 3380×。Whisper-large-v3 约 30×。

**首 token 延迟。** 从音频输入到首个转录 token 的墙钟时间。对流式至关重要。Deepgram Nova-3：~150 ms。

### TTS 指标

**MOS（Mean Opinion Score，平均意见分）。** 1-5 分人工评分。黄金标准但慢。每样本 20+ 听众，每模型 100+ 样本。

**UTMOS（2022-2026）。** 学习的 MOS 预测器。在标准基准上与人工 MOS 相关性约 0.9。F5-TTS：UTMOS 3.95；真值：4.08。

**SECS（Speaker Encoder Cosine Similarity，说话人编码器余弦相似度）。** 用于语音克隆。参考与克隆输出之间的 ECAPA embedding cosine。`% 3E 0.75` = 可识别的克隆。

**往返 ASR-WER。** 在 TTS 输出上运行 Whisper，对输入文本计算 WER。捕获可懂度回归。2026 SOTA：`% 3C 2%` CER。

**TTFA（time-to-first-audio，首音频时间）。** 墙钟延迟。Kokoro-82M：~100 ms；F5-TTS：~1 s。

### 语音克隆专用

**SECS + MOS + CER** 三联。SECS 高但 MOS 低的克隆意味着音色对但不自然；相反意味着自然但说话人不对。

### 说话人验证

**EER（Equal Error Rate，等错误率）。** FAR 等于 FRR 时的阈值。ECAPA 在 VoxCeleb1-O 上：0.87%。

**minDCF（最小检测成本）。** 选定工作点（通常 FAR=0.01）的加权成本。比 EER 更贴近生产实际。

### 说话人分割

**DER（Diarization Error Rate，分割错误率）。** `(FA + Miss + Confusion) / total_speaker_time`。漏检语音 + 误报语音 + 说话人混淆，各占比例。AMI 会议：DER ~10-20% 是现实的。pyannote 3.1 + Precision-2 商业：良好录制音频上 `% 3C 10%` DER。

**JER（Jaccard Error Rate）。** DER 的替代，对短片段偏置更鲁棒。

### 音频分类

多标签：**mAP（mean Average Precision，平均精确率）** 跨所有类。AudioSet：BEATs-iter3 为 0.548 mAP。

多类别互斥：**top-1、top-5 准确率**。Speech Commands v2：99.0% top-1（Audio-MAE）。

不平衡：**macro F1** + **每类召回率**。报告每类——总体准确率掩盖了哪些类失败。

### 音乐生成

**FAD（Fréchet Audio Distance）。** 真实与生成音频的 VGGish-embedding 分布距离。MusicGen-small 在 MusicCaps 上：4.5。MusicLM：4.0。越低越好。

**CLAP Score。** 使用 CLAP embedding 的文本-音频对齐分数。`% 3E 0.3` = 合理对齐。

**听众 panel MOS。** 消费级音乐的最终裁决。Suno v5 在 TTS Arena 上 ELO 1293（来自成对人类偏好）。

### 音频语言基准

**MMAU（Massive Multi-Audio Understanding）。** 1 万音频-QA 对。

**MMAU-Pro。** 1800 个困难项目，四类：语音 / 声音 / 音乐 / 多音频。4 选 1 随机机会 25%。Gemini 2.5 Pro 总体约 60%；多音频约 22%（所有模型）。

**LongAudioBench。** 多分钟片段带语义查询。Audio Flamingo Next 击败 Gemini 2.5 Pro。

**AudioCaps / Clotho。** 字幕基准。SPICE、CIDEr、FENSE 指标。

### 流式语音到语音

**延迟 P50 / P95 / P99。** 从用户语音结束到首个可听响应的墙钟时间。Moshi：200 ms；GPT-4o Realtime：300 ms。

**输出上的 WER / MOS。**

**打断响应性。** 从用户打断到助手静音的时间。目标 `% 3C 150` ms。

### 2026 年排行榜

| 排行榜 | 追踪内容 | URL |
|--------|---------|-----|
| Open ASR Leaderboard (HF) | 英语 + 多语言 + 长文本 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 英语 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，来自成对投票的 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐 LALM | （在 MMAU 内）|
| HEAR benchmark | 自监督音频 | `hearbenchmark.com` |

## 动手实现

### 步骤 1：带归一化的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步骤 2：TTS 往返 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤 3：语音克隆的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤 4：音乐生成的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤 5：说话人验证的 EER（与 Lesson 6 相同代码）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 实际应用

每次部署都搭配固定的评估 harness，在每次模型更新时运行。三条铁律：

1. **评分前归一化。** 小写、去除标点、展开数字。报告归一化规则。
2. **报告分布，而非平均。** 延迟报告 P50/P95/P99。分类报告每类召回率。MMAU 报告每类别。
3. **运行一个标准公共基准。** 即使你的生产数据不同，在 Open ASR / TTS Arena / MMAU 上报告让审稿人能进行同类比较。

## 陷阱

- **UTMOS 外推。** 在 VCTK 风格干净语音上训练；对嘈杂 / 克隆 / 情感音频评分差。
- **MOS panel 偏置。** 20 个 Amazon Mechanical Turk 工人 ≠ 20 个目标用户。如果 stakes 高，为领域 panel 付费。
- **FAD 依赖参考集。** 跨模型比较时使用相同的参考分布。
- **总体 WER。** 总体 5% WER 可能掩盖口音语音上 30% 的 WER。按人口统计切片报告。
- **公共基准饱和。** 大多数前沿模型在标准基准上接近天花板。构建反映你流量的内部留出集。

## 交付产物

保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型发布选择指标、基准和报告格式。

## 练习

1. **简单。** 运行 `code/main.py`。在玩具输入上计算 WER / CER / EER / SECS / 类 FAD / 类 MMAU。
2. **中等。** 构建一个 TTS 往返 WER harness。在 Kokoro 或 F5-TTS 输出上运行 Whisper。在 50 个提示上计算 WER。标记 WER `% 3E 10%` 的提示。
3. **困难。** 在 MMAU-Pro 语音 + 多音频子集（各 50 项）上评分你的 Lesson 10 LALM 选择。报告每类别准确率并与发布数字比较。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| WER | ASR 分数 | 归一化后词级 `(S+D+I)/N`。 |
| CER | 字符 WER | 用于声调语言或字符级系统。 |
| MOS | 人工意见 | 1-5 分；20+ 听众 × 100 样本。 |
| UTMOS | ML MOS 预测器 | 学习模型；与人工 MOS 相关性约 0.9。 |
| SECS | 语音克隆相似度 | 参考与克隆之间的 ECAPA cosine。 |
| EER | 说话人验证分数 | FAR = FRR 时的阈值。 |
| DER | 分割分数 | `(FA + Miss + Confusion) / total`。 |
| FAD | 音乐生成质量 | VGGish embedding 上的 Fréchet 距离。 |
| RTFx | 吞吐量 | 每墙钟秒的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) —— 带归一化工具的 WER/CER 库。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) —— 学习的 MOS 预测器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) —— 音乐生成标准。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) —— 2026 实时排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) —— 人工投票 TTS 排行榜。
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) —— LALM 推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/) —— 音频自监督基准。
