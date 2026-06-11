# 声音克隆与声音转换

> 声音克隆用别人的声音读你的文本。声音转换把你的声音改写成别人的声音，同时保留你说的内容。两者都基于同一个分解：将说话人身份与内容分离。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 06（说话人识别），Phase 6 · 07（TTS）
**时间：** 约 75 分钟

## 问题

2026 年，一段 5 秒的音频就足以在消费级 GPU 上产生任何人的高质量克隆。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供零样本或少样本克隆。这项技术是福音（无障碍 TTS、配音、辅助语音）也是武器（诈骗电话、政治深度伪造、IP 盗窃）。

两个密切相关的任务：

- **声音克隆（TTS 侧）：** 文本 + 5 秒参考声音 → 该声音的音频。
- **声音转换（语音侧）：** 源音频（A 说 X）+ B 的参考声音 → B 说 X 的音频。

两者都将波形分解为（内容、说话人、韵律），然后将一个来源的内容与另一个来源的说话人重新组合。

2026 年你现在交付时的关键约束：**在欧盟（AI 法案，2026 年 8 月生效）和加利福尼亚州（AB 2905，2025 年生效），水印和同意门是法律要求的**。你的流程必须输出不可听水印，并拒绝非自愿克隆。

## 概念

![声音克隆 vs 转换：分解、交换说话人、重新组合](../assets/voice-cloning.svg)

**零样本克隆。** 将 5 秒片段传给一个在数千个说话人上训练过的模型。说话人编码器将片段映射为说话人嵌入；TTS 解码器以该嵌入 + 文本为条件。

使用者：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**少样本微调。** 录制目标声音 5-30 分钟。用 LoRA 微调基础模型一小时。质量从"还行"跃升到"无法区分"。Coqui 和 ElevenLabs 都支持这种模式；社区将其与 F5-TTS 一起使用。

**声音转换（VC）。** 两个家族：

- **识别-合成。** 运行类 ASR 模型提取内容表示（例如软音素后验概率，PPG），然后用目标说话人嵌入重新合成。对语言和口音鲁棒。KNN-VC（2023）、Diff-HierVC（2023）使用。
- **解耦。** 训练一个自编码器，在瓶颈处将内容、说话人和韵律分离到潜在空间中。推理时交换说话人嵌入。质量较低但更快。AutoVC（2019）、VITS-VC 变体使用。

**基于神经编解码器的克隆（2024+）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox——将音频视为来自 SoundStream / EnCodec 的离散 token，在 codec token 上训练大型自回归或流匹配模型。短提示上的质量可与 ElevenLabs 媲美。

### 伦理部分，不是附加组件

**水印。** PerTh（Perth）和 SilentCipher（2024）在音频中不可感知地嵌入约 16-32 位 ID。在重新编码、流式传输和常见编辑后仍然存活。生产就绪的开源方案。

**同意门。** 必须将每次克隆输出与可验证的同意记录配对。"我，Rohit，于 2026-04-22，授权将此声音用于 X 目的。"存储在防篡改日志中。

**检测。** AASIST、RawNet2 和 Wav2Vec2-AASIST 作为检测器发布。ASVspoof 2025 挑战赛公布了针对 ElevenLabs、VALL-E 2 和 Bark 输出的最先进检测器的 EER 为 0.8–2.3%。

### 数字（2026）

| 模型 | 零样本？ | SECS（目标相似度） | WER（可懂度） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

SECS > 0.70 对大多数听众来说通常与目标无法区分。

## 动手实现

### 步骤 1：用识别-合成分解（main.py 中的纯代码演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上简单；实现量在于 `tts_model` 和说话人编码器。

### 步骤 2：使用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录必须完全匹配音频；不匹配会破坏对齐。

### 步骤 3：使用 KNN-VC 进行声音转换

```python
import torch
from knnvc import KNNVC  # 2023 模型, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 运行 WavLM 为源和目标池提取每帧嵌入，然后用池中的最近邻替换每帧源帧。非参数化，用一分钟目标语音即可工作。

### 步骤 4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # 返回 payload 字节
```

约 32 位 payload，在 MP3 重新编码和轻度噪声后可检测。

### 步骤 5：同意门

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "需要签名同意"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 实际应用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 5 秒零样本克隆，开源 | F5-TTS 或 OpenVoice v2 |
| 商业生产克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 声音转换（改写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 陷阱

- **参考转录不对齐。** F5-TTS 及类似模型要求参考文本完全匹配参考音频，包括标点。
- **混响参考。** 回声会杀死克隆。录制干声，近距离麦克风。
- **情绪不匹配。** 训练参考"愉快"会产生一切内容的愉快克隆。匹配参考情绪与目标用途。
- **语言泄漏。** 克隆英语说话人然后让模型说法语通常仍携带口音；使用跨语言模型（XTTS、VALL-E X）。
- **没有水印。** 从 2026 年 8 月起在欧盟法律上无法出货。

## 交付产物

保存为 `outputs/skill-voice-cloner.md`。设计带有同意门 + 水印 + 质量目标的克隆或转换流程。

## 练习

1. **简单。** 运行 `code/main.py`。通过计算交换前后两个"说话人"之间的余弦相似度来演示说话人嵌入交换。
2. **中等。** 使用 OpenVoice v2 克隆你自己的声音。测量参考和克隆之间的 SECS。通过 Whisper 测量 CER。
3. **困难。** 将 SilentCipher 水印应用于 20 个克隆，通过 128 kbps MP3 编码+解码运行它们，检测 payload。报告位准确率。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Zero-shot clone | 5 秒就够了 | 预训练模型 + 说话人嵌入；无需训练。 |
| PPG | 音素后验图 | 用作语言无关内容表示的每帧 ASR 后验概率。 |
| KNN-VC | 最近邻转换 | 用最近邻目标池帧替换每帧源帧。 |
| Neural codec TTS | VALL-E 风格 | EnCodec/SoundStream token 上的自回归模型。 |
| Watermark | 不可听签名 | 嵌入音频中的位，在重新编码后存活。 |
| SECS | 克隆保真度 | 目标和克隆说话人嵌入之间的余弦相似度。 |
| AASIST | 深度伪造检测器 | 反欺骗模型；检测合成语音。 |

## 延伸阅读

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) —— 开源 SOTA 零样本克隆。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 和 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) —— 神经编解码器 TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) —— 基于解耦的声音转换。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) —— 基于检索的 VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) —— 生产就绪的 32 位音频水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/) —— 检测器 vs 合成器的军备竞赛，2026 年更新。
