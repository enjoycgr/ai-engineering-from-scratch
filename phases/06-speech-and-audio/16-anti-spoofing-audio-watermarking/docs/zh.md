# 语音反欺骗与音频水印 —— ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆的交付速度超过了防御措施。2026 年的生产语音系统需要两样东西：一个分类真实 vs 伪造语音的检测器（AASIST、RawNet2），以及一个能经受压缩和编辑的水印（AudioSeal）。两者都上，否则不要交付语音克隆。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 06（说话人识别），Phase 6 · 08（语音克隆）
**时间：** 约 75 分钟

## 问题

三种相关防御：

1. **反欺骗 / 深度伪造检测。** 给定一段音频，它是合成的还是真实的？ASVspoof 基准（ASVspoof 2019 → 2021 → 5）是黄金标准。
2. **音频水印。** 在生成的音频中嵌入人耳不可察觉的信号，检测器后续可以提取。AudioSeal（Meta）和 WavMark 是开源选项。
3. **认证溯源。** 音频文件 + 元数据的加密签名。C2PA / Content Authenticity Initiative。

检测处理不合作的对抗者。水印处理合规性——AI 生成的音频应可被识别为 AI 生成。2026 年两者都需要。

## 概念

![反欺骗 vs 水印 vs 溯源 —— 三层防御](../assets/spoofing-watermark.svg)

### ASVspoof 5 —— 2024-2025 基准

相比之前版本的最大变化：

- **众包数据**（非录音棚纯净）—— 真实条件。
- **~2000 名说话人**（之前约 100 名）。
- **32 种攻击算法。** TTS + 语音转换 + 对抗扰动。
- **两个赛道。** Countermeasure（CM）独立检测；Spoofing-robust ASV（SASV）用于生物识别系统。

ASVspoof 5 上的 SOTA：~7.23% EER。在较早的 ASVspoof 2019 LA 上：0.42% EER。真实世界部署：在野外片段上预期 5-10% EER。

### AASIST 和 RawNet2 —— 检测模型家族

**AASIST**（2021，更新至 2026）。基于谱特征的图注意力网络。ASVspoof 5 countermeasure 任务上的当前 SOTA。

**RawNet2。** 原始波形上的卷积前端 + TDNN 主干。更简单的基线；通过微调仍有竞争力。

**NeXt-TDNN + SSL 特征。** 2025 变体：ECAPA 风格 + WavLM 特征 + focal loss。在 ASVspoof 2019 LA 上达到 0.42% EER。

### AudioSeal —— 2024 水印默认选择

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 2024 年 12 月）。关键设计：

- **局部化。** 以 16 kHz 采样分辨率（1/16000 s）逐帧检测水印。
- **生成器 + 检测器联合训练。** 生成器学习嵌入不可察觉的信号；检测器通过增强学习找到它。
- **鲁棒。** 经受 MP3 / AAC 压缩、EQ、±10% 变速、+10 dB SNR 噪声混合。
- **快速。** 检测器以 485 倍实时速度运行；比 WavMark 快 1000 倍。
- **容量。** 16 位有效载荷（可编码模型 ID、生成时间戳、用户 ID）可嵌入每个话语。

### WavMark

AudioSeal 之前的开源基线。可逆神经网络，32 bits/sec。问题：

- 同步暴力搜索慢。
- 可被高斯噪声或 MP3 压缩移除。
- 不友好于实时场景。

### WaveVerify（2025 年 7 月）

解决 AudioSeal 的弱点——特别是时间操纵（反转、变速）。使用基于 FiLM 的生成器 + Mixture-of-Experts 检测器。在标准攻击上与 AudioSeal 竞争；处理时间编辑。

### 对抗者利用的差距

来自 AudioMarkBench："在 pitch shift 下，所有水印的比特恢复准确率低于 0.6，表明接近完全移除。"**Pitch shift 是通用攻击。** 没有 2026 年的水印能完全抵抗激进的 pitch 修改。这就是为什么你需要检测（AASIST）alongside 水印。

### C2PA / Content Authenticity Initiative

不是 ML 技术——一种清单格式。音频文件携带关于创建工具、作者、日期的加密签名元数据。Audobox / Seamless 使用它。对溯源有好处；如果坏人重新编码并剥离元数据，则无能为力。

## 动手实现

### 步骤 1：简单谱特征检测器（玩具）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音通常具有异常平坦的高频能量。生产检测器使用 AASIST，不是这个。但直觉成立。

### 步骤 2：AudioSeal 嵌入 + 检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: [0, 1] 中的 float —— 水印存在概率
# decoded_payload: 16 位；与嵌入的有效载荷比对
```

### 步骤 3：评估 —— EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 步骤 4：生产集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成输出：(1) 水印，(2) 签名清单，(3) 符合保留策略的审计日志。

## 实际应用

| 用例 | 防御 |
|------|------|
| 交付 TTS / 语音克隆 | 每次输出嵌入 AudioSeal（不可协商）|
| 生物识别语音解锁 | AASIST + ECAPA 集成；活体挑战 |
| 呼叫中心欺诈检测 | 对 20% 来电样本运行 AASIST |
| 播客真实性 | 上传时 C2PA 签名，AI 生成时 AudioSeal |
| 研究 / 训练检测器 | ASVspoof 5 训练/开发/评估集 |

## 陷阱

- **水印但检测器从未运行。** 毫无意义。在 CI 中交付检测器。
- **检测未校准。** 在 ASVspoof LA 上训练的 AASIST 会过拟合；真实世界准确率下降。在你的领域上校准。
- **Pitch shift 差距。** 激进 pitch shift 移除大多数水印。准备检测回退方案。
- **元数据剥离并重托管。** C2PA 通过重新编码就能轻易绕过。始终同时添加加密 + 感知（水印）防御。
- **活体作为检测。** 要求用户说一个随机短语。防止重放攻击但不防止实时克隆。

## 交付产物

保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印、溯源清单和操作手册。

## 练习

1. **简单。** 运行 `code/main.py`。在合成音频上运行玩具检测器 + 玩具水印嵌入/检测。
2. **中等。** 安装 `audioseal`，在 TTS 输出中嵌入 16 位有效载荷，重新解码。用噪声损坏音频并测量比特恢复准确率。
3. **困难。** 在 ASVspoof 2019 LA 上微调 RawNet2 或 AASIST。测量 EER。在一组 F5-TTS 生成片段的留出集上测试——看 OOD 检测如何退化。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| ASVspoof | 基准 | 双年挑战赛；2024 = ASVspoof 5。 |
| CM (countermeasure) | 检测器 | 分类器：真实语音 vs 合成/转换。 |
| SASV | 说话人验证 + CM | 集成的生物识别 + 欺骗检测。 |
| AudioSeal | Meta 水印 | 局部化，16 位有效载荷，比 WavMark 快 485 倍。 |
| Bit Recovery Accuracy | 水印存活率 | 攻击后恢复的 payload 比特比例。 |
| C2PA | 溯源清单 | 关于创建/作者身份的加密元数据。 |
| AASIST | 检测器家族 | 基于图注意力的反欺骗 SOTA。 |

## 延伸阅读

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) —— 当前基准。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) —— 水印默认选择。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) —— 针对时间攻击的 MoE 检测器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) —— SOTA 检测主干。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) —— 鲁棒性评估。
- [C2PA specification](https://c2pa.org/specifications/specifications/) —— 溯源清单格式。
