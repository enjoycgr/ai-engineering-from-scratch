# 语音活动检测与轮替决策 —— Silero、Cobra 与 Flush Trick

> 每个语音助手的生死取决于两个决策：用户现在是否在说话？他们说完了吗？VAD 回答第一个。轮替检测（VAD + 静音保持 + 语义端点模型）回答第二个。任何一个出错，助手要么打断用户，要么永远说个不停。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 11（实时音频处理），Phase 6 · 12（语音助手流程）
**时间：** 约 45 分钟

## 问题

语音助手在每个 20 ms 的音频块上都要做三个不同决策：

1. **这一帧是语音吗？** —— VAD。逐帧二分类。
2. **用户开始说话了吗？** —— 起始检测（onset detection）。
3. **用户说完了吗？** —— 端点检测（end-pointing，轮替结束）。

朴素方案（能量阈值）在任何噪声面前都会失效——交通噪音、键盘敲击、人群嘈杂。2026 年的答案是：Silero VAD（开源、深度学习）+ 轮替检测模型（语义端点）+ VAD 校准的静音保持。

## 概念

![VAD 级联：能量门 → Silero → 轮替检测器 → flush trick](../assets/vad-turn-taking.svg)

### 三级 VAD 级联

**第一层：能量门。** 成本最低。RMS 阈值设为 -40 dBFS。能过滤明显的静音，但任何超过阈值的噪声都会触发。

**第二层：Silero VAD**（2020-2026，MIT）。100 万参数。在 6000+ 种语言上训练。单 CPU 线程处理 30 ms 片段约 1 ms。5% FPR 下 TPR 为 87.7%。开源默认选择。

**第三层：语义轮替检测器。** LiveKit 的轮替检测模型（2024-2026）或你自己的小型分类器。区分"句子中间的停顿"与"说完了"。使用语言上下文（语调 + 近期词汇），而不仅仅是静音。

### 关键参数及其默认值

- **阈值（Threshold）。** Silero 输出概率；默认以 > 0.5 分类为语音，或以 > 0.3（更敏感）分类。阈值越低 = 首词截断越少，假阳性越多。
- **最短语音时长（Minimum speech duration）。** 拒绝短于 250 ms 的语音——通常是咳嗽声或椅子噪音。
- **静音保持（Silence hangover，端点检测）。** VAD 返回 0 后，等待 500-800 ms 再宣布轮替结束。太短 → 打断用户。太长 → 感觉迟钝。
- **预卷缓冲区（Pre-roll buffer）。** VAD 触发前保留 300-500 ms 音频。防止"hey"被截断。

### Flush Trick（Kyutai，2025）

流式 STT 模型有前瞻延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。正常情况下，语音结束后要等这么久才能拿到转录。Flush trick：当 VAD 检测到语音结束时，**向 STT 发送 flush 信号**强制立即输出。STT 以约 4 倍实时速度处理，因此 500 ms 缓冲在约 125 ms 内完成。

端到端：125 ms VAD + flush STT = 对话级延迟。

### 2026 年 VAD 对比

| VAD | 5% FPR 下 TPR | 延迟 | 许可证 |
|-----|--------------|------|--------|
| WebRTC VAD（Google，2013）| 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026）| 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice）| 98.9% | ~1 ms | 商业 |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正确的默认选择。Cobra 是合规/精度的升级选项。纯能量 VAD 在 2026 年生产环境中没有立足之地。

## 动手实现

### 步骤 1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：轮替结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：flush trick 骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能生效。Whisper streaming 不支持——它是基于块的，始终等待分块。

## 实际应用

| 场景 | VAD 选择 |
|------|---------|
| 开放、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机）| Silero VAD ONNX |
| 研究 / 说话人分割 | pyannote segmentation |
| 零依赖回退 | WebRTC VAD（遗留）|
| 需要轮替结束质量 | Silero + LiveKit turn-detector 分层 |

经验法则：除非别无选择，否则永远不要交付纯能量 VAD。

## 陷阱

- **固定阈值。** 安静环境有效，嘈杂环境失效。要么在设备上校准，要么切换到 Silero。
- **静音保持太短。** 助手在句子中间打断。500-800 ms 是对话语音的最佳平衡点。
- **静音保持太长。** 感觉迟钝。与目标用户进行 A/B 测试。
- **没有预卷缓冲区。** 丢失用户音频的前 200-300 ms。始终保留滚动预卷。
- **忽略语义端点。** "嗯，让我想想..."包含长停顿。用户讨厌在思考中途被打断。使用 LiveKit 的 turn-detector 或类似方案。

## 交付产物

保存为 `outputs/skill-vad-tuner.md`。为特定工作负载选择 VAD 模型、阈值、保持时间、预卷和轮替检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟了一段语音 + 静音 + 语音 + 咳嗽序列，并测试三级 VAD。
2. **中等。** 安装 `silero-vad`，处理一段 5 分钟录音，调节阈值以最小化首词截断和误触发。报告精确率/召回率。
3. **困难。** 构建一个迷你轮替检测器：Silero VAD + 一个作用于最近 10 个词 embedding 的 3 层 MLP（使用 sentence-transformers）。在一个手工标注的轮替结束数据集上训练。比纯 Silero 提升 10% F1。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| VAD | 语音检测器 | 逐帧二分类：这是语音吗？ |
| Turn detection | 端点检测 | VAD + 静音保持 + 语义端点。 |
| Silence hangover | 语音后等待 | 宣布轮替结束前等待的时间；500-800 ms。 |
| Pre-roll | 语音前缓冲 | VAD 触发前保留 300-500 ms 音频。 |
| Flush trick | Kyutai 技巧 | VAD → flush-STT → 延迟从 500 ms 降到 125 ms。 |
| Semantic endpoint | "他们是故意停下的吗？" | 看词汇而非静音的 ML 分类器。 |
| TPR @ FPR 5% | ROC 点 | VAD 标准基准；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) —— 参考级开源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) —— 商业精度领先者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) —— 亚 200 ms 工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) —— 生产级语义端点。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) —— 遗留基线。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) —— 说话人分割级分割。
