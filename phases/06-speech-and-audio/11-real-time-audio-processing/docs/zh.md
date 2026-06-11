# 实时音频处理

> 批处理流程处理文件。实时流程在下 20 毫秒到来之前处理当前的 20 毫秒。每个对话式 AI、广播演播室和电话机器人都在这个延迟预算下生死存亡。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图），Phase 6 · 04（ASR），Phase 6 · 07（TTS）
**时间：** 约 75 分钟

## 问题

你想要一个感觉有生命力的语音助手。人类对话轮转的延迟约为 ~230 ms（从沉默到回应）。超过 500 ms 感觉像机器人；超过 1500 ms 感觉坏了。2026 年完整的**听 → 理解 → 回应 → 说**循环的预算为：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 ms |
| VAD | 10 ms |
| ASR（流式） | 150 ms |
| LLM（首个 token） | 100 ms |
| TTS（首段音频） | 100 ms |
| 渲染 → 扬声器 | 20 ms |
| **总计** | **~400 ms** |

Moshi（Kyutai，2024）实现了 200 ms 全双工。GPT-4o-realtime（2024）约为 ~320 ms。2022 年的级联流程为 2500 ms。10 倍提升来自三项技术：(1) 处处流式处理，(2) 异步流水线与部分结果，(3) 可中断生成。

## 概念

![流式音频流程：环形缓冲区、VAD 门、打断](../assets/real-time.svg)

**帧 / 块 / 窗口。** 实时音频以固定大小的块流动。常见选择：20 ms（16 kHz 下 320 个采样点）。下游所有环节必须跟上这个节奏。

**环形缓冲区（Ring buffer）。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。防止热路径中的内存分配。大小 ≈ 最大延迟 × 采样率；2 秒 16 kHz 的环形缓冲区 = 32,000 个采样点。

**VAD（Voice Activity Detection，语音活动检测）。** 当无人说话时关闭下游工作。Silero VAD 4.0（2024）在 CPU 上每 30 ms 帧运行 <1 ms。`webrtcvad` 是旧版替代方案。

**流式 ASR。** 音频到达时发出部分转录文本的模型。Parakeet-CTC-0.6B 流式模式（NeMo，2024）在 320 ms 延迟下达到 2–5% WER。Whisper-Streaming（Macháček 等，2023）将 Whisper 分块以实现近流式处理，延迟约 ~2 秒。

**打断（Interruption）。** 当用户在助手说话时发言，你必须 (a) 检测到插话，(b) 停止 TTS，(c) 丢弃剩余的 LLM 输出。全部在 100 ms 内完成，否则用户会感知到助手"聋了"。

**WebRTC Opus 传输。** 20 ms 帧，48 kHz，自适应比特率 8–128 kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的技术栈。

**抖动缓冲区（Jitter buffer）。** 网络数据包乱序/延迟到达。抖动缓冲区重新排序和平滑；太小 → 可听见的间隙，太大 → 延迟。典型值 60–80 ms。

### 常见陷阱

- **线程争用。** Python 的 GIL + 重型模型可能饿死音频线程。使用 C 回调音频库（sounddevice、PortAudio），让 Python 远离热路径。
- **采样率转换延迟。** 流程内部的重采样增加 5–20 ms。要么预先重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是像 Kokoro 这样快的 TTS，首次请求也有 100–200 ms 的预热。缓存模型 + 在第一次真实轮次前用虚拟运行预热。
- **回声消除。** 没有 AEC，TTS 输出会重新进入麦克风并触发 ASR 识别机器人自己的声音。WebRTC AEC3 是开源默认方案。

## 动手实现

### 步骤 1：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16 kHz 下 32,000 个采样点 = 2 秒。

### 步骤 2：VAD 门

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境中替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤 3：流式 ASR

```python
# 通过 NeMo 的 Parakeet-CTC-0.6B 流式处理
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤 4：打断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 插话
        # 然后喂给流式 ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖异步 I/O 和可取消的 TTS 流式传输。WebRTC peerconnection.stop() 在音频轨道上是规范做法。

## 实际应用

2026 年的技术栈：

| 层 | 选择 |
|-------|------|
| 传输 | LiveKit (WebRTC) 或 Pion (Go) |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首个 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **缓冲 500 ms 求安全。** 缓冲区*就是*你的延迟下限。缩小它。
- **不固定线程。** 音频回调在优先级低于 UI 线程的线程上 = 负载下出现 glitch。
- **TTS 块太小。** 低于 200 ms 的块会使声码器伪影可听。320 ms 块是最佳点。
- **没有抖动缓冲区。** 真实网络有抖动；没有平滑你会听到爆音。
- **单次错误处理。** 音频流程必须是防崩溃的。一个异常就会杀死会话。

## 交付产物

保存为 `outputs/skill-realtime-designer.md`。设计带有每个阶段具体延迟预算的实时音频流程。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲区 + 能量 VAD；打印假 10 秒流的各阶段延迟。
2. **中等。** 使用 `sounddevice`，构建一个以 20 ms 帧处理你的麦克风的直通循环，并在每帧打印 VAD 状态。
3. **困难。** 使用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。用 1 kHz 脉冲测量 glass-to-glass 延迟。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Ring buffer | 循环队列 | 固定大小、无锁（或单生产者单消费者加锁）的音频帧 FIFO。 |
| VAD | 静音门 | 标记语音 vs 非语音的模型或启发式方法。 |
| Streaming ASR | 实时 STT | 音频到达时发出部分文本；有界前瞻。 |
| Jitter buffer | 网络平滑器 | 对乱序数据包重新排序的队列；典型 60–80 ms。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| Barge-in | 用户打断 | 系统在 TTS 期间检测到用户语音；必须取消播放。 |
| Full duplex | 同时双向 | 用户和机器人可以同时说话；Moshi 是全双工。 |

## 延伸阅读

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) —— 分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) —— 200 ms 延迟全双工。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) —— 生产级音频智能体编排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad) —— 亚毫秒级 VAD，Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) —— 开源回声消除。
