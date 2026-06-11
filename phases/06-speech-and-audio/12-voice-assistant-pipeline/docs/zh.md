# 构建语音助手流程 —— Phase 6 顶点项目

> 将 Lessons 01-11 的所有内容缝合在一起。构建一个能听、能推理、能回话的语音助手。2026 年，这是一个已解决的工程问题，不是研究问题——但集成细节决定了它是否能出货。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 04、05、06、07、11；Phase 11 · 09（Function Calling）；Phase 14 · 01（Agent Loop）
**时间：** 约 120 分钟

## 问题

构建一个端到端助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始/结束。
3. 流式转录。
4. 将转录文本传给能调用工具（计时器、天气、日历）的 LLM。
5. 将 LLM 文本流式传输给 TTS。
6. 将音频播放给用户。
7. 如果用户在中途打断，则停止。

延迟目标：在笔记本 CPU 上，用户说完后 800 ms 内发出第一个 TTS 音频字节。质量目标：不遗漏单词、不在静音上产生幻觉字幕、没有声音克隆泄漏、没有提示注入成功。

## 概念

![语音助手流程：麦克风 → VAD → STT → LLM+工具 → TTS → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频捕获。** 麦克风 → 16 kHz 单声道 → 20 ms 块。Python 中通常用 `sounddevice`，生产环境中用原生 AudioUnit/ALSA/WASAPI。
2. **VAD（Lesson 11）。** Silero VAD @ 阈值 0.5，最短语音 250 ms，静音保持 500 ms。发出"开始"和"结束"信号。
3. **流式 STT（Lessons 4-5）。** Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）。部分 + 最终转录文本。
4. **带工具调用的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具 JSON 模式。流式 token。
5. **流式 TTS（Lesson 7）。** Kokoro-82M（最快开源）或 Cartesia Sonic（商业）。在 20 个 LLM token 后启动 TTS。
6. **播放。** 扬声器输出；低带宽网络用 opus 编码。
7. **打断处理器。** 如果 TTS 播放期间 VAD 触发，停止播放，取消 LLM，重启 STT。

### 你会遇到的三种失败模式

1. **首词截断。** VAD 开始晚了一拍。用户的"hey"缺失了。起始阈值设为 0.3，而不是 0.5。
2. **中途打断混乱。** 用户打断后 LLM 继续生成；助手盖过用户说话。将 VAD → 取消 LLM 连线。
3. **静音幻觉。** Whisper 在静音预热帧上输出"Thanks for watching"。始终用 VAD 把关。

### 2026 年生产参考技术栈

| 技术栈 | 延迟 | 许可证 | 备注 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | 商业 API | 2026 年行业默认 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | 大部分开源 | DIY 友好 |
| Moshi（全双工） | 200-300 ms | CC-BY 4.0 | 单一模型；不同架构，Lesson 15 |
| Vapi / Retell（托管） | 300-500 ms | 商业 | 最快上线；定制有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线 | 开源 | 隐私 / 边缘 |

## 动手实现

### 步骤 1：带分块的麦克风捕获（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤 2：VAD 门控的轮转捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤 3：流式 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤 4：LLM 循环内的工具调用

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤 5：打断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 实际应用

参见 `code/main.py` 了解可运行的模拟，它用 stub 模型连接了所有七个组件，因此即使没有硬件也能看到流程形状。对于真实实现，将 stub 替换为：

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` 或 `openai-whisper`
- `openai` (`gpt-4o`) 或 `anthropic`
- `kokoro` 或 `cartesia`
- `sounddevice` 用于 I/O

## 陷阱

- **永远记录 PII。** 完整轮转音频在大多数司法管辖区都是 PII。30 天保留期，静态加密。
- **没有插话。** 用户会打断。你的助手必须停止说话。
- **TTS 阻塞。** 同步 TTS 阻塞事件循环。使用异步或单独线程。
- **没有工具调用错误处理。** 工具会失败。LLM 必须收到错误 + 重试一次，然后优雅降级。
- **过度激进的幻觉过滤器。** 过度过滤会导致助手重复"I can't help with that."过滤不足则它会说任何话。在留出集上调参。
- **没有唤醒词选项。** 始终监听是隐私责任。添加唤醒词门（Porcupine 或 openWakeWord）。

## 交付产物

保存为 `outputs/skill-voice-assistant-architect.md`。给定预算 + 规模 + 语言 + 合规约束，产出完整的技术栈规格。

## 练习

1. **简单。** 运行 `code/main.py`。它用 stub 模块模拟一次完整的端到端轮转，并打印每阶段延迟。
2. **中等。** 将 STT stub 替换为预录制 `.wav` 上的真实 Whisper 模型。测量 WER 和端到端延迟。
3. **困难。** 添加工具调用：实现 `get_weather`（任何 API）和 `set_timer`。让 LLM 通过工具路由，并验证当用户说"set a 5 minute timer"时正确的函数触发，且口头回复确认了它。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Turn | 一轮用户 + 助手往返 | 一次 VAD 边界内的用户语音 + 一次 LLM-TTS 回应。 |
| Barge-in | 打断 | 用户在助手说话时发言；助手停止。 |
| Wake word | "Hey assistant" | 短关键词检测器；Porcupine、Snowboy、openWakeWord。 |
| End-pointing | 轮转结束 | VAD + 最小静音决策，判断用户已说完。 |
| Pre-roll | 语音前缓冲区 | VAD 触发前保留 200-400 ms 音频，避免首词截断。 |
| Tool call | 函数调用 | LLM 发出 JSON；运行时调度；结果回环到循环中。 |

## 延伸阅读

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) —— 生产级参考。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) —— DIY 友好框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) —— 托管的语音原生路径。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) —— 全双工参考（Lesson 15）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) —— 唤醒词门控。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) —— LLM 函数调用。
