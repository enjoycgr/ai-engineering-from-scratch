# 流式语音到语音 —— Moshi、Hibiki 与全双工对话

> 2024-2026 重新定义了语音 AI。Moshi 交付了一个单一模型，以 200 ms 延迟同时听和说。Hibiki 逐块进行语音到语音翻译。两者都放弃了 ASR → LLM → TTS 流程，采用基于 Mimi codec token 的统一全双工架构。这是新的参考设计。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 13（神经音频编解码器），Phase 6 · 11（实时音频处理），Phase 7 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题

由 Lessons 11 + 12 构建的每个语音助手都有一个约 300-500 ms 的基本延迟下限：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有自己的最小延迟。你可以调优和并行化，但流程形状决定了上限。

Moshi（Kyutai，2024-2026）提出了一个不同的问题：如果没有流程会怎样？如果一个模型直接接收音频并连续输出音频，将文本作为中间"内心独白"而非必需阶段，会怎样？

答案是**全双工语音到语音**。理论延迟 160 ms（80 ms Mimi 帧 + 80 ms 声学延迟）。实际延迟在单 L4 GPU 上为 200 ms。这是最佳级联流程化语音助手的一半。

## 概念

![Moshi 架构：两个并行 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi codec 流，均为 12.5 Hz × 8 码本：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自己的音频（由 Moshi 生成）

**Transformer。** 一个 70 亿参数的 Temporal Transformer 处理两个流和一个文本"内心独白"流。在每个 80 ms 步骤，它：

1. 消费最新的用户 Mimi token（8 个码本）。
2. 消费最近的 Moshi Mimi token（8 个码本，由 Moshi 自己产生）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一个 Moshi Mimi token（通过小型 Depth Transformer 生成 8 个码本）。

所有三个流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话时听到用户；可以在用户打断时自我打断；可以在不打断主话语的情况下进行副语言反馈（"嗯嗯"）。

**Depth Transformer。** 在每一帧内，8 个码本不是并行预测的——它们有码本间依赖关系。一个小的 2 层"depth transformer"在 80 ms 内顺序预测它们。这是 AR codec LM 的标准分解方式（VALL-E、VibeVoice 也使用）。

### 为什么内心独白文本有帮助

没有显式文本时，模型必须在声学流中隐式建模语言。Moshi 的洞察：强制它 alongside 音频发出文本 token。文本流本质上是 Moshi 所说内容的转录。这提高了语义连贯性，使得更换语言模型头更容易，并免费为你提供转录。

### Hibiki：流式语音到语音翻译

相同架构，在翻译对上训练。源语言音频输入，目标语言音频输出，持续进行。Hibiki-Zero（2026 年 2 月）消除了词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习进行延迟优化。

最初支持四对语言；可用约 1000 小时数据适配到新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — 全双工对话（法语优先，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同声传译
- **Kyutai STT** — 流式 ASR（500 ms 或 2.5 s 前瞻）
- **Kyutai Pocket TTS** — 1 亿参数 TTS 在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在公共服务器上组合这些的全流程

在 L40S GPU 上的吞吐量：64 个并发会话，3 倍实时速度。

### Sesame CSM — 近亲

Sesame CSM（2025）使用类似的想法——Llama-3 主干 + Mimi codec 头。但 CSM 是单向的（接收上下文 + 文本，产生语音）而非全双工。它是市场上最好的"语音存在感"TTS；与 Moshi 的全双工能力不完全相同。

### 2026 年性能数字

| 模型 | 延迟 | 用例 | 许可证 |
|------|------|------|--------|
| Moshi | 200 ms (L4) | 全双工英语 / 法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语同声传译 | CC-BY 4.0 |
| Hibiki-Zero | 相同 | 5 对语言，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 闭源，Google API | 商业 |

## 动手实现

### 步骤 1：接口

Moshi 暴露一个 WebSocket 服务器，接收 80 ms 的 Mimi 编码音频块，并返回 80 ms 的 Mimi 编码音频块。双向。持续进行。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤 2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### 步骤 3：训练目标（概念性）

对于每个 80 ms 帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在 depth transformer 内按码本顺序预测。

### 步骤 4：Moshi 的优势与劣势

Moshi 的优势：

- 廉价硬件上亚 250 ms 端到端延迟。
- 自然的副语言反馈和打断。
- 无需流程粘合代码。

Moshi 的劣势：

- 工具调用（未针对此训练；需要单独的 LLM 路径）。
- 长推理（Moshi 是一个约 80 亿参数的对话模型，不是 Claude/GPT-4）。
- 小众话题的事实准确性。
- 大多数生产型企业用例（2026 年仍使用流程）。

## 实际应用

| 场景 | 选择 |
|------|------|
| 最低延迟语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi、CSM |
| 带工具的企业智能体 | 流程（Lesson 12），非 Moshi |
| 上下文中的定制声音 TTS | Sesame CSM |
| 任意语言的语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业）|

## 陷阱

- **有限的工具调用。** Moshi 是对话模型，不是智能体框架。结合流程使用工具。
- **特定声音条件。** Moshi 使用单一训练好的角色；克隆是单独的训练运行。
- **语言覆盖。** 法语 + 英语优秀；其他语言有限。Hibiki-Zero 有帮助，但仍需要训练数据。
- **资源成本。** 完整 Moshi 会话占用一个 GPU 槽位；不是廉价的共享租户部署模式。

## 交付产物

保存为 `outputs/skill-duplex-pipeline.md`。为语音智能体工作负载选择流程 vs 全双工架构，并说明原因。

## 练习

1. **简单。** 运行 `code/main.py`。它以符号方式模拟双流 + 内心独白架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户语音结束到 Moshi 响应开始的实际延迟。
3. **困难。** 拿你的 Lesson 12 流程化智能体，与 Moshi 在 20 个匹配测试话语上比较 P50 延迟。写下流程在架构上仍然获胜的场景。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Full-duplex | 同时听和说 | 同一模型上两个音频流同时活跃。 |
| Inner monologue | 模型的文本流 | Moshi alongside 音频输出发出文本 token。 |
| Depth transformer | 码本间预测器 | 在 80 ms 帧内预测 8 个码本的小型 transformer。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 码本；语义+声学；驱动 Moshi。 |
| Streaming S2S | 实时音频 → 音频 | 逐块翻译/对话，无流程阶段。 |
| Back-channeling | "嗯嗯"反应 | Moshi 可以在不打断自己轮替的情况下发出小型确认。 |

## 延伸阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) —— 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) —— 无需对齐数据的流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) —— CSM 规格。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) —— 安装 + 服务器。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) —— 闭源商业竞品。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) —— 底层 STT/TTS 框架。
