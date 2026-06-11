# 语音智能体：Pipecat 和 LiveKit

> 语音智能体（Voice agents）在 2026 年是一流的生产类别。Pipecat 提供基于 Python frame（帧）的流水线：VAD → STT → LLM → TTS → transport（传输层）。LiveKit Agents 通过 WebRTC 将 AI 模型桥接到用户。生产级延迟目标在 premium stacks（优质技术栈）上达到 450–600ms 端到端。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 分钟

## Learning Objectives

- 描述 Pipecat 的 frame-based pipeline（基于帧的流水线）：DOWNSTREAM（下游，source→sink）和 UPSTREAM（上游，控制）。
- 列出 canonical voice pipeline stages（标准语音流水线阶段）以及 Pipecat 支持的 transports（传输协议）。
- 解释 LiveKit Agents 的两个 voice agent classes（语音智能体类）（MultimodalAgent、VoicePipelineAgent）以及各自适用的场景。
- 总结 2026 年生产级 latency（延迟）预期及其如何驱动架构选择。

## The Problem

语音智能体不是文本循环外挂 TTS。Latency budgets（延迟预算）极其紧张（~600ms），partial audio（部分音频）是默认状态，turn detection（回合检测）是一个模型，transports（传输协议）从 telephony SIP 到 WebRTC 不等。你要么构建 frame-based pipeline（基于帧的流水线）（Pipecat），要么依赖平台（LiveKit）。

## The Concept

### Pipecat（pipecat-ai/pipecat）

- Python frame-based pipeline framework（基于帧的流水线框架）。
- `Frame` → `FrameProcessor` 链。
- 两个流向：
  - **DOWNSTREAM（下游）** —— source → sink（音频输入，TTS 输出）。
  - **UPSTREAM（上游）** —— feedback and control（反馈和控制）（cancellation 取消、metrics 指标、barge-in 打断）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和 observers（观察者）管理生命周期，用于 metrics/tracing/RTVI。

典型流水线：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports（传输协议）：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 添加 structured conversations（结构化对话）（state machines 状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型桥接到用户。
- 关键概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两个 voice agent classes：
  - **MultimodalAgent（多模态智能体）** —— 通过 OpenAI Realtime 或等效方案直接传输音频。
  - **VoicePipelineAgent（语音流水线智能体）** —— STT → LLM → TTS 级联；提供 text-level control（文本级控制）。
- 通过 transformer model 进行 semantic turn detection（语义回合检测）。
- 原生 MCP 集成。
- 通过 SIP 支持 Telephony（电话）。
- LiveKit Inference 提供 50+ 无需 API key 的模型；通过 plugins 提供 200+ 更多模型。

### 商业平台

Vapi（在优化的 premium stack 上约 450–600ms）和 Retell（在 180 次测试通话中约 600ms 端到端）构建在这些基础之上。当你想要托管语音栈而又没有 WebRTC 团队时，选择平台。

### 该模式在何处出错

- **无 barge-in handling（打断处理）。** 用户打断；agent 继续说话。Pipecat 需要 UPSTREAM cancel frames（上游取消帧），LiveKit 中等效处理。
- **忽略 STT confidence（语音识别置信度）。** 低置信度转录作为真理输入 LLM。应基于 confidence（置信度）进行 gate（门控）或请求确认。
- **TTS mid-sentence cutoff（TTS 中途截断）。** 当流水线 mid-utterance（话语中途）取消时，TTS 需要知道或切断音频。
- **忽略 latency budget（延迟预算）。** 每个组件增加 50–200ms。发布前累加你的链条。

### 2026 年典型延迟

- VAD: 20–60ms
- STT partial: 100–250ms
- LLM first token: 150–400ms
- TTS first audio: 100–200ms
- Transport RTT: 30–80ms

End-to-end（端到端）450–600ms 是 premium（优质）。800–1200ms 是 common（常见）。> 1500ms 感觉 broken（损坏）。

## Build It

`code/main.py` 是一个 frame-based toy pipeline（基于帧的玩具流水线），包含：

- `Frame` 类型（audio、transcript、text、tts_audio、control）。
- `Processor` 接口 with `process(frame)`。
- 五阶段流水线（VAD → STT → LLM → TTS → transport）作为 scripted processors。
- UPSTREAM cancel frame 演示 barge-in（打断）。

运行方式：

```
python3 code/main.py
```

Trace 显示正常流程和 barge-in cancel（打断取消），该取消在话语中途停止 TTS。

## Use It

- **Pipecat** 用于完全控制 —— custom processors、Python-first、pluggable providers。
- **LiveKit Agents** 用于 WebRTC-first deployments 和 telephony（电话）。
- **Vapi / Retell** 用于无需 WebRTC 团队的 hosted voice agents（托管语音智能体）。
- **OpenAI Realtime / Gemini Live** 用于 direct audio-in/audio-out（直接音频输入/输出）（MultimodalAgent）。

## Ship It

`outputs/skill-voice-pipeline.md` 搭建一个 Pipecat 风格的 voice pipeline（语音流水线），包含 VAD + STT + LLM + TTS + transport 以及 barge-in handling（打断处理）。

## Exercises

1. 为你的 toy pipeline 添加 metrics observer（指标观察者）：每秒统计每个阶段的 frames（帧数）。延迟在哪里累积？
2. 实现 confidence-gated STT（置信度门控语音识别）：低于 threshold（阈值）时，请求 "could you repeat that?"（"能再说一遍吗？"）
3. 添加 semantic turn detection（语义回合检测）：简单规则 —— 如果 transcript 以 "?" 结尾，即为回合结束。
4. 阅读 Pipecat 的 transport docs。将 stdlib transport 替换为 SmallWebRTCTransport config（stub 存根）。
5. 测量同一查询上 OpenAI Realtime 与 STT+LLM+TTS cascade（级联）的延迟。Text-level control（文本级控制）带来多少延迟成本？

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Frame | "Event（事件）" | Pipeline 中的 typed unit of data（数据类型化单元）（audio、transcript、text、control） |
| Processor | "Pipeline stage（流水线阶段）" | 带有 process(frame) 的 handler（处理器） |
| DOWNSTREAM | "Forward flow（正向流）" | Source to sink：音频输入，语音输出 |
| UPSTREAM | "Feedback flow（反馈流）" | Control（控制）：cancel（取消）、metrics（指标）、barge-in（打断） |
| VAD | "Voice activity detection（语音活动检测）" | 检测用户何时在说话 |
| Semantic turn detection | "Smart end-of-turn（智能回合结束）" | 基于模型的 decision（决策），判断用户是否说完 |
| MultimodalAgent | "Direct audio agent（直接音频智能体）" | Audio in, audio out；中间没有文本 |
| VoicePipelineAgent | "Cascade agent（级联智能体）" | STT + LLM + TTS；text-level control（文本级控制） |

## Further Reading

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) —— frame-based pipeline、processors、transports
- [LiveKit Agents docs](https://docs.livekit.io/agents/) —— WebRTC + voice primitives（语音原语）
- [Vapi](https://vapi.ai/) —— 托管语音平台
- [Retell AI](https://www.retellai.com/) —— 托管语音，经过 latency benchmarked（延迟基准测试）
