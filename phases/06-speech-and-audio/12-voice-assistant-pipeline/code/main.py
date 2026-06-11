"""端到端语音助手模拟器——7 个组件，桩实现。

模拟完整用户轮次：麦克风 → VAD → STT → LLM（带 tool-call）→ TTS。
打印每阶段延迟 + 决策追踪。

没有真实模型——将每个桩替换为 Silero VAD / Whisper / GPT-4o /
Kokoro 即可得到生产管线。

运行：python3 code/main.py
"""

import math
import random
import time


def mic_generator(duration_s=2.0, sr=16000, chunk_ms=20, speech_mask=None):
    rng = random.Random(0)
    n_chunks = int(duration_s * 1000 / chunk_ms)
    if speech_mask is None:
        speech_mask = [False] * 5 + [True] * 60 + [False] * 20
    for i in range(min(n_chunks, len(speech_mask))):
        is_speech = speech_mask[i]
        n = int(sr * chunk_ms / 1000)
        if is_speech:
            chunk = [0.2 * rng.gauss(0, 1.0) for _ in range(n)]
        else:
            chunk = [0.003 * rng.gauss(0, 1.0) for _ in range(n)]
        yield chunk, is_speech


def vad(chunk, threshold_dbfs=-35.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    return 20.0 * math.log10(max(rms, 1e-10)) > threshold_dbfs


def streaming_stt(utterance, sr=16000):
    time.sleep(0.08 + len(utterance) / sr * 0.05)
    return "set a timer for five minutes"


def llm_with_tools(transcript):
    time.sleep(0.12)
    if "timer" in transcript:
        return {
            "tool_calls": [{"name": "set_timer", "args": {"seconds": 300}}],
            "text": "Sure, setting a 5 minute timer.",
        }
    return {"tool_calls": [], "text": "OK."}


def dispatch_tool(name, args):
    time.sleep(0.01)
    if name == "set_timer":
        return {"ok": True, "expires_at": time.time() + args["seconds"]}
    return {"ok": False}


def streaming_tts(text):
    time.sleep(0.10)
    return [f"(audio chunk: {word})" for word in text.split()]


def play(audio_chunks):
    for _ in audio_chunks:
        time.sleep(0.02)


def main():
    random.seed(0)

    print("=== 步骤 1：通过 VAD 门控捕获轮次 ===")
    buffered = []
    pre_roll = []
    triggered = False
    silent_ms = 0
    turn_start = time.time()
    for chunk, truth in mic_generator():
        pre_roll.append(chunk)
        if len(pre_roll) > 15:
            pre_roll.pop(0)
        if vad(chunk):
            if not triggered:
                for c in pre_roll:
                    buffered.extend(c)
                triggered = True
            buffered.extend(chunk)
            silent_ms = 0
        elif triggered:
            silent_ms += 20
            buffered.extend(chunk)
            if silent_ms >= 400:
                break
    t_capture = (time.time() - turn_start) * 1000
    print(f"  捕获了 {len(buffered)} 个样本 ({len(buffered)/16000:.3f} s) 用时 {t_capture:.0f} ms 墙上时间")

    print()
    print("=== 步骤 2：流式 STT ===")
    t0 = time.time()
    text = streaming_stt(buffered)
    t_stt = (time.time() - t0) * 1000
    print(f"  转录: {text!r}   STT 延迟: {t_stt:.1f} ms")

    print()
    print("=== 步骤 3：带 tool calling 的 LLM ===")
    t0 = time.time()
    response = llm_with_tools(text)
    t_llm = (time.time() - t0) * 1000
    print(f"  tool_calls: {response['tool_calls']}")
    for call in response["tool_calls"]:
        result = dispatch_tool(call["name"], call["args"])
        print(f"  {call['name']}({call['args']}) → {result}")
    print(f"  回复文本: {response['text']!r}   LLM 延迟: {t_llm:.1f} ms")

    print()
    print("=== 步骤 4：流式 TTS + 播放 ===")
    t0 = time.time()
    audio = streaming_tts(response["text"])
    t_tts_ttfa = (time.time() - t0) * 1000
    print(f"  TTFA: {t_tts_ttfa:.1f} ms    音频块: {len(audio)}")
    t0 = time.time()
    play(audio)
    t_play = (time.time() - t0) * 1000

    print()
    print("=== 步骤 5：端到端预算 ===")
    stages = [
        ("VAD + 捕获 (语音结束后)", silent_ms),
        ("STT",  t_stt),
        ("LLM + tool",  t_llm),
        ("TTS TTFA",    t_tts_ttfa),
    ]
    total = sum(ms for _, ms in stages)
    for name, ms in stages:
        bar = "#" * int(ms / 10)
        print(f"  {name:<40s} {ms:>6.1f} ms  {bar}")
    print(f"  用户感知总计 (到第一帧音频): {total:.1f} ms   (目标: < 800 ms)")

    print()
    print("=== 步骤 6：2026 参考技术栈 ===")
    stacks = [
        ("LiveKit + Deepgram + GPT-4o + Cartesia",       "350-500 ms", "行业默认"),
        ("Pipecat + Whisper-stream + GPT-4o + Kokoro",   "500-800 ms", "DIY 友好"),
        ("Moshi (全双工单模型)",              "200-300 ms", "见第 15 课"),
        ("Vapi / Retell (托管)",                        "300-500 ms", "最快上线"),
        ("whisper.cpp + llama.cpp + Kokoro-ONNX",         "离线",    "边缘 / 隐私"),
    ]
    for s, lat, note in stacks:
        print(f"  {s:<46s} {lat:<12s} {note}")


if __name__ == "__main__":
    main()
