"""实时语音助手管线模拟器。

模拟音频块流经过 VAD → STT → LLM → TTS，带延迟预算。
没有真实模型；追踪时序以展示预算去向。

运行：python3 code/main.py
"""

import math
import random
import time


CHUNK_MS = 20
VAD_THRESHOLD_DBFS = -40.0


def rms_dbfs(chunk):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    return 20.0 * math.log10(max(rms, 1e-10))


def simulate_chunk(is_speech, rng):
    n = int(0.001 * CHUNK_MS * 16000)
    if is_speech:
        return [0.15 * rng.gauss(0, 1.0) for _ in range(n)]
    return [0.002 * rng.gauss(0, 1.0) for _ in range(n)]


def vad(chunk, threshold_dbfs=VAD_THRESHOLD_DBFS):
    return rms_dbfs(chunk) > threshold_dbfs


def fake_stt(utterance_duration_s):
    latency_ms = 80 + utterance_duration_s * 50
    time.sleep(latency_ms / 1000.0)
    return "hello world"


def fake_llm(text):
    time.sleep(0.15)
    return "sure, one second"


def fake_tts_first_audio(text):
    time.sleep(0.10)
    return "(audio chunk)"


def main():
    random.seed(0)
    rng = random.Random(0)

    print("=== 步骤 1：模拟 1.5 s 用户语音作为 20 ms 块 ===")
    chunks = [simulate_chunk(True, rng) for _ in range(75)]
    chunks += [simulate_chunk(False, rng) for _ in range(20)]
    print(f"  生成了 {len(chunks)} 个块, 每个 {CHUNK_MS} ms = {len(chunks)*CHUNK_MS} ms")

    print()
    print("=== 步骤 2：VAD 门控并缓冲语音 ===")
    buffered = []
    in_speech = False
    for c in chunks:
        active = vad(c)
        if active:
            buffered.extend(c)
            in_speech = True
        elif in_speech and len(buffered) >= 16000 * 0.3:
            break
    print(f"  缓冲了 {len(buffered) / 16000:.3f} s 语音")

    print()
    print("=== 步骤 3：模拟 STT / LLM / TTS 时序 ===")
    budget = {}
    t = time.time()

    t0 = time.time()
    text = fake_stt(len(buffered) / 16000.0)
    budget["STT"] = (time.time() - t0) * 1000

    t0 = time.time()
    reply = fake_llm(text)
    budget["LLM"] = (time.time() - t0) * 1000

    t0 = time.time()
    first_audio = fake_tts_first_audio(reply)
    budget["TTS TTFA"] = (time.time() - t0) * 1000

    total = (time.time() - t) * 1000

    print(f"  用户说: {text!r}")
    print(f"  助手回复: {reply!r}")
    print()
    print("  延迟分解:")
    for stage, ms in budget.items():
        bar = "#" * int(ms / 10)
        print(f"    {stage:<10s}  {ms:>6.1f} ms  {bar}")
    print(f"  端到端: {total:.1f} ms   (目标: < 500 ms)")

    print()
    print("=== 步骤 4：2026 生产预算去向 ===")
    rows = [
        ("network in",  "50-100"),
        ("VAD",          "20-80"),
        ("STT stream",   "100-300"),
        ("LLM stream",   "100-500"),
        ("TTS TTFA",     "100-300"),
        ("network out",  "50-100"),
        ("TOTAL",        "400-1400"),
    ]
    print("  | 阶段           | 典型 ms |")
    for name, ms in rows:
        print(f"  | {name:<15} | {ms:>10} |")

    print()
    print("  < 500 ms: LiveKit + Silero + Deepgram + GPT-4o + Cartesia")
    print("  < 200 ms: Moshi (全双工) 或 Sesame CSM — 不同架构 (见第 15 课)")


if __name__ == "__main__":
    main()
