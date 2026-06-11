"""Moshi 风格全双工模拟。

模拟 Moshi 的并行流架构形状：
  - 用户 Mimi token 流（输入）
  - Moshi Mimi token 流（输出）
  - Moshi 文本流（内心独白）

让一段卡通"对话"通过循环运行；测量每 80 ms 帧的延迟。
没有真正的编解码器或 transformer —— 只是结构。

运行：python3 code/main.py
"""

import math
import random
import time


FRAME_MS = 80
CODEBOOKS = 8
SAMPLE_RATE = 24000


def fake_mimi_encode(audio_80ms):
    s = sum(abs(x) for x in audio_80ms) / max(1, len(audio_80ms))
    rng = random.Random(int(s * 1000))
    return [rng.randint(0, 1023) for _ in range(CODEBOOKS)]


def fake_mimi_decode(tokens):
    s = sum(tokens) / (1024.0 * CODEBOOKS)
    n = int(SAMPLE_RATE * FRAME_MS / 1000)
    return [0.1 * s * math.sin(2.0 * math.pi * 220.0 * i / SAMPLE_RATE) for i in range(n)]


def depth_transformer(context_text, context_user_mimi, context_moshi_mimi):
    time.sleep(0.003)
    rng = random.Random(len(context_user_mimi) + len(context_moshi_mimi))
    return [rng.randint(0, 1023) for _ in range(CODEBOOKS)]


def inner_monologue_next_token(text_so_far, user_mimi_stream):
    time.sleep(0.002)
    return f"tok_{len(text_so_far)}"


def simulate_user_speech(n_frames):
    audio = []
    for i in range(n_frames):
        chunk = [0.15 * math.sin(2 * math.pi * (220 + 20 * i) * j / SAMPLE_RATE) for j in range(int(SAMPLE_RATE * FRAME_MS / 1000))]
        audio.append(chunk)
    return audio


def main():
    print(f"=== Moshi 风格全双工模拟 —— {FRAME_MS} ms 帧, {CODEBOOKS} 个码本 ===")
    print()

    user_audio_stream = simulate_user_speech(25)
    user_mimi = []
    moshi_mimi = []
    moshi_text = []
    per_frame_ms = []

    for t, user_chunk in enumerate(user_audio_stream):
        frame_start = time.time()

        user_tokens = fake_mimi_encode(user_chunk)
        user_mimi.append(user_tokens)

        next_text = inner_monologue_next_token(moshi_text, user_mimi)
        moshi_text.append(next_text)

        next_moshi_tokens = depth_transformer(
            context_text=moshi_text,
            context_user_mimi=user_mimi,
            context_moshi_mimi=moshi_mimi,
        )
        moshi_mimi.append(next_moshi_tokens)

        out_audio = fake_mimi_decode(next_moshi_tokens)
        frame_ms = (time.time() - frame_start) * 1000
        per_frame_ms.append(frame_ms)

    print(f"处理了 {len(user_audio_stream)} 帧 ({len(user_audio_stream)*FRAME_MS} ms 墙钟音频)")
    print(f"  用户 Mimi:    {len(user_mimi)} × {CODEBOOKS} 码本")
    print(f"  Moshi Mimi:   {len(moshi_mimi)} × {CODEBOOKS} 码本")
    print(f"  Moshi 文本:   {len(moshi_text)} 个 token   (前 5 个: {moshi_text[:5]})")

    print()
    print("=== 每帧延迟 ===")
    avg = sum(per_frame_ms) / len(per_frame_ms)
    p95 = sorted(per_frame_ms)[int(len(per_frame_ms) * 0.95)]
    print(f"  均值: {avg:.2f} ms   p95: {p95:.2f} ms   目标: < 80 ms 每帧 (实时)")

    print()
    print("=== 2026 流式 S2S 模型速查表 ===")
    rows = [
        ("Moshi (Kyutai)",       "200 ms L4",   "全双工对话, 英+法",    "CC-BY 4.0"),
        ("Hibiki",                "12.5 Hz",    "英↔法 流式翻译",   "CC-BY 4.0"),
        ("Hibiki-Zero (2月)",  "12.5 Hz",    "5 种语言, 无需对齐数据",       "CC-BY 4.0"),
        ("Sesame CSM-1B",         "200 ms",      "上下文 TTS (非全双工)", "Apache-2.0"),
        ("GPT-4o Realtime",        "~300 ms",     "闭源, API",                   "商业"),
        ("Gemini 2.5 Live",       "~350 ms",     "闭源, API",                   "商业"),
    ]
    print("  | 模型                | 延迟      | 描述                            | 许可证       |")
    for name, lat, desc, lic in rows:
        print(f"  | {name:<20} | {lat:<9} | {desc:<30}  | {lic:<12} |")

    print()
    print("要点:")
    print("  - 全双工架构: 2 个并行 Mimi 流 + 文本内心独白")
    print("  - 160 ms 理论延迟下限 (80 ms 帧 + 80 ms 声学延迟)")
    print("  - Moshi 最适合语音伴侣; 流程化方案 (lesson 12) 在工具调用上仍然获胜")
    print("  - Hibiki 是流式翻译; 相同结构, 不同训练数据")


if __name__ == "__main__":
    main()
