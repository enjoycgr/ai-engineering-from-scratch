"""Whisper 流水线 —— 纯标准库实现：分帧、每帧能量、任务提示。

完整的 log-mel 频谱图需要 FFT（Fast Fourier Transform，快速傅里叶变换）。
为了教学目的，我们展示分帧形状（这是 Transformer 实际看到的全部内容）
以及控制 Whisper 行为的任务 token 前缀。
"""

import math


SAMPLE_RATE = 16000
FRAME_SIZE = 400   # 16 kHz 采样率下 25 ms
HOP = 160          # 16 kHz 采样率下 10 ms
MAX_SECONDS = 30
TARGET_FRAMES = 3000  # 30 s / 10 ms


def sine_wave(freq, duration_s, sr=SAMPLE_RATE):
    n = int(duration_s * sr)
    return [math.sin(2 * math.pi * freq * i / sr) for i in range(n)]


def frame_signal(x, frame_size=FRAME_SIZE, hop=HOP):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames


def frame_energy(frame):
    """平方和能量，取对数缩放。在教学中替代 mel 功率。"""
    e = sum(v * v for v in frame)
    return math.log(e + 1e-9)


def pad_or_clip(frames, target):
    if len(frames) >= target:
        return frames[:target]
    pad_frame = [0.0] * len(frames[0]) if frames else [0.0] * FRAME_SIZE
    return frames + [pad_frame] * (target - len(frames))


def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens


def main():
    print("=== Whisper 预处理流水线 ===")
    print(f"目标: {MAX_SECONDS}s 音频, 采样率 {SAMPLE_RATE} Hz")
    print(f"帧长: {FRAME_SIZE} 采样点 ({FRAME_SIZE / SAMPLE_RATE * 1000:.0f} ms)")
    print(f"步长: {HOP} 采样点 ({HOP / SAMPLE_RATE * 1000:.0f} ms)")
    print()

    # 1 秒 440 Hz 正弦波
    x = sine_wave(440, duration_s=1.0)
    frames = frame_signal(x)
    print(f"1s 信号 → {len(x)} 采样点 → {len(frames)} 帧")

    # 5 秒
    x5 = sine_wave(440, duration_s=5.0)
    frames5 = frame_signal(x5)
    print(f"5s 信号 → {len(x5)} 采样点 → {len(frames5)} 帧")

    # 填充至 Whisper 的 30 秒窗口
    padded = pad_or_clip(frames5, TARGET_FRAMES)
    print(f"填充至 {MAX_SECONDS}s 后: {len(padded)} 帧 (目标 {TARGET_FRAMES})")

    # 每帧"能量"（mel 替代值）。Whisper 每帧使用 80 个 mel 频带。
    energies = [frame_energy(f) for f in frames5]
    print(f"前 5 帧的对数能量: " + ", ".join(f"{e:+.3f}" for e in energies[:5]))
    print()

    print("=== 任务提示 —— 控制 Whisper 行为的开关 ===")
    examples = [
        ("英语转录，带时间戳",
         whisper_prompt(lang="en", task="transcribe", timestamps=True)),
        ("法语翻译成英语，无时间戳",
         whisper_prompt(lang="fr", task="translate", timestamps=False)),
        ("日语转录，带时间戳",
         whisper_prompt(lang="ja", task="transcribe", timestamps=True)),
    ]
    for name, toks in examples:
        print(f"  {name}:")
        print(f"    " + "  ".join(toks))
    print()

    print("=== Whisper 尺寸表 (large-v3 结构) ===")
    configs = [
        ("tiny",      39,  4,  384,  6),
        ("base",      74,  6,  512,  8),
        ("small",    244, 12,  768, 12),
        ("medium",   769, 24, 1024, 16),
        ("large-v3",1550, 32, 1280, 20),
        ("turbo",    809, 32, 1280, 20),
    ]
    print(f"  {'name':<10}  {'params(M)':>10}  {'layers':>7}  {'d_model':>8}  {'heads':>6}")
    for name, p, L, d, h in configs:
        print(f"  {name:<10}  {p:>10}  {L:>7}  {d:>8}  {h:>6}")
    print()
    print("turbo = large-v3 编码器 + 4 层解码器。解码速度提升 8 倍。")


if __name__ == "__main__":
    main()
