"""从零实现音频基础：合成、DFT、检测峰值、演示混叠。

仅标准库：math、wave、struct、os、tempfile。
运行：python3 code/main.py
"""

import math
import os
import struct
import tempfile
import wave


def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2.0 * math.pi * freq_hz * i / sr) for i in range(n)]


def mix(*signals):
    length = min(len(s) for s in signals)
    return [sum(s[i] for s in signals) / len(signals) for i in range(length)]


def write_wav(path, samples, sr):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(struct.pack("<h", max(-32768, min(32767, int(s * 32767)))) for s in samples)
        w.writeframes(frames)


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    ints = struct.unpack("<" + "h" * n, raw)
    return [x / 32768.0 for x in ints], sr


def dft(x):
    n = len(x)
    out = []
    for k in range(n):
        re = 0.0
        im = 0.0
        for j in range(n):
            angle = -2.0 * math.pi * k * j / n
            re += x[j] * math.cos(angle)
            im += x[j] * math.sin(angle)
        out.append((re, im))
    return out


def magnitudes(spectrum):
    return [math.sqrt(re * re + im * im) for re, im in spectrum]


def peak_freq(samples, sr):
    mags = magnitudes(dft(samples))
    half = len(mags) // 2
    mags = mags[:half]
    k = max(range(len(mags)), key=lambda i: mags[i])
    return k * sr / len(samples), k


def downsample_naive(samples, factor):
    return samples[::factor]


def main():
    sr = 8000
    duration = 0.064

    print("=== 步骤 1：合成 440 Hz 正弦波, 8 kHz, 64 ms ===")
    a = sine(440.0, sr, duration)
    print(f"  样本数: {len(a)}")
    print(f"  前 5 个: {[round(x, 4) for x in a[:5]]}")

    print()
    print("=== 步骤 2：WAV 文件往返 ===")
    tmpdir = tempfile.mkdtemp(prefix="audio_fundamentals_")
    path = os.path.join(tmpdir, "a440.wav")
    write_wav(path, a, sr)
    loaded, loaded_sr = read_wav(path)
    size = os.path.getsize(path)
    print(f"  写入 {path} ({size} 字节, sr={loaded_sr})")
    diff = max(abs(a[i] - loaded[i]) for i in range(len(a)))
    print(f"  往返最大绝对误差 (16-bit 量化): {diff:.5f}")

    print()
    print("=== 步骤 3：440 Hz 的 DFT 峰值检测 ===")
    freq, k = peak_freq(a, sr)
    print(f"  峰值 bin k={k}, freq={freq:.1f} Hz (期望 ~440.0 Hz, bin 分辨率 {sr / len(a):.2f} Hz)")

    print()
    print("=== 步骤 4：混合信号 (220 + 440 + 880) ===")
    mixed = mix(sine(220, sr, duration), sine(440, sr, duration), sine(880, sr, duration))
    mags = magnitudes(dft(mixed))[: len(mixed) // 2]
    top3 = sorted(range(len(mags)), key=lambda i: -mags[i])[:3]
    peaks_hz = sorted(round(k * sr / len(mixed), 1) for k in top3)
    print(f"  前 3 个峰值: {peaks_hz} Hz")

    print()
    print("=== 步骤 5：混叠 —— 7 kHz 音调以 10 kHz 采样 ===")
    alias_sr = 10000
    tone = sine(7000.0, alias_sr, 0.0512)
    alias_freq, _ = peak_freq(tone, alias_sr)
    folded = alias_sr - 7000.0
    print(f"  真实频率: 7000.0 Hz (高于奈奎斯特 = {alias_sr / 2} Hz)")
    print(f"  DFT 报告:    {alias_freq:.1f} Hz")
    print(f"  期望混叠: {folded:.1f} Hz  (= sr - f_true)")

    print()
    print("=== 步骤 6：正确下采样 vs 朴素抽取 ===")
    orig_sr = 24000
    sig = sine(7000.0, orig_sr, 0.032)
    decimated = downsample_naive(sig, 3)
    new_sr = orig_sr // 3
    peak_new, _ = peak_freq(decimated, new_sr)
    print(f"  24 kHz 7 kHz 音调, 无低通抽取到 8 kHz:")
    print(f"    抽取后峰值: {peak_new:.1f} Hz (应为折叠导致的 1000 Hz)")
    print(f"    教训: 抽取前始终要低通滤波")


if __name__ == "__main__":
    main()
