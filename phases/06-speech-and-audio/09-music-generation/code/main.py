"""音乐生成卡通：从提示生成符号和弦/鼓点。

这是一个教学替代。真实音乐生成使用神经编解码器 LM
（MusicGen / ACE-Step）或潜在扩散（Stable Audio）。
这里我们在符号层面演示"随时间展开的 token"概念，使结构可见。

仅标准库。运行：python3 code/main.py
"""

import random


MAJOR_KEYS = {
    "C": ["C", "Dm", "Em", "F", "G", "Am", "Bdim"],
    "G": ["G", "Am", "Bm", "C", "D", "Em", "F#dim"],
    "D": ["D", "Em", "F#m", "G", "A", "Bm", "C#dim"],
    "A": ["A", "Bm", "C#m", "D", "E", "F#m", "G#dim"],
}

COMMON_PROGRESSIONS = {
    "pop":     [1, 5, 6, 4],
    "ballad":  [1, 6, 4, 5],
    "jazz":    [2, 5, 1, 6],
    "rock":    [1, 4, 5, 1],
    "lofi":    [6, 4, 1, 5],
}

DRUM_PATTERNS = {
    "pop":    "X.o.X.o.X.o.X.o.",
    "rock":   "X..oX..oX..oX..o",
    "lofi":   "X...o...X...o.o.",
    "jazz":   "X.oox.oxX.oox.ox",
    "trap":   "Xooox.oxXooox.ox",
}


def chord_progression(key, genre, bars=8):
    scale = MAJOR_KEYS[key]
    pat = COMMON_PROGRESSIONS.get(genre, COMMON_PROGRESSIONS["pop"])
    repeats = bars // len(pat) + 1
    seq = (pat * repeats)[:bars]
    return [scale[i - 1] for i in seq]


def drum_pattern(genre, bars=8):
    base = DRUM_PATTERNS.get(genre, DRUM_PATTERNS["pop"])
    return (base * bars)[: bars * 16]


def fake_generate(prompt, rng=None):
    rng = rng or random.Random(0)
    prompt_lower = prompt.lower()
    key = "C"
    for k in MAJOR_KEYS:
        if f" {k.lower()}" in " " + prompt_lower:
            key = k
            break
    genre = "pop"
    for g in COMMON_PROGRESSIONS:
        if g in prompt_lower:
            genre = g
            break
    bars = 8
    bpm = 120
    for token in prompt_lower.split():
        if token.endswith("bpm"):
            try:
                bpm = int(token[:-3])
            except ValueError:
                pass
    return {
        "key": key,
        "genre": genre,
        "bpm": bpm,
        "bars": bars,
        "chords": chord_progression(key, genre, bars),
        "drums": drum_pattern(genre, bars),
    }


def visualize(piece):
    print(f"  调: {piece['key']}  风格: {piece['genre']}  速度: {piece['bpm']} bpm  小节: {piece['bars']}")
    print(f"  和弦: {' | '.join(piece['chords'])}")
    drum = piece["drums"]
    print(f"  鼓点 (kick=X snare=o): {drum}")


def main():
    prompts = [
        "upbeat pop in G major at 128 bpm",
        "slow lofi groove in C",
        "rock anthem in D at 140 bpm",
        "jazz swing in A",
    ]

    print("=== 步骤 1：提示 → 符号音乐片段 (玩具) ===")
    for p in prompts:
        print(f"提示: {p!r}")
        piece = fake_generate(p)
        visualize(piece)
        print()

    print("=== 步骤 2：2026 音乐生成模型速查表 ===")
    models = [
        ("MusicGen-large",     3300, "30 s",  "no",  "MIT"),
        ("Stable Audio Open",  1200, "47 s",  "no",  "non-commercial"),
        ("ACE-Step XL (Apr 26)", 4000, "2 min+", "yes", "Apache-2.0"),
        ("YuE",                7000, "2 min+", "yes", "Apache-2.0"),
        ("Suno v5 (closed)",      0, "4 min",  "yes", "commercial"),
        ("Udio v4 (closed)",      0, "4 min",  "yes + stems", "commercial"),
    ]
    print("  | 模型               | 参数量 (M) | 长度 | 人声 | 许可证        |")
    for name, p, length, v, lic in models:
        print(f"  | {name:<20} | {p:>10} | {length:>6} | {v:<12} | {lic:<14} |")

    print()
    print("要点:")
    print("  - 开源模型: MusicGen (器乐), ACE-Step / YuE (完整歌曲)")
    print("  - 商业: Suno v5 = 质量领先; Udio v4 = 制作工具 (stems + inpaint)")
    print("  - 法律: Warner + UMG 和解 (2025-2026) 定义安全区")
    print("  - 始终用水印 + 元数据披露标记 AI 生成音乐")


if __name__ == "__main__":
    main()
