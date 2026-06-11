"""从零实现的音频评估指标。

实现 WER、CER、EER、简单 SECS、类 FAD embedding 距离，
和类 MMAU 多选准确率。仅标准库。

运行：python3 code/main.py
"""

import math
import random


def _edit_distance(a_tokens, b_tokens):
    dp = [[0] * (len(b_tokens) + 1) for _ in range(len(a_tokens) + 1)]
    for i in range(len(a_tokens) + 1):
        dp[i][0] = i
    for j in range(len(b_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(a_tokens) + 1):
        for j in range(1, len(b_tokens) + 1):
            cost = 0 if a_tokens[i - 1] == b_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[len(a_tokens)][len(b_tokens)]


def normalize(text):
    import re
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wer(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    return _edit_distance(r, h) / max(1, len(r))


def cer(ref, hyp):
    return _edit_distance(list(ref), list(hyp)) / max(1, len(ref))


def eer_from_scores(same, diff):
    thresholds = sorted(set(same + diff))
    best = (1.0, 0.0, 0.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff if s >= t) / max(1, len(diff))
        frr = sum(1 for s in same if s < t) / max(1, len(same))
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), t, far, frr)
    gap, t, far, frr = best
    return (far + frr) / 2, t


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)


def embedding_fad_like(real_embeds, fake_embeds):
    def mean_var(embs):
        n = len(embs[0])
        mean = [sum(e[i] for e in embs) / len(embs) for i in range(n)]
        var = [sum((e[i] - mean[i]) ** 2 for e in embs) / len(embs) for i in range(n)]
        return mean, var
    mu_r, v_r = mean_var(real_embeds)
    mu_f, v_f = mean_var(fake_embeds)
    mean_dist = sum((a - b) ** 2 for a, b in zip(mu_r, mu_f))
    var_dist = sum((math.sqrt(a) - math.sqrt(b)) ** 2 for a, b in zip(v_r, v_f))
    return math.sqrt(mean_dist + var_dist)


def mmau_accuracy(predictions, golds):
    correct = sum(1 for p, g in zip(predictions, golds) if p == g)
    return correct / max(1, len(predictions))


def main():
    print("=== WER + CER ===")
    pairs = [
        ("turn on the kitchen lights",  "turn off the kitchen lights"),
        ("what's the weather today",     "what is the weather today"),
        ("play jazz",                    "play jazz"),
        ("set a 5 minute timer",         "set a five minute timer"),
    ]
    for ref, hyp in pairs:
        print(f"  参考: {ref!r}")
        print(f"  假设: {hyp!r}")
        print(f"    WER = {wer(ref, hyp):.3f}   CER = {cer(ref, hyp):.3f}")

    print()
    print("=== EER (玩具说话人验证) ===")
    random.seed(0)
    rng = random.Random(0)
    same = [rng.gauss(0.80, 0.06) for _ in range(100)]
    diff = [rng.gauss(0.20, 0.15) for _ in range(500)]
    eer, t = eer_from_scores(same, diff)
    print(f"  同一人均值 cosine: {sum(same)/len(same):.3f}")
    print(f"  不同人均值 cosine: {sum(diff)/len(diff):.3f}")
    print(f"  EER = {eer * 100:.2f}%   在阈值 {t:.3f}")

    print()
    print("=== SECS (玩具语音克隆相似度) ===")
    ref_emb = [rng.gauss(0, 0.1) for _ in range(192)]
    clone_emb = [ref_emb[i] + rng.gauss(0, 0.1) for i in range(192)]
    secs = cosine(ref_emb, clone_emb)
    print(f"  SECS = {secs:.3f}   (目标: > 0.75 以得到可识别的克隆)")

    print()
    print("=== 类 FAD embedding 距离 ===")
    real_embs = [[rng.gauss(0, 1.0) for _ in range(32)] for _ in range(50)]
    fake_embs = [[rng.gauss(0.1, 1.1) for _ in range(32)] for _ in range(50)]
    fad = embedding_fad_like(real_embs, fake_embs)
    print(f"  类 FAD = {fad:.3f}   (MusicGen-small 在 MusicCaps 上: 4.5)")

    print()
    print("=== 类 MMAU-Pro 多选准确率 ===")
    predictions = ["A", "C", "B", "A", "D", "C", "B", "A", "A", "C"]
    golds       = ["A", "B", "B", "A", "D", "A", "B", "A", "C", "C"]
    acc = mmau_accuracy(predictions, golds)
    print(f"  准确率 = {acc:.3f}  (4 选 1 随机: 0.250)")

    print()
    print("=== 值得了解的 2026 基准 ===")
    rows = [
        ("Open ASR Leaderboard",  "LibriSpeech + 多语言", "Parakeet-TDT 6.05%, Whisper-LV3-turbo 1.58%"),
        ("TTS Arena",             "盲测成对 TTS",          "Kokoro ELO 1059, ElevenLabs v3 1179"),
        ("Artificial Analysis Speech", "TTS + STT 竞技场",        "Inworld TTS-1.5-Max ELO 1236 领先"),
        ("MMAU-Pro",              "LALM 推理",              "Gemini 2.5 Pro ~60%, GPT-4o Audio 52.5%"),
        ("LongAudioBench",        "多分钟 LALM",           "Audio Flamingo Next 击败 Gemini 2.5 Pro"),
        ("VoxCeleb1-O",           "说话人验证 EER",    "ECAPA 0.87%, 3D-Speaker 0.50%"),
        ("AudioSet mAP",          "多标签分类",  "BEATs-iter3 0.548 mAP"),
        ("ASVspoof 5",            "反欺骗 EER",           "SOTA ~7.23% 野外"),
    ]
    print("  | 排行榜                    | 维度                      | 2026 SOTA                                   |")
    for name, axis, sota in rows:
        print(f"  | {name:<24} | {axis:<25} | {sota:<43} |")

    print()
    print("要点:")
    print("  - 每个任务有 2-3 个主要指标; 训练前选择")
    print("  - 计算 WER/CER 前归一化文本; 报告归一化规则")
    print("  - 延迟报告 P50/P95/P99, 分类报告每类, MMAU 报告每类别")
    print("  - 公共基准 + 你自己的留出领域集 = 两者都要, 永远")


if __name__ == "__main__":
    main()
