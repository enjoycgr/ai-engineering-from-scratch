# 说话人识别与验证

> ASR 问"他们说了什么？"说话人识别问"是谁说的？"数学看起来一样——embedding 加余弦相似度——但每个生产决策都取决于单一的 EER 数字。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 02（频谱图与 Mel），Phase 5 · 22（Embedding Models）
**时间：** 约 45 分钟

## 问题

用户说一个口令。你想知道：这是否是他们声称的那个人（*验证*，1:1），还是你的注册库中的第一个人（*识别*，1:N）？或者都不是——这是否是未知说话人（*开放集*）？

2018 年前：GMM-UBM + i-vectors。EER 合理但对信道偏移（手机 vs 笔记本）和情绪很脆弱。2018–2022：x-vectors（用 angular margin 训练的 TDNN 主干）。2022+：ECAPA-TDNN 和 WavLM-large embedding。到 2026 年，该领域由三个模型和一个指标主导。

这个指标是 **EER**——等错误率（Equal Error Rate）。设置决策阈值使得 False Accept Rate = False Reject Rate。交叉点就是 EER。每篇论文、每个排行榜、每个采购电话都会用到。

## 概念

![注册 + 验证流程：embedding + 余弦相似度 + EER](../assets/speaker-verification.svg)

**流程。** 注册：录制目标说话人 5–30 秒；计算固定维度 embedding（ECAPA-TDNN 为 192 维，WavLM-large 为 256 维）。验证：获取测试语音的 embedding；计算余弦相似度；与阈值比较。

**ECAPA-TDNN（2020，2026 年仍占主导）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。带 squeeze-excitation 的 1D 卷积块，multi-head attention pooling，然后线性层到 192 维。在 VoxCeleb 1+2（2,700 个说话人，110 万条语句）上用 Additive Angular Margin loss（AAM-softmax）训练。

**WavLM-SV（2022+）。** 用 AAM loss 微调预训练的 WavLM-large SSL 主干。质量更高但更慢——300+ MB vs 15 MB。

**x-vector（基线）。** TDNN + statistics pooling。经典；在 CPU / 边缘设备上仍有价值。

**AAM-softmax。** 标准 softmax 在角度空间增加 margin `m`：正确类别用 `cos(θ + m)`。强制类间角度分离。典型值 `m=0.2`，尺度 `s=30`。

### 打分

- **余弦相似度** 在注册和测试 embedding 之间。基于阈值的决策。
- **PLDA（Probabilistic LDA，概率线性判别分析）。** 将 embedding 投影到一个潜在空间，其中同说话人 vs 不同说话人有闭式似然比。在余弦基础上再降低 10–20% EER。2020 年前标准；现在只用于闭集场景。
- **分数归一化。** `S-norm` 或 `AS-norm`：用一组冒名顶替者的均值和标准差对每个分数归一化。跨域评估必不可少。

### 你应该知道的数字（2026）

| 模型 | VoxCeleb1-O EER | 参数量 | 吞吐量 (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (经典) | 3.10% | 5 M | 400× 实时 |
| ECAPA-TDNN | 0.87% | 15 M | 200× 实时 |
| WavLM-SV large | 0.42% | 316 M | 20× 实时 |
| Pyannote 3.1 分割 + embedding | 0.65% | 6 M | 100× 实时 |
| ReDimNet (2024) | 0.39% | 24 M | 100× 实时 |

### 说话人分离（Diarization）

多说话人片段中的"谁在什么时候说话"。流程：VAD → 分段 → 每段 embedding → 聚类（凝聚或谱聚类）→ 平滑边界。现代栈：`pyannote.audio` 3.1，将说话人分割 + embedding + 聚类封装在一个调用后面。2026 年在 AMI 上的 SOTA DER 约为 ~15%（从 2022 年的 23% 下降）。

## 动手实现

### 步骤 1：MFCC 统计量的 toy embedding

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

离 SOTA 差得远——仅用于教学。`code/main.py` 在合成说话人数据上使用它作为概念验证。

### 步骤 2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤 3：从相似度对计算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer)。两者都报告。

### 步骤 4：使用 SpeechBrain 的生产环境

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# 注册：平均 3-5 条干净样本的 embedding
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# 验证
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA 典型阈值；在你的数据上调参
```

### 步骤 5：使用 pyannote 进行说话人分离

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 实际应用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 闭集 1:1 验证，边缘设备 | ECAPA-TDNN + 余弦阈值 |
| 开放集验证，云端 | WavLM-SV + AS-norm |
| 说话人分离（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺骗（重放 / 深度伪造检测） | AASIST 或 RawNet2 |
| 微型嵌入式（KWS + 注册） | Titanet-Small (NeMo) |

## 陷阱

- **信道不匹配。** 在 VoxCeleb（网络视频）上训练的模型 ≠ 电话音频。始终在目标信道上评估。
- **短语句。** 测试音频低于 3 秒时 EER 急剧恶化。
- **噪声注册。** 一条有噪声的注册样本会污染锚点。使用 ≥3 条干净样本并取平均。
- **跨条件固定阈值。** 始终在来自目标领域的留出开发集上调参。
- **未归一化 embedding 上的余弦相似度。** 先 L2 归一化；否则幅度占主导。

## 交付产物

保存为 `outputs/skill-speaker-verifier.md`。为给定任务选择模型、注册协议、阈值调参计划和欺诈防护措施。

## 练习

1. **简单。** 运行 `code/main.py`。构建合成"说话人"（不同的谐波轮廓），注册，在 100 对测试列表上计算 EER。
2. **中等。** 在 30 条 VoxCeleb1 语句（5 个说话人 × 每人 6 条）上使用 SpeechBrain ECAPA。比较余弦 vs PLDA 的 EER。
3. **困难。** 使用 `pyannote.audio` 构建完整的注册 → 分离 → 验证流程。在 AMI 开发集上评估 DER。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| EER | 标题指标 | False Accept = False Reject 时的阈值。 |
| Verification | 1:1 | "这是 Alice 吗？" |
| Identification | 1:N | "谁在说话？" |
| Open-set | 可能未知 | 测试集可包含未注册的说话人。 |
| Enrollment | 注册 | 计算说话人的参考 embedding。 |
| AAM-softmax | 那个损失 | 带加性角度 margin 的 softmax；强制聚类分离。 |
| PLDA | 经典打分 | 概率 LDA；embedding 上的似然比打分。 |
| DER | 分离指标 | 说话人分离错误率——漏检 + 误报 + 混淆。 |

## 延伸阅读

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) —— 经典深度 embedding 论文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) —— 2020–2026 年的主导架构。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) —— SV 和分离的 SSL 主干。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) —— 生产级分离 + embedding 栈。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) —— 各模型的当前 EER 排名。
