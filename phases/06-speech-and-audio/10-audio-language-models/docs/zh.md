# 音频-语言模型 —— Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的音频-语言模型对语音 + 环境声音 + 音乐进行推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上匹敌 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上击败 Gemini 2.5 Pro。开源与闭源的差距基本消除——除了多音频任务，那里所有人都接近随机水平。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 04（ASR），Phase 12 · 03（视觉-语言模型），Phase 7 · 10（音频 Transformer）
**时间：** 约 45 分钟

## 问题

你有 5 秒音频：狗叫，有人喊"stop!"，然后沉默。有用的问题跨越多个维度：

- **转录。** "说了什么？"——ASR 领域。
- **语义推理。** "这个人有危险吗？"——需要联合理解 bark + yell + silence。
- **音乐推理。** "什么乐器演奏旋律？"
- **长音频检索。** "在这 90 分钟的讲座中，讲师在哪里解释了 gradient descent（梯度下降）？"

一个能用单一提示回答所有这些问题的模型就是**音频-语言模型**（LALM / ALM）。与纯 ASR 不同：LALM 生成自由形式的自然语言答案，不仅仅是转录文本。

## 概念

![音频-语言模型：音频编码器 + 投影器 + LLM 解码器](../assets/alm-architecture.svg)

### 三组件模板

每个 2026 年的 LALM 都有相同的骨架：

1. **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM · 或每个模型自定义的编码器。
2. **投影器（Projector）。** 线性或 MLP，将音频编码器特征桥接到 LLM 的 token embedding 空间。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。接收交错的文本 + 音频 token；生成文本。

训练：

- **阶段 1。** 冻结编码器 + LLM；仅在 ASR / 字幕数据上训练投影器。
- **阶段 2。** 在指令遵循音频任务（QA、推理、音乐理解）上做全量 / LoRA 微调。
- **阶段 3（可选）。** 语音输入/语音输出增加一个语音解码器。Qwen2.5-Omni 和 AF3-Chat 这样做。

### 2026 年模型图谱

| 模型 | 主干 | 音频编码器 | 输出模态 | 访问 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（闭源） | Gemini | 专有 | 文本 + 语音 | API |
| GPT-4o Audio（闭源） | GPT-4o | 专有 | 文本 + 语音 | API |

### 基准现实检查（2026）

**MMAU-Pro。** 1800 个 QA 对，覆盖语音 / 声音 / 音乐 / 混合。包含多音频子集。

| 模型 | 总体 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | LongAudioBench SOTA | — | — | — | — |

**多音频列对所有人都是致命的。** 4 选项多选题的随机概率 = 25%；大多数模型得分都在那里。LALM 仍然难以比较两个片段。

### 2026 年 LALM 有用的场景

- **呼叫中心录音的合规审计。** "座席是否提到了必要的披露？"
- **无障碍。** 向聋人用户描述声音事件（不仅仅是转录）。
- **内容审核。** 检测暴力语言 + 威胁语气 + 背景上下文。
- **播客/会议分章。** 语义摘要，不仅仅是说话人轮次。
- **音乐曲库分析。** "找到所有 B 段有转调的曲目。"

### 尚未有用的场景

- 细粒度音乐理论（和弦级别以下）。
- 长对话上的说话人归因推理（超过 10 分钟会退化）。
- 多音频比较（22-26%  barely above random）。
- 实时流式推理（大多数是离线批处理推理）。

## 动手实现

### 步骤 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤 2：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。投影器通常是 1-3 个线性层。在 ASR 对（音频 → 转录文本）上训练它是阶段 1 的预训练任务。

### 步骤 3：MMAU / LongAudioBench 基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

分别报告每个类别（语音 / 声音 / 音乐 / 多音频）。汇总数字会隐藏模型失败的地方。

## 实际应用

| 任务 | 2026 年选择 |
|------|-----------|
| 自由形式音频 QA（开源） | Qwen2.5-Omni-7B |
| 长音频最佳开源 | Audio Flamingo Next |
| 最佳闭源 | Gemini 2.5 Pro |
| 语音输入/语音输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（音乐专用 AF-CLAP） |
| 呼叫中心审计 | Gemini 2.5 Pro 通过 API，配合策略文档的 RAG |

## 陷阱

- **过度信任多音频能力。** 如果你的任务需要"哪个片段有 X"，随机概率水平的性能是真实的。
- **长音频退化。** 超过 10 分钟，大多数模型的说话人归因会崩溃。先分离（Lesson 6），然后摘要。
- **静音上的幻觉。** 使用 Whisper 编码器的 LALM 继承了同样的 Whisper 风格问题。用 VAD 把关。
- **基准 cherry-picking。** 厂商博客文章突出最佳情况类别。自己运行 MMAU-Pro 多音频子集。

## 交付产物

保存为 `outputs/skill-alm-picker.md`。为给定音频理解任务选择 LALM + 基准子集 + 输出模态（文本 vs 语音）。

## 练习

1. **简单。** 运行 `code/main.py` 查看 toy 投影器模式 + 假 LALM 将（音频 embedding、文本 token）路由到输出 token。
2. **中等。** 在 100 个 MMAU-Pro 语音项目上给 Qwen2.5-Omni-7B 打分。与论文报告的数字比较。
3. **困难。** 构建一个最小音频字幕基线：BEATs 编码器 + 2 层投影器 + 冻结的 Llama-3.2-1B。仅在 AudioCaps 上微调投影器。与 SALMONN 在 Clotho-AQA 上比较。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| LALM | 音频 ChatGPT | 音频编码器 + 投影器 + LLM 解码器。 |
| Projector | 适配器 | 将音频特征映射到 LLM embedding 空间的小型 MLP。 |
| MMAU | 那个基准 | 跨语音、声音、音乐的 10k 音频-QA 对。 |
| MMAU-Pro | 更难的 MMAU | 1800 个多音频/重推理问题。 |
| LongAudioBench | 长音频评估 | 带有语义查询的多分钟片段。 |
| Voice-in / voice-out | 语音原生 | 模型摄入语音并发射语音，无需文本中转。 |

## 延伸阅读

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) —— 参考架构。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) —— 语音输入/语音输出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) —— 开源长音频领先者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) —— LongAudioBench SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) —— 双编码器先驱。
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) —— 2026 年实时排名。
