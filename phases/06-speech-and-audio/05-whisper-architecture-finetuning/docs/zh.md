# Whisper —— 架构与微调

> Whisper 是一个 30 秒窗口的 transformer 编码器-解码器，在 68 万小时的多语言弱监督音频-文本对上进行训练。一种架构，多种任务，在 99 种语言上表现鲁棒。2026 年的参考 ASR。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 04（ASR），Phase 5 · 10（Attention），Phase 7 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题

Whisper 由 OpenAI 于 2022 年 9 月发布，是第一个作为商品交付的 ASR 模型：粘贴音频，获得文本，99 种语言，抗噪声，可在笔记本上运行。到 2024 年，OpenAI 已发布 Large-v3 和 Turbo 变体；到 2026 年，Whisper 已成为从播客转录到语音助手再到 YouTube 字幕的默认基线。

但 Whisper 不能永远被当作黑箱处理。领域偏移会杀死它——技术术语、说话人口音、专有名词、短片段、沉默。你需要知道：

1. 它内部到底是什么。
2. 如何正确地给它分块、流式或长格式音频。
3. 何时微调以及如何微调。

## 概念

![Whisper 编码器-解码器、任务、分块推理、微调](../assets/whisper.svg)

**架构。** 标准 transformer 编码器-解码器。

- 输入：30 秒 log-mel 频谱图，80 mel，10 ms hop → 3000 帧。较短的片段补零，较长的片段分块。
- 编码器：卷积下采样（stride 2）+ `N` 个 transformer 块。Large-v3：32 层，1280 维，20 头。
- 解码器：`N` 个 transformer 块，带因果自注意力 + 对编码器输出的 cross-attention。与编码器大小相同。
- 输出：51,865 token 词汇表上的 BPE token。

Large-v3 有 15.5 亿参数。Turbo 使用 4 层解码器（从 32 层削减），延迟降低 8 倍，WER 损失 <1%。

**提示格式。** Whisper 是一个通过解码器提示中的特殊 token 进行引导的多任务模型：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` —— 语言标签；控制翻译 vs 转录行为。
- `<|transcribe|>` 或 `<|translate|>` —— 从任何语言输入翻译为英语输出，或逐字转录。
- `<|notimestamps|>` —— 跳过词级别时间戳（更快）。

提示就是让一个模型做多种任务的原因。将 `<|en|>` 改为 `<|fr|>`，它就转录法语。

**30 秒窗口。** 一切都锚定在 30 秒。更长的片段需要分块；更短的片段需要填充。窗口不能原生流式处理——这就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**Log-mel 归一化。** `(log_mel - mean) / std`，统计量来自 Whisper 自己的训练语料。你*必须*使用 Whisper 的预处理（`whisper.audio.log_mel_spectrogram`），而不是 `librosa.feature.melspectrogram`。

### 2026 年的变体

| 变体 | 参数量 | 延迟 (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× 实时 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 流式 | 2.0% |

### 微调

2026 年的标准流程：

1. 收集 10–100 小时目标领域音频及对齐的转录文本。
2. 使用 `transformers.Seq2SeqTrainer` 并带 `generate_with_loss` 回调。
3. 参数高效：在 attention 层的 `q_proj`、`k_proj`、`v_proj` 上使用 LoRA，GPU 内存减少 4 倍，WER 成本 <0.3。
4. 如果数据 <10 小时，冻结编码器。只微调解码器。
5. 使用 Whisper 自己的 tokenizer 和提示格式；永远不要更换 tokenizer。

社区结果：在 20 小时医疗听写上微调 Medium，医疗词汇上的 WER 从 12% 降至 4.5%。在 4 小时冰岛语上微调 Turbo，WER 从 18% 降至 6%。

## 动手实现

### 步骤 1：开箱即用运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 防止失控重复
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你应该始终覆盖的关键默认值：`temperature=0.0`（采样默认 0.0 → 0.2 → 0.4 … 回退链），`condition_on_previous_text=False`（防止级联幻觉问题），以及 `no_speech_threshold=0.6`（静音检测）。

### 步骤 2：分块长音频

```python
# whisperx 是 2026 年带词级别时间戳的长音频转录参考
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 增加了 (1) Silero VAD 把关，(2) 通过 wav2vec 2.0 的词级别对齐，(3) 通过 `pyannote.audio` 的说话人分离。2026 年生产转录的主力工具。

### 步骤 3：使用 LoRA 微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M 可训练 / 809M 总计
```

然后标准 Trainer 循环。每 1000 步保存 checkpoint。在留出集上用 WER 评估。

### 步骤 4：检查每层学到了什么

```python
# 在解码过程中获取 cross-attention 权重，查看解码器关注什么
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: 层 × 头 × 步 × 源长度
```

用热图可视化——你会看到对角线对齐，因为解码器步骤扫描编码器帧。那条对角线就是 Whisper 对词时间戳的理解。

## 实际应用

2026 年的技术栈：

| 场景 | 选择 |
|-----------|------|
| 通用英语、离线 | Large-v3-turbo 通过 `whisperx` |
| 移动端 / 边缘 | Whisper-Tiny 量化 (int8) 或 Moonshine |
| 多语言长音频 | Large-v3 通过 `whisperx` + 说话人分离 |
| 低资源语言 | 使用 LoRA 微调 Medium 或 Turbo |
| 流式 (2 s 延迟) | Whisper-Streaming 或 Parakeet-TDT |
| 词级别时间戳 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（CTranslate2 后端）是 2026 年最快的 CPU+GPU 推理运行时——比原版快 4 倍，输出完全相同。

## 2026 年仍会出现的陷阱

- **静音上的幻觉文本。** Whisper 在字幕上训练，包含 "Thanks for watching!"、"Subscribe!"、歌词。调用前始终先用 VAD 把关。
- **`condition_on_previous_text` 级联。** 一个幻觉会污染后续窗口。除非需要跨块流畅性，否则设为 `False`。
- **短片段填充。** 2 秒片段填充到 30 秒可能在尾部静音中产生幻觉。使用 `pad=False` 或 VAD 把关。
- **错误的 mel 统计量。** 使用 librosa 的 mel 而不是 Whisper 的会产生接近随机的输出。使用 `whisper.audio.log_mel_spectrogram`。

## 交付产物

保存为 `outputs/skill-whisper-tuner.md`。为给定领域设计 Whisper 微调或推理流程。

## 练习

1. **简单。** 运行 `code/main.py`。它对 Whisper 风格的提示进行 tokenize，计算解码形状预算，并打印 10 分钟片段的分块计划。
2. **中等。** 安装 `faster-whisper`，转录一段 10 分钟播客，与人工转录比较 WER。尝试 `language="auto"` vs 强制 `language="en"`。
3. **困难。** 使用 HF `datasets`，挑选一种 Whisper 表现不佳的语言（例如乌尔都语），用 LoRA 在 2 小时数据上微调 2 个 epoch Medium，报告 WER 变化。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| 30-sec window | Whisper 的限制 | 硬输入上限；更长的音频需要分块。 |
| SOT | 转录开始 | `<|startoftranscript|>` 启动解码器提示。 |
| Timestamps token | 时间对齐 | 每 0.02 秒偏移量是 51k 词汇表中的一个特殊 token。 |
| Turbo | 快速变体 | 4 层解码器，快 8 倍，WER 损失 <1%。 |
| WhisperX | 长音频封装器 | VAD + Whisper + wav2vec 对齐 + 说话人分离。 |
| LoRA fine-tune | 高效微调 | 在 attention 上加低秩适配器；只训练约 0.3% 的参数。 |
| Hallucination | 静默失败 | Whisper 从噪声/静音中产生流利的英语。 |

## 延伸阅读

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) —— 原始架构和训练配方。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) —— 4 层解码器，8 倍加速。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) —— 长音频、词对齐、说话人分离。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) —— CTranslate2 后端，快 4 倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) —— 标准的 LoRA / 全量微调教程。
