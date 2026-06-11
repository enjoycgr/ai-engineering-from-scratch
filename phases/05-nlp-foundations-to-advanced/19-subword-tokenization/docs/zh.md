# 子词分词 — BPE、WordPiece、Unigram、SentencePiece

> 词级别分词器遇到未登录词就会卡住。字符级别分词器会让序列长度爆炸。子词分词器取折中。每个现代 LLM 都基于它构建。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 分钟

## The Problem

你的词表有 50,000 个词。用户输入了 "untokenizable"。分词器返回 `[UNK]`。模型现在对这个词没有任何信号。更糟的是：语料中第 90 百分位的文档包含 40 个罕见词，这意味着每篇文档丢失 40 比特信息。

Subword tokenization (子词分词) 解决了这个问题。常见词保持为单个 token。罕见词分解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。训练数据覆盖一切，因为任何字符串最终都是字节序列。

2026 年的每个前沿 LLM 都基于三种算法之一（BPE、Unigram、WordPiece），封装在三个库之一（tiktoken、SentencePiece、HF Tokenizers）中。不选一个就无法发布语言模型。

## The Concept

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding, 字节对编码).** 从字符级别词表开始。统计每个相邻字符对。将最频繁的字符对合并为一个新 token。重复直到达到目标词表大小。主导算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**Byte-level BPE (字节级 BPE).** 同样的算法，但基于原始字节（256 个基础 token）而非 Unicode 字符。保证零 `[UNK]` token —— 任何字节序列都能编码。GPT-2 使用 50,257 个 token（256 字节 + 50,000 次合并 + 1 个特殊 token）。

**Unigram (一元语言模型).** 从一个巨大的词表开始。为每个 token 分配一个 unigram 概率。迭代地剪除那些移除后最小幅增加语料 log-likelihood (对数似然) 的 token。推理时具有概率性：可以采样不同的分词结果（对通过 subword regularization (子词正则化) 进行数据增强很有用）。T5、mBART、ALBERT、XLNet、Gemma 使用它。

**WordPiece.** 合并能最大化训练语料 likelihood (似然) 的字符对，而非原始频率。BERT、DistilBERT、ELECTRA 使用它。

**SentencePiece vs tiktoken.** SentencePiece 是直接在原始 Unicode 文本上*训练*词表（BPE 或 Unigram）的库，将空白编码为 `▁`。tiktoken 是 OpenAI 的针对预构建词表的快速*编码器*；它不能训练。

经验法则：

- **训练新词表：** SentencePiece（多语言，无需预分词）或 HF Tokenizers。
- **针对 GPT 词表的快速推理：** tiktoken（cl100k_base、o200k_base）。
- **两者兼顾：** HF Tokenizers —— 一个库，训练 + 服务。

## Build It

### Step 1: 从零实现 BPE

参见 `code/main.py`。循环如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

该算法编码了三个事实。`</w>` 标记词尾，因此 "low"（后缀）和 "lower"（前缀）保持不同。频率加权使高频字符对优先合并。合并列表是有序的 —— 推理时按训练顺序应用合并。

### Step 2: 使用学到的合并进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素的 O(n·|merges|)。生产实现（tiktoken、HF Tokenizers）使用 merge-rank (合并排名) 查找配合优先队列，运行时间接近线性。

### Step 3: 实践中的 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # 或 "unigram"
    character_coverage=0.9995, # 对 CJK 可更低（英文 0.9995，日文 0.995）
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需预分词，空格编码为 `▁`，`character_coverage` 控制罕见字符是被保留还是映射到 `<unk>` 的激进程度。

### Step 4: 用于 OpenAI 兼容词表的 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码。快速（Rust 后端）。与 GPT-4/5 分词完全匹配，用于字节计数、成本估算、上下文窗口预算。

## Pitfalls that still ship in 2026

- **Tokenizer drift (分词器漂移).** 在词表 A 上训练，在词表 B 上部署。Token ID 不同；模型输出乱码。在 CI 中检查 `tokenizer.json` 哈希。
- **Whitespace ambiguity (空白歧义).** BPE 对 "hello" 和 " hello" 产生不同的 token。始终显式指定 `add_special_tokens` 和 `add_prefix_space`。
- **Multilingual undertraining (多语言欠训练).** 英语为主的语料产生的词表将非拉丁文字拆分为 5-10 倍更多的 token。同样的 prompt 在 GPT-3.5 上用日文/阿拉伯文成本贵 5-10 倍。o200k_base 部分修复了这个问题。
- **Emoji splits (表情符号拆分).** 单个表情符号可能占 5 个 token。在预算上下文时检查表情符号处理。

## Use It

2026 年技术栈：

| 场景 | 选择 |
|-----------|------|
| 从零训练单语模型 | HF Tokenizers (BPE) |
| 训练多语言模型 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| 提供 OpenAI 兼容 API | tiktoken (`o200k_base` 用于 GPT-4+) |
| 领域专用词表（代码、数学、蛋白质） | 在领域语料上训练自定义 BPE，与基础词表合并 |
| 边缘推理，小模型 | Unigram（更小的词表效果更好） |

词表大小是扩展决策，不是常数。粗略启发式：<1B 参数用 32k，1-10B 用 50-100k，多语言/前沿模型用 200k+。

## Ship It

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: 为给定语料和部署目标选择分词器算法、词表大小和库。
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

给定语料（大小、语言、领域）和部署目标（从零训练 / 微调 / API 兼容推理），输出：

1. Algorithm (算法). BPE、Unigram 或 WordPiece。一句话理由。
2. Library (库). SentencePiece、HF Tokenizers 或 tiktoken。理由。
3. Vocab size (词表大小). 四舍五入到最近的 1k。理由与模型大小和语言覆盖相关。
4. Coverage settings (覆盖设置). `character_coverage`、`byte_fallback`、特殊 token 列表。
5. Validation plan (验证计划). 在留出集上的平均 tokens-per-word (每词 token 数)、OOV rate (未登录词率)、compression ratio (压缩比)、round-trip decode equality (往返解码一致性)。

拒绝在包含罕见文字内容的语料上训练 character-coverage <0.995 的分词器。拒绝发布没有在 CI 中冻结 `tokenizer.json` 哈希检查的词汇表。将任何低于 16k 词表的单语分词器标记为可能规格不足。
```

## Exercises

1. **Easy.** 在 `code/main.py` 的微型语料上训练一个 500-merge 的 BPE。编码三个留出词。有多少词恰好产生 1 个 token，有多少产生 >1 个？
2. **Medium.** 比较 `cl100k_base`、`o200k_base` 和你训练的 vocab=32k 的 SentencePiece BPE 在 100 句英文 Wikipedia 上的 token 数量。报告各自的 compression ratio (压缩比)。
3. **Hard.** 用 BPE、Unigram 和 WordPiece 训练同一语料。测量在小型情感分类器上使用每种方法时的下游准确率。选择是否能让 F1 变化超过 1 个点？

## Key Terms

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 贪心合并最频繁的字符对，直到达到目标词表大小。 |
| Byte-level BPE | 永远没有未知 token | 基于 256 字节的 BPE；GPT-2 / Llama 使用它。 |
| Unigram | 概率分词器 | 使用 log-likelihood 从大候选集中剪枝；T5、Gemma 使用。 |
| SentencePiece | 处理空白的那个 | 在原始文本上训练 BPE/Unigram 的库；空格编码为 `▁`。 |
| tiktoken | 快速的那个 | OpenAI 的 Rust 后端 BPE 编码器，用于预构建词表。不能训练。 |
| Merge list | 魔法数字 | `(a, b) → ab` 合并的有序列表；推理时按顺序应用。 |
| Character coverage | 多罕见算太罕见？ | 分词器必须覆盖的训练语料字符比例；~0.9995 是典型值。 |

## Further Reading

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 论文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 论文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 该库。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 简明参考。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — 食谱 + 编码列表。
