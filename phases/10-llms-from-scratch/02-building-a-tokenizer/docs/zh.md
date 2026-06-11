# Building a Tokenizer from Scratch（从零构建分词器）

> 第 01 课给了你玩具。这一课给你武器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## Learning Objectives

- 构建一个生产级的 BPE tokenizer，能处理 Unicode、空白字符归一化和 special tokens（特殊 token）
- 实现 byte-level fallback（字节级回退），使 tokenizer 能编码任何输入（包括 emoji、CJK 文字和代码）而不产生未知 token
- 添加 pre-tokenization（预分词）正则表达式模式，在应用 BPE 合并之前于词边界处切分文本
- 在语料上训练自定义 tokenizer，并在多语言文本上评估其 compression ratio（压缩比）与 tiktoken 的对比

## The Problem

你第 01 课的 BPE tokenizer 在英文文本上能工作。现在扔给它日文。或者 emoji。或者混用制表符和空格的 Python 代码。

它会崩。

不是因为 BPE 错了——而是实现不完整。一个生产级 tokenizer 要处理任何编码的原始字节、在切分前归一化 Unicode、管理永远不会被合并的 special tokens、将 pre-tokenization 与子词切分链式组合，并且速度足够快，不会成为处理 15 万亿 token 的训练流水线的瓶颈。

GPT-2 的 tokenizer 有 50,257 个 token。Llama 3 有 128,256 个。GPT-4 大约有 100,000 个。这些不是玩具数字。那些 vocabulary 背后的合并表是在数百 GB 的文本上训练的，而周围的机制——归一化、pre-tokenization、special token 注入、chat template（对话模板）格式化——才是区分一个只能处理 "hello world" 的 tokenizer 和一个能处理整个互联网的 tokenizer 的关键。

你要构建的就是这些机制。

## The Concept

### The Full Pipeline（完整流水线）

一个生产级 tokenizer 不是单一算法。它是一个五阶段流水线，每个阶段解决不同的问题。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每个阶段有特定的职责：

| Stage | What It Does | Why It Matters |
|-------|-------------|----------------|
| Normalize | NFKC Unicode, lowercase optional, strip accents optional | "fi" ligature (U+FB01) becomes "fi" (two chars). Without this, same word gets different tokens. |
| Pre-Tokenize | Split text into chunks before BPE | Prevents BPE from merging across word boundaries. "the cat" should never produce a token "e c". |
| BPE Merge | Apply learned merge rules to byte sequences | The core compression. Turns raw bytes into subword tokens. |
| Special Tokens | Inject [BOS], [EOS], [PAD], chat template markers | These tokens have fixed IDs. They never participate in BPE merges. The model needs them for structure. |
| ID Mapping | Convert token strings to integer IDs | The model sees integers, not strings. |

### Byte-Level BPE（字节级 BPE）

第 01 课的 tokenizer 操作于 UTF-8 字节。这是正确的选择。但我们跳过了重要的一点：当这些字节不是有效的 UTF-8 时会发生什么？

Byte-level BPE 通过将每个可能的字节值（0-255）视为有效 token 来解决这个问题。你的基础 vocabulary 恰好是 256 个条目。任何文件——文本、二进制、损坏的——都可以被分词而不产生未知 token。

GPT-2 加了一个技巧：将每个字节映射到一个可打印的 Unicode 字符，使 vocabulary 保持人类可读。字节 0x20（空格）在他们的映射中变成字符 "G"。这只是装饰性的。算法并不关心。

真正的威力在于：byte-level BPE 能处理地球上的每种语言。中文字符每个占 3 个 UTF-8 字节。日文可能是 3-4 字节。阿拉伯文、天城文、emoji——都只是字节序列。BPE 算法在这些字节序列中寻找模式的方式，与它在英文 ASCII 字节中寻找模式的方式完全相同。

### Pre-Tokenization（预分词）

在 BPE 接触你的文本之前，你需要把它切分成 chunks（块）。这防止合并算法创建跨越词边界的 token。

GPT-2 使用一个正则表达式模式来切分文本：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

这个模式在缩略形式（"don't" 变成 "don" + "'t"）、带可选前导空格的单词、数字、标点符号和空白字符处切分。前导空格保留在单词上——所以 "the cat" 变成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，它完全跳过了正则表达式。它将原始字节流视为一个长序列，让 BPE 算法自己找出边界。这更简单，但给 BPE 更多自由去创建跨词 token。

这个选择很重要。GPT-2 的正则防止 tokenizer 学习将一个词末尾的 "the" 和下一个词开头的 "the" 合并。SentencePiece 允许这样做，这有时会产生更高效的压缩，但 token 的可解释性更差。

### Special Tokens（特殊 Token）

每个生产级 tokenizer 都为结构标记保留 token ID：

| Token | Purpose | Used By |
|-------|---------|---------|
| `[BOS]` / `<s>` | Beginning of sequence | Llama 3, GPT |
| `[EOS]` / `</s>` | End of sequence | All models |
| `[PAD]` | Padding for batch alignment | BERT, T5 |
| `[UNK]` | Unknown token (byte-level BPE eliminates this) | BERT, WordPiece |
| `<\|im_start\|>` | Chat message boundary start | ChatGPT, Qwen |
| `<\|im_end\|>` | Chat message boundary end | ChatGPT, Qwen |
| `<\|user\|>` | User turn marker | Llama 3 |
| `<\|assistant\|>` | Assistant turn marker | Llama 3 |

Special tokens 永远不会被 BPE 切分。它们在合并算法运行之前被精确匹配，替换为固定 ID，周围的文本正常分词。

### Chat Templates（对话模板）

这是大多数人感到困惑、大多数实现出问题的地方。

当你向对话模型发送消息时，API 接受一个消息列表：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON。它看到一个扁平的 token 序列。Chat template 使用 special tokens 将消息转换成那个扁平序列。每个模型的做法都不同：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Template 搞错，模型就会输出垃圾。它是在一种精确的格式上训练的。任何偏差——缺少换行、token 顺序错了、多空格——都会让输入脱离训练分布。

### Speed（速度）

Python 对于生产级分词来说太慢了。

tiktoken（OpenAI）是用 Rust 编写并带有 Python 绑定的。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。它们比纯 Python 快 10-100 倍。

作为参考：以每秒 100 万 token 的速度（快速的 Python）为 Llama 3 的 pre-training（预训练）tokenize 15 万亿 token 需要 174 天。以每秒 1 亿 token 的速度（Rust），只需要 1.7 天。

你用 Python 构建是为了理解算法。在生产环境中，你会使用编译后的实现，只接触 Python 包装层。

## Build It

### Step 1: Byte-Level Encoding（字节级编码）

基础。将任何字符串转换成字节序列，将每个字节映射到一个可打印字符用于显示，并逆转这个过程。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用多语言文本测试以观察字节数：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 字节。"你好" 是 6 字节（每个字符 3 字节）。火焰 emoji 是 4 字节。字节级 tokenizer 不在乎是什么语言。字节就是字节。

### Step 2: Pre-Tokenizer with Regex（基于正则的预分词器）

使用 GPT-2 正则模式将文本切分成 chunks。每个 chunk 由 BPE 独立分词。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模块支持 Unicode 属性转义（`\p{L}` 表示字母，`\p{N}` 表示数字）。标准库 `re` 模块不支持，所以我们回退到 ASCII 字符类。对于生产级多语言 tokenizer，请安装 `regex`。

试一下：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

前导空格保留在单词上。缩略形式在撇号处切分。标点符号成为独立的 chunk。BPE 永远不会跨越这些边界合并 token。

### Step 3: BPE on Byte Sequences（字节序列上的 BPE）

第 01 课的核心算法，但现在独立地在预分词的 chunks 上操作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Step 4: Special Token Handling（特殊 Token 处理）

Special tokens 需要精确匹配和固定 ID。它们完全绕过 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Step 5: Full Tokenizer Class（完整 Tokenizer 类）

将所有环节链在一起：归一化、按 special tokens 切分、pre-tokenize、BPE 合并、映射为 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Step 6: Multilingual Test（多语言测试）

真正的测试。把英文、中文、emoji 和代码都扔给它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字符每个产生 3 字节。Emoji 产生 4 字节。这些都不会让 tokenizer 崩溃。都不会产生未知 token。这就是 byte-level BPE 的威力。

## Use It

### Comparing Real Tokenizers（对比真实 Tokenizer）

加载 Llama 3、GPT-4 和 Mistral 的实际 tokenizer。看看每个如何处理同一段多语言段落。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

你会看到同一段文本有不同的 token 数。Llama 3 的 128K vocabulary 更激进地合并常见模式。GPT-4 的 100K 居中。Mistral 的 32K 产生更多 token，但 embedding layer 更小。

权衡始终相同：更大的 vocabulary 意味着更短的序列，但更多参数。

## Ship It

本课程产出一个用于构建和调试生产级 tokenizer 的 prompt。参见 `outputs/prompt-tokenizer-builder.md`。

## Exercises

1. **Easy：** 添加一个 `get_token_bytes(id)` 方法，显示任意 token ID 的原始字节。用它来检查你最常见的合并 token 实际代表什么。

2. **Medium：** 实现 Llama 风格的 pre-tokenizer，它在空白和数字处切分但保留前导空格。在相同语料上将其与 GPT-2 正则方法的 vocabulary 进行对比。

3. **Hard：** 添加一个 chat template 方法，接受 `{"role": ..., "content": ...}` 消息列表，并生成 Llama 3 对话格式的正确 token 序列。与 HuggingFace 实现进行对比测试。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Byte-level BPE | "Tokenizer that works on bytes" | BPE with a base vocabulary of 256 byte values -- handles any input without unknown tokens |
| Pre-tokenization | "Splitting before BPE" | Regex or rule-based splitting that prevents BPE from merging across word boundaries |
| NFKC normalization | "Unicode cleanup" | Canonical decomposition followed by compatibility composition -- "fi" ligature becomes "fi", fullwidth "A" becomes "A" |
| Chat template | "How messages become tokens" | The exact format for converting a list of role/content messages into a flat token sequence -- model-specific and must match training format |
| Special tokens | "Control tokens" | Reserved token IDs that bypass BPE -- [BOS], [EOS], [PAD], chat markers -- matched exactly before merge |
| Fertility | "Tokens per word" | Ratio of output tokens to input words -- 1.3 for English in GPT-4, 2-3 for Korean, higher means wasted context |
| tiktoken | "OpenAI tokenizer" | Rust BPE implementation with Python bindings -- 10-100x faster than pure Python |
| Merge table | "The vocabulary" | Ordered list of byte-pair merges learned during training -- this IS the tokenizer's learned knowledge |

## Further Reading

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- Rust BPE implementation used by GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- Rust tokenizer library supporting BPE, WordPiece, Unigram
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- details on 128K vocabulary and tokenizer training
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- language-agnostic tokenization
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- the original byte-to-Unicode mapping
