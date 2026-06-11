# Machine Translation（机器翻译）

> 翻译是三十年来为 NLP 研究买单的任务，并且至今仍在买单。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 分钟

## The Problem（问题）

模型读取一种语言的句子，并生成另一种语言的句子。长度不同。词序不同。有些源词映射到多个目标词，反之亦然。习语拒绝一对一映射。"I miss you" 在法语中是 "tu me manques"——字面意思是 "你对我有所欠缺"。没有任何词级别的对齐能在这种情况下存活。

Machine translation（机器翻译）是推动 NLP 发明 encoder-decoder（编码器-解码器）、attention（注意力）、transformer，并最终催生整个 LLM 范式的任务。每一步进步都来自于翻译质量可衡量，而人机差距又顽固不化。

本课跳过历史课，教授 2026 年的工作 pipeline（流水线）：pre-trained（预训练）multilingual（多语言）encoder-decoder（NLLB-200 或 mBART）、subword tokenization（子词分词）、beam search（束搜索）、BLEU 和 chrF 评估，以及仍然会未经过检测就部署到生产环境的少数几种 failure mode（失效模式）。

## The Concept（概念）

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

Modern MT 是一个在平行文本上训练的 transformer encoder-decoder（编码器-解码器）。Encoder 读取源语言文本，使用其语言的分词方式。Decoder 通过 cross-attention（交叉注意力，见第 10 课）利用 encoder 的输出，一次生成一个 subword（子词）。Decoding 使用 beam search 以避免 greedy decoding（贪心解码）的陷阱。输出经过 detokenize（去分词）、detruecase（还原大小写）后，再与 reference（参考译文）进行评分。

三个操作层面的选择决定了真实世界中的 MT 质量。

- **Tokenizer（分词器）。** 在混合语言语料上训练的 SentencePiece BPE。跨语言共享词汇表是 NLLB 实现 zero-shot（零样本）语言对的关键。
- **Model size（模型大小）。** NLLB-200 distilled 600M 可在笔记本电脑上运行。NLLB-200 3.3B 是公开发布的生产默认模型。54.5B 是研究天花板。
- **Decoding（解码）。** 通用内容使用 beam width（束宽）4-5。使用 length penalty（长度惩罚）避免输出过短。需要术语一致性时使用 constrained decoding（约束解码）。

## Build It（动手实现）

### Step 1: a pretrained MT call（调用预训练 MT 模型）

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

这里有三个关键点。`src_lang` 告诉 tokenizer 应用哪种文字和分词方式。`forced_bos_token_id` 告诉 decoder 生成哪种语言。两者都是 NLLB 特有的技巧；mBART 和 M2M-100 使用各自的约定，它们之间不可互换。

### Step 2: BLEU and chrF

BLEU 衡量输出与 reference 之间的 n-gram 重叠。四个 reference n-gram 大小（1-4），precisions（精确率）的几何平均，以及对过短输出的 brevity penalty（简短惩罚）。分数范围是 [0, 100]。常用但难以解释：30 BLEU 是"可用"；40 是"良好"；50 是"卓越"；1 BLEU 以下的差异属于噪声。

chrF 衡量字符级别的 F-score。对形态丰富的语言更敏感，因为 BLEU 会漏掉一些匹配。通常与 BLEU 一起报告。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它规范化 tokenization，使得分数在不同论文之间可比较。自己实现 BLEU 计算是产生误导性 benchmark（基准测试）的常见原因。

### The three-tier evaluation hierarchy（三层评估体系，2026）

Modern MT evaluation 使用三种互补的 metric family（指标族）。部署时至少使用两种。

- **Heuristic（启发式）**（BLEU, chrF）。快速、基于 reference、可解释，但对 paraphrase（意译）不敏感。用于 legacy comparison（遗留对比）和 regression detection（回归检测）。
- **Learned（学习式）**（COMET, BLEURT, BERTScore）。在人类判断上训练的神经网络模型；比较翻译与源文本及 reference 的语义相似度。COMET 自 2023 年以来与 MT 研究的关联度最高，在 2026 年质量优先的生产环境中是默认选择。
- **LLM-as-judge（LLM 作为评判者）**（无 reference）。提示大模型从流畅度、充分性、语气、文化适当性等维度为翻译评分。当评分标准设计良好时，GPT-4-as-judge 与人类一致率约为 80%。用于没有 reference 的开放式内容。

2026 年实用 stack：`sacrebleu` 用于 BLEU 和 chrF，`unbabel-comet` 用于 COMET，以及一个 prompted LLM 作为最终面向人类的信号。在将其用于生产数据之前，先用 50-100 条人工标注样本校准每个 metric。

Reference-free metrics（无参考指标）（COMET-QE, BLEURT-QE, LLM-as-judge）允许你在没有 reference 的情况下评估翻译，这对 long-tail（长尾）语言对很重要，因为这些语言对往往没有 reference translation（参考译文）。

### Step 3: what breaks in production（生产中会出现什么问题）

上述工作 pipeline 80% 的时间能流畅翻译，剩余 20% 会静默失败。已命名的 failure mode：

- **Hallucination（幻觉）。** 模型编造源文本中不存在的内容。在不熟悉的领域词汇中常见。症状：输出流畅，但声称了源文本未陈述的事实。缓解措施：对领域术语使用 constrained decoding，对受监管内容进行人工审核，监控输出长度是否远超输入。
- **Off-target generation（目标语言错误生成）。** 模型翻译成了错误的语言。NLLB 在 rare language pairs（稀有语言对）上 surprisingly（出人意料地）容易出现此问题。缓解措施：验证 `forced_bos_token_id`，并在输出后始终用 language-ID model 进行检查。
- **Terminology drift（术语漂移）。** "Sign up" 在文档 1 中变成 "s'inscrire"，在文档 2 中变成 "créer un compte"。对于 UI 文本和面向用户的字符串，一致性比原始质量更重要。缓解措施：glossary-constrained decoding（词汇表约束解码）或 post-edit dictionary（译后编辑词典）。
- **Formality mismatch（正式程度不匹配）。** 法语 "tu" 与 "vous"、日语的敬语级别。模型会选择训练集中更常见的形式。对于面向客户的内容，这通常是错误的。缓解措施：如果模型支持，在 prompt 前添加 formality token（正式程度标记）；或者在一个仅包含正式语料的语料库上 fine-tune（微调）一个小模型。
- **Length explosion on short input（短输入长度爆炸）。** 非常短的输入句子常常产生过长的翻译，因为长度惩罚在少于约 5 个源 token 时会失效。缓解措施：设置与源长度成比例的硬最大长度上限。

### Step 4: fine-tuning for a domain（领域微调）

Pretrained models（预训练模型）是通才。法律、医学或游戏对话翻译从领域平行数据的 fine-tuning 中获益显著。配方并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千条高质量的平行示例胜过几十万条嘈杂的网页抓取数据。训练数据的质量是生产中最大的单一杠杆。

## Use It（使用）

2026 年 MT 生产 stack：

| Use case | Recommended starting point |
|---------|---------------------------|
| Any-to-any, 200 languages | `facebook/nllb-200-distilled-600M` (laptop) or `nllb-200-3.3B` (production) |
| English-centric, high quality, 50 languages | `facebook/mbart-large-50-many-to-many-mmt` |
| Short runs, cheap inference, English-French/German/Spanish | Helsinki-NLP / Marian models |
| Latency-critical browser-side | ONNX-quantized Marian (~50 MB) |
| Maximum quality, willing to pay | GPT-4 / Claude / Gemini with translation prompts |

截至 2026 年，LLM 在多个语言对上已经超越了 specialized MT models（专用 MT 模型），尤其是在习语内容和长上下文方面。权衡在于 per-token cost（按 token 计费的成本）和 latency（延迟）。当 context length（上下文长度）、stylistic consistency（风格一致性）或通过 prompting 实现的 domain adaptation（领域适配）比 throughput（吞吐量）更重要时，选择 LLM。

## Ship It（部署）

Save as `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Exercises（练习）

1. **Easy.** 使用 `nllb-200-distilled-600M` 将一段 5 句英文段落翻译成法文，再翻译回英文。测量 round-trip（回译）结果与原文的接近程度。你应该能看到语义保留，但用词有漂移。
2. **Medium.** 使用 `fasttext lid.176` 或 `langdetect` 对翻译输出实现 language-ID check。将其集成到 MT 调用中，以便在返回前捕获 off-target generations。
3. **Hard.** 在你选择的 5,000 对领域语料库上 fine-tune `nllb-200-distilled-600M`。测量 fine-tuning 前后在 held-out set（留出集）上的 BLEU。报告哪些句子改善了，哪些退化了。

## Key Terms（关键术语）

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BLEU | Translation score | N-gram precision with brevity penalty. [0, 100]. |
| chrF | Character F-score | Character-level F-score. More sensitive for morphologically rich languages. |
| NMT | Neural MT | Transformer encoder-decoder trained on parallel text. The 2017+ default. |
| NLLB | No Language Left Behind | Meta's 200-language MT model family. |
| Constrained decoding | Controlled output | Force specific tokens or n-grams to appear / not appear in the output. |
| Hallucination | Invented content | Model output that is not supported by the source. |

## Further Reading（延伸阅读）

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为什么 `sacrebleu` 是报告 BLEU 的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 实用 fine-tuning 教程。
