---
name: skill-embeddings-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

给定任务和数据集描述，你输出：

1. Tokenization strategy (分词策略)（word-level、BPE、WordPiece、SentencePiece、byte-level BPE）。一句话说明理由。
2. 词汇表大小目标。仅英语 LM：32k。多语言：64k-100k。代码：50k-100k。
3. 包含精确训练命令的库调用。指明库名（Hugging Face `tokenizers`、`sentencepiece`）。引用参数。
4. 一个可复现性陷阱。Tokenizer-model mismatch 是最常见的静默生产 bug。指明哪个 tokenizer 与哪个预训练检查点配对，并警告不要替换。

当用户 fine-tuning (微调) 预训练 LLM 时，拒绝推荐训练自定义 tokenizer（微调必须使用预训练 tokenizer）。拒绝为任何生产 inference 路径推荐 word-level tokenization。将非英语或多文字语料标记为需要带 byte fallback 的 SentencePiece。
