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
