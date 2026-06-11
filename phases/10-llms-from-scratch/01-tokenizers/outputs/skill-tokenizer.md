---
name: skill-tokenizer
description: 为 LLM 项目选择和构建 tokenizer
version: 1.0.0
phase: 10
lesson: 1
tags: [tokenizer, bpe, wordpiece, sentencepiece, llm, nlp]
---

# Tokenizer Selection and Implementation（分词器选择与实现）

启动 LLM 项目时，应用以下决策框架来选择 tokenizer。

## When to use each tokenizer（何时使用每种 tokenizer）

**Byte-level BPE (tiktoken)：** 你在基于 GPT 系列模型构建或 fine-tuning（微调）。你需要保证处理任何输入字节序列。你不希望出现未知 token。

**WordPiece (Hugging Face)：** 你在使用 BERT 系列模型进行分类、NER 或 embedding（嵌入）任务。你需要下游任务依赖词边界信号的 "##" 延续前缀。

**SentencePiece (BPE 或 Unigram)：** 你在从头训练。你需要语言无关的分词。你的数据包含 CJK 语言、泰文或其他没有空格词边界的文字。LLaMA、T5 和大多数多语言模型使用这个。

## Vocabulary size guidelines（词表大小指南）

- 32K tokens：单语言模型的良好默认值，保持 embedding layer 较小
- 50K-64K tokens：多语言或代码密集型模型的更好选择
- 100K+ tokens：只有当你有海量训练数据且想要更短序列时才使用

更大的 vocabulary 意味着更短的序列（更便宜的 inference）但 embedding matrix 中更多参数。对于 100K vocabulary 和 4096 维 embedding，仅 embedding layer 就有 400M 参数。

## Pre-tokenization rules that matter（重要的预分词规则）

1. 在 BPE 之前按空白拆分，防止跨词合并
2. 如果你想让模型学会算术，单独拆分每个数字
3. 分词前归一化 Unicode（NFC），确保行为一致
4. 为你的用例添加 special tokens：`<pad>`、`<eos>`、`<bos>`、`<unk>`，以及任何任务特定标记

## Red flags in tokenizer behavior（tokenizer 行为中的危险信号）

- 目标语言的 fertility 超过 2.0：模型浪费上下文窗口
- 常见领域词拆分为 3+ token：用领域数据重新训练
- 数字分词不一致：检查数字拆分规则
- 大 vocabulary 中有许多只用一次的 token：减小 vocabulary 大小

## Building a custom tokenizer - checklist（构建自定义 tokenizer 检查清单）

1. 收集代表性训练数据（目标领域至少 1GB 文本）
2. 选择算法：通用用例选 BPE，多语言选 Unigram
3. 基于上述指南设置 vocabulary 大小
4. 配置 pre-tokenization：空白拆分、数字处理、标点符号
5. 添加 special tokens
6. 使用 Hugging Face tokenizers 库训练（Rust 后端，速度快）
7. 验证：在所有目标语言的 held-out 文本上检查 fertility
8. 测试边界情况：空字符串、超长输入、二进制数据、emoji、RTL 文本
9. 将 tokenizer 与模型 checkpoint 一起保存和版本控制
