---
name: prompt-tokenizer-builder
description: 为 LLM 项目构建和调试生产级 tokenizer
version: 1.0.0
phase: 10
lesson: 2
tags: [tokenizer, bpe, byte-level, special-tokens, chat-template, multilingual]
---

# Production Tokenizer Builder（生产级分词器构建器）

为 LLM 项目构建或调试 tokenizer 时，遵循此框架。

## Pipeline Checklist（流水线检查清单）

每个生产级 tokenizer 都需要这五个阶段。如果缺少任何一个，你将在生产中遇到边界情况。

1. **Normalize（归一化）** -- 应用 NFKC Unicode 归一化。这会折叠连字（"fi" -> "fi"）、归一化全角字符、标准化空白字符。跳过这一步，同一个词会根据输入方式不同而获得不同的 token ID。

2. **Pre-Tokenize（预分词）** -- 在 BPE 之前将文本拆分为 chunks。英语中心模型使用 GPT-2 的正则模式。多语言模型使用 SentencePiece 的原始字节方法。这个选择决定了 BPE 是否能跨词边界合并。

3. **BPE Merge（BPE 合并）** -- 将学习到的合并表应用于每个 chunk 内的字节序列。合并表就是 tokenizer 学到的知识。其他一切都是管道工程。

4. **Special Token Injection（特殊 Token 注入）** -- 在 BPE 运行前精确匹配 special tokens。[BOS]、[EOS]、[PAD]、对话模板标记获得固定 ID。它们永远不会参与合并。

5. **ID Mapping（ID 映射）** -- 将 token 字符串转换为整数。模型只看见整数。

## Debugging Tokenizer Issues（调试 Tokenizer 问题）

**症状：模型在对话输入上输出垃圾**
- 检查 chat template。每个模型有不同的格式。Llama 3 使用 `<|start_header_id|>` 标记。ChatGPT 使用 `<|im_start|>` 标记。错误的模板会让输入脱离训练分布。

**症状：非英语文本使用太多 token**
- 检查 fertility（每词 token 数）。超过 2.0 意味着 tokenizer 在该语言上浪费上下文窗口。解决方案：用更多多语言数据重新训练、增加 vocabulary 大小，或使用 Unigram 的 SentencePiece。

**症状：数字和算术失败**
- 检查数字如何被分词。"1234" 作为一个 token 意味着模型无法进行逐位操作。在预分词期间单独拆分每个数字。

**症状：代码 token 效率低下**
- 检查缩进如何处理。GPT-2 的 tokenizer 在空格上浪费 token。Codex 和 StarCoder 使用特殊缩进 token（4 个空格 = 1 个 token）。

## Vocabulary Size Decision（词表大小决策）

- 32K tokens：单语言、小模型、有限算力。Embedding layer 是 32K * d_model 参数。
- 50K-64K：多语言或代码密集型。大多数项目的良好平衡。
- 100K+（GPT-4、Llama 3）：只有海量训练数据时才使用。序列更短但 embedding 参数为 100K * d_model。

对于 4096 维模型：32K vocab = 131M embedding 参数。128K vocab = 524M embedding 参数。仅 embedding layer 就有 4 亿参数的差异。

## Speed Requirements（速度要求）

- 训练数据分词：使用 Rust 支持的库（tiktoken、HuggingFace tokenizers）。纯 Python 慢 10-100 倍。
- 推理分词：延迟影响较小（单序列），但仍使用编译实现。
- 基准测试：分词 1GB 文本并测量 wall clock time。如果超过 60 秒，切换到 Rust 后端。

## Chat Template Validation（对话模板验证）

部署任何对话模型前，验证模板：

1. 用 tokenizer 编码一段已知对话
2. 将其解码回文本
3. 与模型文档中的预期格式逐字符对比
4. 注意：header token 后的换行、内容前的空格、turn 结束标记
5. 测试边界情况：空 system 消息、超长 user 消息、多轮 assistant 回复

弄错 chat template 是聊天模型性能下降的最常见原因。
