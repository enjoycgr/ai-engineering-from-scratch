---
name: seq2seq-picker
description: 为一个新的序列到序列（sequence-to-sequence）任务选择编码器-解码器（encoder-decoder）还是仅解码器（decoder-only）。
version: 1.0.0
phase: 7
lesson: 8
tags: [transformers, t5, bart, seq2seq]
---

给定一个 seq2seq 任务（翻译 / 摘要 / 语音转文本 / 结构化抽取 / 改写）、输入和输出的长度分布，以及质量与延迟的优先级，输出：

1. 架构（Architecture）。以下之一：编码器-解码器（T5 / BART / Whisper 风格）、仅解码器指令微调（decoder-only instruction-tuned）、仅编码器 + 提示模板（encoder-only + prompt template）。附一句理由。
2. 预训练目标（Pretraining objective）。跨度破坏（span corruption，T5）、去噪（denoising，BART）、下一 token 预测（next-token，仅解码器），或“跳过预训练，直接微调现有检查点”。需指明检查点名称。
3. 输入格式（Input formatting）。任务前缀字符串（T5 风格）vs 系统提示（system prompt，仅解码器）vs 原始 token（BART）。包含 BOS/EOS 处理方式。
4. 解码策略（Decoding strategy）。束搜索宽度（beam search width）和长度惩罚（length penalty，用于翻译/摘要），或核采样/最小 P（nucleus/min-p，用于类聊天任务）。需说明该任务适用哪一种。
5. 评估（Eval）。任务相关的指标：BLEU / ROUGE / WER / F1 / exact match。包含测试集划分大小。

拒绝为生成式输出推荐仅编码器架构。拒绝在输入已经是对话的情况下推荐编码器-解码器——仅解码器更自然地适配对话记忆。若推荐仅解码器处理语音转文本任务，必须提及 Whisper 作为需要超越的基线。
