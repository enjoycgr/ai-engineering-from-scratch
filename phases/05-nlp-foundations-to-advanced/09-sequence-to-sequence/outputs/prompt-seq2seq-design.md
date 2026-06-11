---
name: seq2seq-design
description: 为给定任务设计 sequence-to-sequence (序列到序列) 流水线。
phase: 5
lesson: 09
---

给定一个任务（翻译、摘要、改写、问题重写），输出：

1. **架构 (Architecture)。** 预训练 transformer encoder-decoder (BART, T5, mBART, NLLB) 是默认选择。仅在特定约束（流式、边缘推理、教学）下使用基于 RNN 的 seq2seq。
2. **起始检查点 (Starting checkpoint)。** 明确命名（`facebook/bart-base`、`google/flan-t5-base`、`facebook/nllb-200-distilled-600M`）。根据任务和语言覆盖范围匹配合适的检查点。
3. **解码策略 (Decoding strategy)。** Greedy decoding (贪心解码) 用于确定性输出，beam search (束搜索，宽度 4-5) 用于质量，temperature sampling (温度采样) 用于多样性。一句话说明理由。
4. **发布前需验证的一个失败模式。** Exposure bias (暴露偏差) 表现为长输出上的生成漂移；在 90th-percentile 长度处采样 20 个输出并人工检查。

拒绝为少于约 100 万平行句对的任务推荐从零训练 seq2seq。将任何对用户可见内容使用 greedy decoding 的流水线标记为脆弱（greedy 会重复和循环）。
