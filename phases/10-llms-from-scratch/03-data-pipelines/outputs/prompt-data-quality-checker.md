---
name: prompt-data-quality-checker
description: 验证和调试 LLM 预训练流水线中的数据质量
version: 1.0.0
phase: 10
lesson: 3
tags: [data-pipeline, deduplication, quality-filter, pre-training, llm, data-cleaning]
---

# Data Quality Checker for LLM Pre-Training（LLM 预训练数据质量检查器）

为 LLM 预训练构建或审计数据流水线时，使用此框架在问题到达模型之前捕获它们。

## Red Flags in Pipeline Output（流水线输出中的危险信号）

**去重移除了不到 20% 的网页数据。** Common Crawl 通常包含 30-40% 的重复。如果你的去重步骤移除不到 20%，你的 MinHash 参数太保守或阈值太高。检查：shingle 大小 k、哈希函数数量、LSH band 数量、Jaccard 阈值。

**Compression ratio（压缩比）低于 2.0 字符/token。** 这意味着你的 tokenizer 切分太激进。要么用更多合并重新训练，增加 vocabulary 大小，要么检查 pre-tokenization 是否不必要地碎片化文本。

**Compression ratio 高于 6.0 字符/token。** 你的 tokenizer 学了非常特定领域的合并，可能无法泛化。这对领域特定模型没问题，但对通用模型是警告信号。

**Sequence utilization（序列利用率）低于 90%。** 太多 padding。要么你的文档很短（过滤它们或增加最小文档长度），要么你的 sequence packing 效率低下（从朴素 padding 切换到多文档打包）。

**Vocab utilization（词表利用率）低于 50%。** 超过一半的 vocabulary 在此语料上未使用。要么 vocabulary 对你的领域来说太大，要么 tokenizer 是在非常不同的数据上训练的。

## Quality Filter Calibration（质量过滤器校准）

在流水线的每个阶段对 1,000 个随机文档样本运行这些检查：

1. **清洗后阅读 20 个随机文档。** 它们是否包含残留 HTML、JavaScript、导航文本或样板？如果是，你的 HTML 剥离不完整。

2. **阅读 20 个通过质量过滤器的随机文档。** 其中是否有垃圾邮件、关键词列表或机器生成的内容？如果是，收紧过滤器阈值。

3. **阅读 20 个未通过质量过滤器的随机文档。** 其中是否有真正好的内容？如果是，你的过滤器太激进。放宽阈值或为特定模式添加例外。

4. **阅读 20 个来自去重的随机近似重复对。** 它们是否真的相似？如果不是，降低 Jaccard 阈值或增加哈希函数数量。

## Data Mixing Ratios（数据混合比例）

没有通用公式。从这些基线开始并根据评估进行调整：

| Category | Llama 3 Ratio | Starting Point |
|----------|--------------|----------------|
| Web text | 50% | 50% |
| Code | 25% | 15-25% |
| Books/academic | 13% | 10-15% |
| Math | 8% | 5-10% |
| Multilingual web | 4% | 5-10% |

如果模型应该擅长编程，增加代码比例。如果推理很重要，增加数学比例。如果需要更少噪声，减少网页比例。更改比例后始终在基准测试上评估。

## Scaling Estimates（扩展估算）

对于给定的目标 token 数量：

- 1T token 来自网页：预期 ~3-5TB 原始文本，清洗和去重后 ~1.5-2TB
- 分词速度（Rust）：~100M token/秒/核心
- 分词速度（Python）：~1-10M token/秒/核心
- MinHash 去重（128 哈希，16 band）：~10K 文档/秒/核心
- Sequence packing：I/O 受限，语料超过 10GB 时使用内存映射文件

对于 15T token（Llama 3 规模），计划 ~30-50TB 原始输入数据，在 64 核机器上 1-2 周的预处理时间，以及 100TB+ 的中间文件磁盘空间。

## Checklist Before Training（训练前检查清单）

1. 总 token 数匹配你的算力预算（使用 Chinchilla scaling law 或 Llama 3 过训练比例作为指南）
2. 去重移除了 30-40% 的网页数据
3. 质量过滤器移除了剩余数据的 10-20%
4. Compression ratio 对英文为 3-5 字符/token
5. Sequence utilization 高于 95%
6. 随机抽查显示每个流水线阶段都有干净、连贯的文本
7. 数据混合比例已在小规模训练运行上验证
8. PII 移除已在样本上验证
9. 所有二进制格式（打包序列、token ID 数组）通过往返编码/解码测试
10. 流水线是可复现的：相同输入在固定随机种子下产生相同输出
