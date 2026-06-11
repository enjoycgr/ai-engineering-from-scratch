# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本，拆分为 chunk，嵌入 chunk，存储向量。每一步都丢失信号：OCR 丢弃图表数据，分块破坏表格行，文本嵌入忽略图表。ColPali (Faysse 等人, 2024 年 7 月) 问了一个更简单的问题：为什么完全提取文本？通过 PaliGemma 直接嵌入页面图像，使用 ColBERT 风格 late interaction 检索，保留文档携带的所有布局、图表、字体和格式信号。发布基准：视觉丰富文档上端到端准确率比文本 RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了模式。本课阅读视觉原生 RAG 论点并构建微型 ColPali 类索引器。

**类型：** Build
**语言：** Python (stdlib, multi-vector indexer + MaxSim scorer)
**前置知识：** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**时间：** ~180 分钟

## 学习目标

- 解释 bi-encoder 检索（每文档一个向量）与 late-interaction 检索（每文档多个向量）之间的差异。
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 推广到图像 patch。
- 构建微型 ColPali 类索引器：页面 → patch embedding → query-term embedding 上的 MaxSim → top-k 页面。
- 在发票 / 财务报告用例上对比 ColPali + Qwen2.5-VL 生成器 vs 文本 RAG + GPT-4。

## 问题

PDF 上的文本 RAG 丢弃了文档的大部分。财务报告的 Q3 收入增长通常在图表中；医学报告的发现结果在注释图像中；法律合同的签名块是布局事实，不是文本事实。

文本 RAG 流水线：

1. PDF → 通过 OCR / pdftotext 提取文本。
2. 文本 → 300-500 token chunk。
3. Chunk → bi-encoder embedding（一个向量）。
4. 用户查询 → embedding → 余弦相似度 → top-k chunk。
5. Chunk + 查询 → LLM。

五步丢失。图表未捕捉。表格跨 chunk 破坏。多栏布局展平。图表注释消失。

ColPali 修复：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格 late interaction 检索，让模型在查询时关注细粒度 patch。

## 概念

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) 是文本检索方法。不是每文档一个向量，它每 token 产生一个向量。查询时：

- 查询 token 获得自己的 embedding（N_q 向量）。
- 文档 token 获得 embedding（N_d 向量，通常缓存）。
- 分数 = 查询 token 上文档 token 余弦相似度最大值的和：Σ_i max_j cos(q_i, d_j)。

这是 MaxSim 操作。每个查询 token "挑选"其最佳匹配文档 token。最终分数是和。

优势：强召回，处理词级语义。劣势：每文档 N_d 向量，存储昂贵。

### ColPali

ColPali (Faysse 等人, arXiv:2407.01449) 将 ColBERT 模式应用于图像。

- 每页由 PaliGemma（ViT + 语言）编码为 patch embedding：每页 N_p 向量。
- 每个用户查询（文本）编码为查询 token embedding：N_q 向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，即查询文本 token 和页面图像 patch 上的 MaxSim。
- 按总分数检索 top-k 页面。

文档摄入时：用 PaliGemma 嵌入每页，存储所有 patch embedding。查询时：嵌入查询 token，对所有存储页面 embedding 计算 MaxSim，返回 top-k 页面。

优势：端到端在视觉丰富文档上击败文本 RAG 20-40%。每 patch 向量捕捉局部布局和内容。

劣势：N_p patch × 4 字节浮点 × D-dim 向量每页 = 存储快速增长。PQ / OPQ 量化缓解。

### ColQwen2 和 ColSmol

ColQwen2 (illuin-tech, 2024-2025) 将 PaliGemma 换为 Qwen2-VL。更好的基础编码器，更好的检索。

ColSmol 是本地 / 边缘使用的小规模变体。~1B 参数的 ColSmol 检索器在消费级 GPU 上运行。

### VisRAG

VisRAG (Yu 等人, arXiv:2410.10594) 是不同的变体：不是 patch 上的 MaxSim，而是将每页 pool 为单个向量并用 VLM 然后 bi-encoder 检索。更快的索引 + 更小存储，更弱召回。

质量 vs 成本权衡：质量选 ColPali，规模选 VisRAG。

### M3DocRAG

M3DocRAG (Cho 等人, arXiv:2411.04952) 将多模态检索扩展到多页多文档推理。跨文档检索页面，为多页上下文组合给 VLM。

### ViDoRe — 基准

ColPali 的配套基准。Visual Document Retrieval Evaluation。任务包括财务报告、科学论文、行政文档、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上评分 ~80% nDCG@5；同一文档上文本 RAG 评分 ~50-60%。

### 端到端 RAG 流水线

视觉原生 RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch embedding。
2. 查询：用户文本 → 查询 token embedding → 对所有索引页面 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

无处 OCR。图表、字体、布局都流入答案。

### 存储数学

50 页财务报告，每页 729 patch，128-dim embedding：

- ColPali：50 * 729 * 128 * 4 字节 = ~18 MB 原始，PQ 后 ~4 MB。
- 文本 RAG：50 chunk * 768-dim * 4 字节 = ~150 kB。

ColPali 每文档多 ~30 倍存储。规模上 OPQ / PQ 将其降到 ~5-10 倍，通常可容忍。

### 文本 RAG 仍获胜时

- 无布局信号的纯文本文档（维基文章、聊天记录）。文本 RAG 更简单且存储更便宜。
- 存储主导成本的多百万页档案。
- 需要可提取 OCR 文本的严格监管要求。

2026 年其他一切——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 获胜。

## 使用它

`code/main.py`：

- 玩具 patch 编码器：将"页面"（小特征向量网格）映射为 patch embedding 数组。
- MaxSim 评分器：计算查询 token embedding 集与页面 patch 集之间的 ColBERT 风格分数。
- 索引 5 个玩具页面，运行 3 个查询，返回带分数的 top-k。

## 交付它

本课产生 `outputs/skill-vision-rag-designer.md`。给定文档 RAG 项目，在 ColPali / ColQwen2 / VisRAG / 文本 RAG 之间挑选并调整存储大小。

## 练习

1. 200 页年度报告在 729 patch/页、128-dim emb、4 字节浮点。计算原始存储和 PQ 压缩（8x）存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个和捕获了简单均值相似度未捕获的什么？

3. ColPali 将页面索引为 patch 集。如果改在词级索引（如 ColBERT 所做）什么改变？权衡？

4. 为 500ms 查询延迟预算的 100 万页语料库设计端到端流水线。挑选 ColQwen2 / VisRAG 并证明。

5. 阅读 M3DocRAG (arXiv:2411.04952)。描述多页 attention 模式及它与单页 ColPali 检索的差异。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Late interaction | "ColBERT-style" | 使用每 token 或每 patch embedding + MaxSim 的检索，非单文档向量 |
| MaxSim | "Max-over-patches" | 每个查询 token 挑选最高相似度文档 token；跨查询求和 |
| Bi-encoder | "Single-vector" | 每文档一个向量；更快但丢失粒度 |
| Multi-vector | "Many-vectors-per-doc" | 每文档/页存储 N_p 向量；存储成本增长但召回改善 |
| Patch embedding | "Page feature" | 来自 VLM 编码器的每图像 patch 向量，每页缓存 |
| ViDoRe | "Vision doc bench" | ColPali 的视觉文档检索基准套件 |
| PQ quantization | "Product quantization" | 保持向量相似性的压缩，缩小存储 ~8x |

## 延伸阅读

- [Faysse 等人 — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu 等人 — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho 等人 — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
