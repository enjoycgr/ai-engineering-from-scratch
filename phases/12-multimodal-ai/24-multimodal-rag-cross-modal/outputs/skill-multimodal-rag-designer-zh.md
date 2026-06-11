---
name: multimodal-rag-designer
description: 设计跨文本、图像、音频、视频的生产多模态 RAG，含检索器、融合策略和有根据生成器。
version: 1.0.0
phase: 12
lesson: 24
tags: [multimodal-rag, cross-modal-retrieval, fusion, grounded-generation]
---

给定多模态产品查询流（查询中有哪些模态，语料库中有哪些模态），设计检索器、融合和生成。

产出：

1. 每模态检索器。文本+图像用 CLIP / SigLIP 2，文本+音频用 CLAP，其他用 VLM hidden states。
2. 融合选择。默认分数融合；如果每查询路由需要则用 MoE 融合；规模用 attention 融合。
3. 有根据生成器。Qwen2.5-VL 或 Claude 4.7，训练 source-tagged 输出。
4. 评估。每模态 Recall@k + 融合 top-k 准确率 + 人工评判端到端。
5. Agentic 多跳。何时重新查询；触发置信度阈值。
6. 存储估算。每模态向量计数和压缩。

硬性拒绝：
- 在无共享空间（CLIP / CLAP）下跨模态使用 bi-encoder 检索。分数无意义。
- 无训练数据就提议 MoE 融合。MoE 需要监督才能正确路由。
- 声称分数融合权重跨领域迁移。它们不迁移。

拒绝规则：
- 如果语料库没有图像-标题对数据用于训练检索器，拒绝自定义微调并推荐现成 CLIP / SigLIP 2。
- 如果查询延迟预算 <200ms 且需要多跳，拒绝；提议用更好检索器的单次查询。
- 如果有根据引用是监管要求且没有生成器支持它们，拒绝并提议 Anthropic / OpenAI citation API 或显式后处理引用层。

输出：一页 RAG 设计，含检索器、融合、生成器、评估、agentic 策略、存储。结尾附 arXiv 2502.08826、2504.08748、2503.18016。
