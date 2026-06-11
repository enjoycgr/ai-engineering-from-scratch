---
name: qa-architect
description: 选择 QA 架构、检索策略和评估方案。
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

给定需求（语料库大小、问题类型、事实性约束、延迟预算），输出：

1. 架构 (Architecture)。抽取式 (Extractive)、RAG + 抽取式 reader、RAG + 生成式 reader，或闭卷 LLM。一句话说明理由。
2. 检索器 (Retriever)。无、BM25、dense（命名编码器如 `all-MiniLM-L6-v2`），或 hybrid（混合检索）。
3. 阅读器 (Reader)。SQuAD 微调模型（如 `deepset/roberta-base-squad2`）、指定名称的 LLM，或 "domain-fine-tuned DistilBERT"。
4. 评估 (Evaluation)。抽取式基准使用 EM + F1；生产环境使用 answer accuracy（答案准确率）+ citation accuracy（引用准确率）+ refusal calibration（拒答校准）。说明测量什么以及如何测量。

对于监管或合规敏感问题，拒绝使用闭卷 LLM 回答。拒绝任何没有 retrieval-recall 基线的 QA 系统（如果不知道检索器是否召回了正确段落，就无法评估 reader）。对于需要多跳推理 (multi-hop reasoning) 的问题，标记为需要专门的多跳检索器，如 HotpotQA 训练的系统。
