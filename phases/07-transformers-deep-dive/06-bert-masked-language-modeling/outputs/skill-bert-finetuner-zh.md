---
name: bert-finetuner
description: 为一个新的分类、抽取或检索任务 scope 一个 BERT fine-tune (微调) 方案。
version: 1.0.0
phase: 7
lesson: 6
tags: [bert, fine-tuning, nlp]
---

给定一个下游任务（classification (分类) / NER (命名实体识别) / retrieval (检索) / reranking (重排序) / NLI (自然语言推理)）、标注数据规模，以及部署约束（latency (延迟)、device (设备)），输出：

1. **Backbone choice (骨干网络选择)。** 模型名称（ModernBERT-base / large、DeBERTa-v3、multilingual-e5 等）及一句话理由。对于需要 ≤8K 上下文的英文任务，优先选择 ModernBERT。
2. **Head spec (输出层规范)。** Classification (分类)：`[CLS]` → dropout (随机失活) → linear(num_classes)。NER：per-token linear + CRF 可选。Retrieval (检索)：mean-pool (均值池化) + contrastive loss (对比损失)。
3. **Training recipe (训练配方)。** Optimizer (优化器)（AdamW，典型 lr 2e-5）、warmup % (预热比例)（6–10%）、epochs (轮次)（3–5）、batch size (批量大小)、fp16/bf16。
4. **Eval plan (评估计划)。** 任务相关的 metrics (指标)（classification (分类) 用 accuracy + F1，NER 用 entity-level F1，retrieval 用 MRR/NDCG）。Held-out split size (留出验证集大小)。
5. **Failure mode check (失效模式检查)。** 一个具名的风险：label leakage (标签泄露)、class imbalance (类别不平衡)、context truncation (上下文截断)、pretrain 和 fine-tune 语料之间的 tokenizer mismatch (分词器不匹配)。

拒绝在 generative output (生成式输出)（文本生成）上 fine-tune BERT —— 推荐改用 decoder-only (仅解码器) 模型。拒绝在 minority class (少数类) 低于 10% 时交付没有 class-stratified eval (按类别分层评估) 的 fine-tune。对于任何在 <1,000 条标注样本下解冻全部 backbone (骨干网络) 的 fine-tune，标记为 likely overfit (可能过拟合)。
