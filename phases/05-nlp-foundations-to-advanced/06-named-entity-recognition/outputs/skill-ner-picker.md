---
name: ner-picker
description: 为给定的抽取任务选择正确的 NER 方法。
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

给定任务描述（domain（领域）、label set（标签集）、language（语言）、latency（延迟）、data volume（数据量）），输出：

1. Approach（方法）。Rule-based + gazetteer（基于规则 + 地名词典）、CRF、BiLSTM-CRF 或 transformer fine-tune（微调）。
2. Starting model（起始模型）。命名它（spaCy model ID 如 `en_core_web_sm` / `en_core_web_trf`、Hugging Face checkpoint ID 如 `dslim/bert-base-NER`、或 "custom, trained from scratch（自定义，从头训练）"）。
3. Labeling strategy（标注策略）。BIO、BILOU 或 span-based（基于跨度）。用一句话说明理由。
4. Evaluation（评估）。使用 `seqeval`。始终报告 entity-level F1（实体级 F1），不要报告 token-level。

拒绝推荐在少于 500 个标注样本的情况下 fine-tune transformer，除非用户已有 pretrained domain model（预训练领域模型，例如医学领域的 BioBERT）。标记 nested entities（嵌套实体）需要使用 span-based 或 multi-pass models。如果用户提到 "production scale（生产规模）" 同时标签与 CoNLL-2003 保持不变，则要求进行 gazetteer audit（地名词典审计）。
