---
name: mha-configurator
description: 为新的 transformer 推荐 head count (头数)、KV-head count (KV 头数) 和 projection strategy (投影策略)（MHA / MQA / GQA / MLA）。
version: 1.0.0
phase: 7
lesson: 3
tags: [transformers, attention, mha, gqa]
---

Given a transformer spec (parameter budget, hidden size `d_model`, target context length, inference device memory, training vs inference priority), output:

给定一个 transformer 规格（参数预算、隐藏层大小 `d_model`、目标上下文长度、推理设备内存、训练 vs 推理优先级），输出：

1. Projection variant (投影变体). One of: MHA, GQA, MQA, MLA. One-sentence reason tied to KV-cache constraints.
   投影变体。可选：MHA、GQA、MQA、MLA。一句话说明原因，需关联 KV-cache (KV 缓存) 约束。
2. Head geometry (头几何配置). `n_heads`, `n_kv_heads`, `d_head`. Values must satisfy `d_model = n_heads * d_head` and `n_heads % n_kv_heads == 0`.
   头几何配置。`n_heads`、`n_kv_heads`、`d_head`。数值必须满足 `d_model = n_heads * d_head` 且 `n_heads % n_kv_heads == 0`。
3. KV cache estimate (KV 缓存估算). Bytes per token per layer (fp16) for the chosen variant at the target context length. Flag if one batch exceeds the target device memory.
   KV 缓存估算。在目标上下文长度下，所选变体每层每个 token 的 fp16 字节数。如果单个 batch (批次) 超出目标设备内存，需标记警告。
4. Initialization (初始化). Xavier / Kaiming scale for Q, K, V, O matrices. Note whether bias terms are included (most 2026 models drop them).
   初始化。Q、K、V、O 矩阵的 Xavier / Kaiming 缩放。注明是否包含 bias terms (偏置项)（大多数 2026 年的模型已去掉偏置）。
5. Testability hook (可测试性钩子). A single synthetic task (e.g. induction-head pattern `A B A ? → B`) that a trained two-layer version of this config should solve to ≥95% on.
   可测试性钩子。一个单一的 synthetic task (合成任务)（例如 induction-head (归纳头) 模式 `A B A ? → B`），该配置训练后的两层版本应能达到 ≥95% 的解决率。

Refuse to recommend `d_head < 32` — attention dynamics break down. Refuse to recommend MHA with `n_heads > 16` for context lengths above 32K without explicitly pricing the KV cache and suggesting GQA or MLA instead. Refuse to suggest MLA for models under 1B parameters unless the user is explicitly benchmarking it.

拒绝推荐 `d_head < 32` —— attention dynamics (注意力动态) 会崩溃。对于上下文长度超过 32K 的情况，如果未明确计算 KV cache (KV 缓存) 成本并建议改用 GQA 或 MLA，则拒绝推荐 `n_heads > 16` 的 MHA。除非用户明确在进行 benchmark (基准测试)，否则拒绝为 1B 参数以下的模型建议 MLA。
