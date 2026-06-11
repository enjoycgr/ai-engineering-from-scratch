---
name: prompt-gpt-architecture-analyzer
description: 分析任何 GPT 风格 transformer 模型中的架构选择
version: 1.0.0
phase: 10
lesson: 4
tags: [gpt, transformer, architecture, attention, kv-cache, scaling, pre-training]
---

# GPT Architecture Analyzer（GPT 架构分析器）

从技术报告、模型卡或训练日志评估 GPT 风格模型时，使用此框架分解架构并识别设计权衡。

## Analysis Protocol（分析协议）

### 1. Parameter Allocation Breakdown（参数分配分解）

计算每个组件的精确参数数量：

- **Token embeddings**：vocab_size x embed_dim
- **Position embeddings**：max_seq_len x embed_dim
- **Per-block attention**：4 x embed_dim x embed_dim（Q、K、V、output projections）
- **Per-block FFN**：2 x embed_dim x ff_dim + embed_dim + ff_dim（两个线性层 + 偏置）
- **Per-block LayerNorm**：4 x embed_dim（两个 norm，每个有 scale + bias）
- **Final LayerNorm**：2 x embed_dim
- **Output head**：vocab_size x embed_dim（如果与 token embeddings 做 weight tying 则为 0）

标记是否有任何单个组件超过总参数的 40%。Embedding matrix 在小模型中占主导。Attention 和 FFN 在大模型中占主导。

### 2. Attention Design Analysis（注意力设计分析）

评估 attention 配置：

- **Head dimension**：embed_dim / num_heads。标准是 64（GPT-2）或 128（Llama 3）。低于 32 限制每头的表达能力。高于 128 浪费算力且收益甚微。
- **Heads per layer**：更多 heads = 更多样化的 attention 模式，但 KV cache 内存更大。
- **Grouped Query Attention (GQA)**：模型是否在多个 Q heads 间共享 K/V heads？Llama 3 使用 GQA，32 个 Q heads 对应 8 个 KV heads。这将 KV cache 减少 4 倍。
- **Context length**：最大位置 embedding。RoPE 允许外推到训练长度之外。绝对位置 embedding 不行。

### 3. Memory Budget（内存预算）

在模型最大上下文长度下进行 inference：

- **Weights (FP16)**：total_params x 2 字节
- **KV Cache (FP16)**：2 x num_layers x num_kv_heads x head_dim x max_seq_len x 2 字节
- **Activations**：batch_size x seq_len x embed_dim x 2 字节 x num_layers（近似）

标记 KV cache 是否超过权重内存。这发生在长上下文模型（128K+）上，表明模型在 decode 期间受内存限制。

### 4. Compute Profile（计算画像）

- **Prefill FLOPS per token**：约 2 x total_params（每个参数一次 matmul，forward pass）
- **Decode FLOPS per token**：与 prefill 相同，但在单个 token 上
- **Prefill bottleneck**：compute-bound（GPU TFLOPS）
- **Decode bottleneck**：memory-bound（GPU memory bandwidth）
- **Arithmetic intensity**：每字节内存访问的 FLOPS。低于 100 = 内存受限。

### 5. Scaling Decisions（扩展决策）

根据已知 scaling laws（扩展定律）评估：

- **Chinchilla optimal**：对于给定算力预算 C，最优模型大小 N 和 token 数 D 满足 N ~ D（大致同等扩展）。7B 模型需要 ~140B token。
- **Llama 3 overtrained**：Meta 在 15T token 上训练 Llama 3 8B（100 倍 Chinchilla 最优）。在小模型上更多数据过训练能产生更好的每 token inference 成本。
- **Width vs depth**：对于相同参数数量，更深的模型（更多层）通常比更宽的模型（更大 embed_dim）更具样本效率。

## Red Flags（危险信号）

- **FFN ratio 不是 4x**：标准是 ff_dim = 4 x embed_dim。Llama 使用 SwiGLU 的 8/3 x embed_dim。偏差应该有正当理由。
- **没有 weight tying**：除非 vocab_size 相对于 embed_dim 非常大，否则 output head 应与 token embeddings 共享权重。
- **13B 以上没有 GQA**：没有 grouped-query attention 的 13B 以上模型会有过大的 KV cache。
- **长上下文没有 RoPE**：绝对位置 embedding 不能外推到训练长度之外。目标 32K+ 上下文的模型应使用 rotary embeddings。
- **学习率对模型大小来说太高**：更大的模型需要更低的峰值学习率。GPT-2 Small 使用 6e-4。Llama 3 405B 使用 8e-5。

## Output Format（输出格式）

1. **Parameter Table**：按组件统计参数数量及百分比
2. **Memory Budget**：最大上下文长度下的权重、KV cache 和激活内存
3. **Compute Profile**：A100/H100 的 prefill 和 decode 吞吐量估计
4. **Design Assessment**：模型做对了什么以及什么是非标准的
5. **Scaling Verdict**：模型大小是否适合其训练数据
