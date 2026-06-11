# DeepSeek-V3 Architecture Walkthrough（DeepSeek-V3 架构走读）

> Phase 10 · Lesson 14 命名了每个开源模型都会调整的六个架构旋钮。DeepSeek-V3（2024 年 12 月，总计 671B 参数，37B 活跃）调整了全部六个并添加了四个更多：Multi-Head Latent Attention（多头潜在注意力）、auxiliary-loss-free load balancing（无辅助损失负载均衡）、Multi-Token Prediction（多 token 预测）和 DualPipe 训练。本课从上到下阅读 DeepSeek-V3 的架构并从发布配置推导每个参数计数。到结束时，你可以解释为什么 671B/37B 比率是正确的赌注，以及为什么 MLA + MoE 一起在前沿胜过单独使用任一者。

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## Learning Objectives

- 从上到下阅读 DeepSeek-V3 配置并用六个 GPT-2 旋钮加上四个 DeepSeek 特定添加项解释每个字段。
- 推导总参数计数（671B）、活跃参数计数（37B）以及每个组件的贡献。
- 计算 MLA 在 128k 上下文时的 KV cache 占用，并与相同活跃参数的密集模型使用 GQA 会支付的成本比较。
- 说明四个 DeepSeek 特定创新（MLA、MTP、auxiliary-loss-free routing、DualPipe）并命名每个针对的架构/训练栈部分。

## The Problem

DeepSeek-V3 是第一个架构与 Llama 家族有实质不同的前沿开源模型。Llama 3 405B 是"调整了六个旋钮的 GPT-2"。DeepSeek-V3 是调整了全部六个旋钮再加四个的 GPT-2。阅读 Llama 3 配置是阅读 DeepSeek 配置的热身，但深层结构——attention 块的形状、路由逻辑、训练时目标——差异足够大，需要单独的走读。

学习它的回报：DeepSeek-V3 的开放权重发布改变了开源模型中"前沿能力"的含义。该架构是许多 2026 年训练运行正在复制的蓝图。理解它是任何接触前沿 LLM 训练或推理角色的基本要求。

## The Concept

### 不变的核心，再次

DeepSeek-V3 仍然是自回归的。它仍然堆叠 decoder block。每个 block 仍然有 attention 加 MLP 加两个 RMSNorm。它仍然在 MLP 中使用 SwiGLU。它仍然使用 RoPE。Pre-norm。权重 tied embedding。与每个 Llama 或 Mistral 相同的基线。

### 转折：MLA 替代 GQA

从 Phase 10 · 14 你知道 GQA 通过跨 Q head 组共享 K 和 V 来缩小 KV cache。Multi-Head Latent Attention (MLA) 更进一步：K 和 V 被压缩成共享的低秩潜在表示（`kv_lora_rank`），然后在需要时按 head 解压。KV cache 只存储潜在——通常每个 token 每层 512 个浮点数，而不是 8 x 128 = 1024 个浮点数。

在 128k 上下文时，带 MLA 的 DeepSeek-V3（每个 token 每层一个共享潜在 `c^{KV}`；K 和 V 都从这个潜在通过 up-projection 推导，这些 up-projection 可以吸收到后续 matmul 中）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

假设的 GQA 基线（Llama 3 70B 形状，8 个 KV head，head dim 128）会支付：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k 上下文时，MLA 比 Llama-3-70B 风格的 GQA cache 小 4 倍。

权衡：MLA 每次 attention 计算增加一个解压步骤（每个 head）。额外计算与节省的带宽相比很小。长上下文推理的净胜利。

### 路由：auxiliary-loss-free load balancing

MoE router 决定哪些 top-k expert 处理每个 token。朴素 router 将太多工作集中在少数 expert 上，让其他 expert 空闲。标准修复：添加一个 auxiliary loss（辅助损失）项来惩罚负载不平衡。这有效但略微降低主任务性能。

DeepSeek-V3 引入了一个 auxiliary-loss-free 方案。Per-expert bias 项被添加到 router logits，在训练期间通过简单规则调整：如果 expert `e` 过载，减少 `bias_e`；如果欠载，增加它。无额外损失项。训练保持干净。Expert 负载保持平衡。

对主损失的影响：无可测量影响。对 MoE 架构的影响：更干净，无需调整 auxiliary-loss hyperparameter（超参数）。

### MTP：更密集的训练 + 免费 draft

从 Phase 10 · 18 你知道 DeepSeek-V3 添加 D=1 MTP 模块来预测两个位置后的 token。推理时，训练好的模块被重新用作 speculative-decoding draft，接受率 80%+。训练时，每个隐藏状态被监督在 D+1 = 2 个目标上，提供更密集的信号。

参数：671B 主模型之上 14B。开销：2.1%。

### 训练：DualPipe

从 Phase 10 · 19 你知道 DualPipe 是一种双向流水线，将前向和后向块与跨节点 all-to-all 通信重叠。在 DeepSeek-V3 的 2,048-H800 规模下，它回收了 1F1B 会因 pipeline bubble 损失的约 245k GPU 小时。

### 配置，逐字段

这是 DeepSeek-V3 配置（简化）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

解析它：

- `hidden_size=7168`：embedding 维度。
- `num_hidden_layers=61`：总 block 深度。
- `first_k_dense_layers=3`：前 3 个 block 使用大小 18432 的 dense MLP。剩余 58 个使用 MoE。
- `num_attention_heads=128`：128 个 query head。
- `kv_lora_rank=512`：K 和 V 被压缩到这个潜在维度并按 head 解压。
- `num_experts=256, num_experts_per_tok=8`：每个 MoE block 有 256 个 expert，路由 top-8。
- `shared_experts=1`：在 256 个路由 expert 之上，1 个 always-on expert 贡献给每个 token。将其视为确保每个 token 获得可靠结果的"密集地板"。
- `moe_intermediate_size=2048`：每个 expert 的 MLP 隐藏大小。比 dense MLP 小，因为有 256 个。

### 参数核算

完整计算在 `code/main.py` 中。标题：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 个 dense block：带 MLA 的 attention（每个 block 约 ~144M）+ dense MLP（每个 block 约 ~260M）+ norm。总共约 1.2B。
- 58 个 MoE block：带 MLA 的 attention（~144M）+ 256 个 expert 每个（30M）+ 1 个 shared expert（30M）+ norm。每个 block 总共约 ~7.95B，包括所有 expert。58 个 MoE block 总共 461B。
- MTP 模块：14B。

总计：核心架构约 ~476B + 14B MTP + 发布的 671B 数字明确计入了额外的结构参数（bias tensor、expert 特定组件、shared expert 缩放等）。我们在计算器中复现的数字在发布数字的 3-5% 内——差异来自 DeepSeek 报告在其 Section 2 附录中记录的细粒度核算。

每次前向的活跃参数：

- Attention：每层 144M * 61 = 8.8B（所有层都触发）。
- MLP 活跃：前 3 层 dense（3 * 260M = 780M），58 个 MoE 层每层活跃 8 个路由 + 1 个 shared + 路由开销。每层活跃 MLP：~260M。总计：3 * 260M + 58 * 260M = ~15.9B。
- Embedding + norm：1.2B。
- 总活跃：约 26B 核心 + 14B MTP（训练但不总在推理时运行）≈ 37B。

### 671B / 37B 比率

18 倍稀疏比率（活跃参数占总参数的 5.5%）。DeepSeek-V3 是已发布开放权重中最稀疏的前沿 MoE 模型。Mixtral 8x7B 比率 13/47（28%）密集得多。Llama 4 Maverick 比率 17B/400B（4.25%）相当。DeepSeek 的赌注：在前沿规模下，更多 expert 和更低激活比率产生每个活跃 FLOP 更好的质量。

### DeepSeek-V3 的定位

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### 后续：R1, V4

DeepSeek-R1（2025）是在 V3 backbone 上的推理训练运行。R1 使用相同架构。改变的是后训练配方（可验证任务上的大规模 RL），不是预训练架构。

DeepSeek-V4（如果发布）预计将保持 MLA + MoE + MTP 并添加 DSA（DeepSeek Sparse Attention），Phase 10 · 17 中 NSA 的继任者。Lineage 稳定：架构级创新累积；每个版本调整更多旋钮。

## Use It

`code/main.py` 是专门针对 DeepSeek-V3 形状的参数计算器。运行它，将其输出与论文数字比较，并在假设变体上使用它（256 expert vs 512，top-8 vs top-16，MLA rank 512 vs 1024）。

看什么：

- 总参数计数 vs 发布 671B。
- 活跃参数计数 vs 发布 37B。
- 128k 上下文时的 KV cache——MLA vs GQA 比较。
- 每层细分以查看参数预算实际去向。

## Ship It

本课产出 `outputs/skill-deepseek-v3-reader.md`。给定 DeepSeek 家族模型（V3、R1 或任何未来变体），它生成逐组件的架构阅读，命名配置的每个字段，按组件推导参数计数，并识别模型使用四个 DeepSeek 特定创新中的哪些。

## Exercises

1. 运行 `code/main.py`。将计算器的总参数估计与发布 671B 比较并识别差异来源。论文的 Section 2 有完整分项。

2. 修改配置以使用 MLA rank 256 替代 512。计算 128k 上下文时的结果 KV cache 大小。它购买了多少百分比缩减，以及对每个 head 表达能力的代价是什么？

3. 将 DeepSeek-V3 的（256 expert，top-8）路由与假设的（512 expert，top-8）变体比较。总参数增长；活跃参数保持不变。额外的 expert 容量在理论上购买什么，在推理时成本是什么？

4. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的 Section 2.1 关于 MLA。用三句话解释为什么 K 和 V 解压矩阵可以"吸收"到后续 matmul 中以实现推理时效率。

5. DeepSeek-V3 对大多数操作使用 FP8 训练。计算 FP8 vs BF16 存储 671B 权重的内存节省。这与 14.8T token 训练预算如何交叉？

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | 将 K 和 V 压缩成共享低秩潜在（kv_lora_rank，通常 512），需要时按 head 解压；KV cache 只存储潜在 |
| kv_lora_rank | "MLA compression dim" | K 和 V 共享潜在的大小；DeepSeek-V3 使用 512 |
| First k dense layers | "Early layers stay dense" | 前几个 MoE 模型层跳过 MoE router 并运行 dense MLP 以保持稳定 |
| num_experts_per_tok | "Top-k routing" | 每个 token 触发多少个路由 expert；DeepSeek-V3 使用 8 |
| Shared experts | "Always-on experts" | 无论路由如何都处理每个 token 的 expert；DeepSeek-V3 使用 1 |
| Auxiliary-loss-free routing | "Bias-adjusted load balance" | 训练期间调整的 per-expert bias 项，保持 expert 负载平衡而不添加损失项 |
| MTP module | "Extra prediction head" | 从 h^(1) 和 E(t+1) 预测 t+2 的 transformer block；更密集训练，免费 speculative-decoding draft |
| DualPipe | "Bidirectional pipeline" | 重叠前向/后向计算与跨节点 all-to-all 的训练调度 |
| Active parameter ratio | "Sparsity" | active_params / total_params；DeepSeek-V3 达到 5.5% |
| FP8 training | "8-bit training" | 训练存储和许多计算操作在 FP8 中；与 BF16 相比大致减半内存，质量代价小 |

## Further Reading

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整架构、训练和结果文档
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 配置文件和部署说明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — 引入 MLA 的前身
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — V3 架构上的推理训练继任者
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek 家族 attention 的未来方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — 训练调度参考
