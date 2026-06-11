# KV Cache、Flash Attention 与推理优化

> 训练是并行且受 FLOP 限制的。推理是串行且受内存限制的。瓶颈不同，技巧也不同。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 02（自注意力机制），Phase 7 · 05（完整 Transformer），Phase 7 · 07（GPT）
**时间：** ~75 分钟

## 问题所在

一个朴素的自回归解码器在生成 `N` 个 token 时要做 `O(N²)` 的工作量：每一步都重新计算对完整前缀的注意力。对于一个 4K token 的回复，这就是 1600 万次注意力操作，其中大部分是冗余的。前缀 token 的每个隐藏状态一旦计算出来就是确定的——你只需要将新 token 的 query 与之前所有 token 缓存的 key 和 value 做注意力即可。

此外，注意力本身会移动大量数据。标准注意力会实例化一个 N×N 的分数矩阵、N×d 的 softmax 输出、N×d 的最终输出——对 HBM（High Bandwidth Memory，高带宽内存）的读写太多了。当 N≥2K 时，注意力在成为 FLOP 瓶颈之前就已经成为内存瓶颈。经典的注意力核函数对现代 GPU 的利用率只有 4–10 分之一。

两项优化（均来自 Dao 等人）将前沿推理从"慢"推到了"快"：

1. **KV cache（KV 缓存）。** 存储每个前缀 token 的 K 和 V 向量。每个新 token 的注意力就是一次 query 对缓存的 key 的查询。推理从 `O(N²)` 降低到每生成一步 `O(N)`。
2. **Flash Attention。** 对注意力计算进行分块（tile），使完整的 N×N 矩阵永远不会进入 HBM。softmax 和矩阵乘法全部在 SRAM（Static Random Access Memory，静态随机存取存储器）中完成。在 A100 上有 2–4 倍的实际速度提升；在 H100 上使用 FP8 可达 5–10 倍。

到 2026 年，这两者已经成为标配。每个生产级推理栈（vLLM、TensorRT-LLM、SGLang、llama.cpp）都依赖它们。每个前沿模型都默认启用 Flash Attention。

## 核心概念

![KV cache 增长与 Flash Attention 分块](../assets/kv-cache-flash-attn.svg)

### KV cache 数学

每个解码器层、每个 token、每个 head：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K 和 V
```

对于一个 7B 模型，32 层、32 个 head、d_head=128、fp16：

```
每层每 token = 2 * 128 * 2 = 512 字节
每 token（32 层） = 16 KB
每 32K 上下文 = 512 MB
```

对于 Llama 3 70B（80 层，d_head=128，GQA（Grouped-Query Attention，分组查询注意力）使用 8 个 KV head）：

```
每层每 token = 2 * 8 * 128 * 2 = 4096 字节（4 KB）
每 32K 上下文 = 10.4 GB
```

这 10 GB 就是为什么 Llama 3 70B 在 128K 上下文、batch size 为 1 时，仅 KV cache 就需要占用 40 GB A100 的大部分显存。

**GQA 是 KV cache 的制胜关键。** 如果使用 MHA（Multi-Head Attention，多头注意力）的 64 个 head，将是 32 GB。MLA（Multi-head Latent Attention，多头潜在注意力）还能进一步压缩。

拖动维度参数，观察 cache 大小的变化。将序列长度或 batch size 推高，看看它多快就超过单张 GPU 的容量：

```figure
kv-cache-sizer
```

### Flash Attention —— 分块技巧

标准注意力：

```
S = Q @ K^T          （HBM 读取，N×N，HBM 写入）
P = softmax(S)       （HBM 读取，HBM 写入）
O = P @ V            （HBM 读取，HBM 写入）
```

三次 HBM 往返。在 H100 上，HBM 带宽为 3 TB/s；SRAM 为 30 TB/s。每次 HBM 往返相比全部在片内计算都是 10 倍的减速。

Flash Attention：

```
对每个 Q 块（tile size ~128 × 128）：
    将 Q_tile 加载到 SRAM
    对每个 K, V 块：
        将 K_tile, V_tile 加载到 SRAM
        计算 S_tile = Q_tile @ K_tile^T     （在 SRAM 中）
        运行 softmax 聚合                   （在 SRAM 中）
        累加到 O_tile                        （在 SRAM 中）
    将 O_tile 写回 HBM
```

每个 tile 一次 HBM 往返。总内存占用从 `O(N²)` 降到 `O(N)`。反向传播从正向传播中重算一些值，而不是存储它们——又一个内存优势。

**数值技巧。** 运行 softmax 跨 tile 维护 `(max, sum)`，因此最终的归一化是精确的。不是近似——Flash Attention 计算的输出与标准注意力是 bit-identical（逐位相同）的（模 fp16 的非结合性）。

**版本演进：**

| 版本 | 年份 | 关键变化 | 参考硬件上的加速比 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 分块 SRAM 核函数 | A100 上 2× |
| Flash 2 | 2023 | 更好的并行性，因果优先排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper 异步，FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 级流水线，软件 exp2 | 推理优先（最初仅前向传播） |

Flash 4 在发布时仅支持前向传播。训练仍使用 Flash 3。Flash 4 的 GQA 和变长序列支持待定（2026 年中）。

### 投机解码（Speculative Decoding）—— 另一个延迟优化

用小模型（cheap model）提议 N 个 token。大模型并行验证这 N 个 token。如果验证接受了 k 个 token，你就用 1 次大模型前向传播换来了 k 个生成。在代码和散文上，典型的 k=3–5。

2026 年默认方案：
- **EAGLE 2 / Medusa。** 集成的 draft head，共享验证器的隐藏状态。无损质量下 2–3 倍加速。
- **带 draft 模型的投机解码。** 在消费级硬件上 2–4 倍加速。
- **Lookahead decoding。** Jacobi 迭代；无需 draft 模型。小众但免费。

### 连续批处理（Continuous Batching）

经典批处理推理：等待最慢的序列完成，然后启动新 batch。短回复提前完成时浪费 GPU。

连续批处理（首先在 Orca 中推出，现在在 vLLM、TensorRT-LLM、SGLang 中）：旧请求一完成就立即将新请求换入 batch。对于典型聊天工作负载，吞吐量提升 5–10 倍。

### PagedAttention —— 将 KV cache 作为虚拟内存

vLLM 的招牌功能。KV cache 以 16-token 块分配；页表将逻辑位置映射到物理块。允许在并行采样（beam search、并行采样）中共享 KV，热交换前缀以实现 prompt caching（提示缓存），并对内存进行碎片整理。相比朴素的连续分配，吞吐量提升 4 倍。

## 动手构建

参见 `code/main.py`。我们实现了：

1. 一个朴素的 `O(N²)` 增量解码器。
2. 一个 `O(N)` 的 KV cache 解码器。
3. 一个分块 softmax，模拟 Flash Attention 的运行最大值算法。

### 步骤 1：KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单：在每个层、每个 head 的列表中不断增长的每 token K、V 向量。

### 步骤 2：分块 softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

输出与一次性计算 `softmax(qK) V` 是 bit-identical（逐位相同）的，但任意时刻的工作集都是一个 `tile × d_head` 块，而不是完整的 `N × d_head`。

### 步骤 3：在 100-token 生成上比较朴素 vs 缓存解码

统计注意力操作次数。朴素：`O(N²)` = 5050。缓存：`O(N)` = 100。代码会打印两者。

## 实际使用

```python
# HuggingFace transformers 在 decoder-only 的 generate() 上自动启用 KV cache。
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper 使用 FA3
    torch_dtype="bfloat16",
)
# generate() 自动使用 KV cache
```

vLLM 生产环境：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的 prefix caching（前缀缓存）是 2026 年的重大胜利——相同的 system prompt、few-shot 示例或长上下文文档可以在多次调用间复用 KV。对于带有重复工具提示的 agent 工作负载，prefix caching 通常能带来 5 倍的吞吐量提升。

## 交付上线

参见 `outputs/skill-inference-optimizer.md`。该 skill 为新的推理部署选择注意力实现、KV cache 策略、量化和投机解码方案。

## 练习

1. **简单。** 运行 `code/main.py`。确认朴素和缓存解码器产生相同的输出；注意操作次数的差异。
2. **中等。** 实现 prefix caching（前缀缓存）：给定一个 prompt P 和多个补全，对 P 运行一次前向传播填充 KV cache，然后每个补全分支复用。测量与每次重新编码 P 相比的加速比。
3. **困难。** 实现一个玩具版 PagedAttention：KV cache 以固定 16-token 块分配，带空闲列表（free-list）。当一个序列完成时，将其块归还到池中。模拟 1000 次长度各异的聊天补全。与连续分配相比内存碎片情况。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| KV cache | "让解码变快的技巧" | 存储每个前缀 token 的 K 和 V；新的 query 直接对它们做注意力，无需重新计算。 |
| HBM | "GPU 主内存" | High Bandwidth Memory（高带宽内存）；H100 上 80 GB，B200 上 192 GB。带宽约 3 TB/s。 |
| SRAM | "片上内存" | 每个 SM 的快速内存，H100 上每个 SM 约 256 KB。带宽约 30 TB/s。 |
| Flash Attention | "分块注意力核函数" | 计算注意力而不将 N×N 矩阵实例化到 HBM 中。 |
| Continuous batching | "无等待批处理" | 完成的序列换出，新序列换入，无需排空整个 batch。 |
| PagedAttention | "vLLM 的招牌功能" | KV cache 以固定块分配，带页表；消除碎片。 |
| Prefix caching | "复用长提示" | 在多个请求间缓存共享前缀的 KV；对 agent 大幅削减成本。 |
| Speculative decoding | "草稿 + 验证" | 廉价 draft 模型提议 token；大模型一次验证 k 个。 |

## 延伸阅读

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) —— Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) —— Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) —— Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) —— Blackwell 5 级流水线和 software-exp2 技巧；阅读仓库 README 了解本课提到的前向-only 发布注意事项。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) —— vLLM 论文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 投机解码。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— 本课引用的集成 draft 方法 EAGLE-1/2 论文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— 与 EAGLE 一并提及的 Medusa 方法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) —— 关于 16-token 块和页表设计的权威深入解析。
