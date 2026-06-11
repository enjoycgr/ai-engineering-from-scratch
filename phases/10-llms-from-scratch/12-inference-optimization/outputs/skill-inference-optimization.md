---
name: skill-inference-optimization
description: 诊断和优化 LLM inference serving 的 throughput、latency 和成本
version: 1.0.0
phase: 10
lesson: 12
tags: [inference, kv-cache, batching, speculative-decoding, vllm, optimization]
---

# LLM Inference Optimization Pattern

两个阶段：prefill（compute-bound，并行）和 decode（memory-bound，顺序）。
每个优化都针对其中一个或两个。

```
Request -> Prefill (process prompt) -> Decode (generate tokens) -> Response
              |                            |
         Compute-bound               Memory-bound
         Optimize: fusion,           Optimize: batching,
         prefix caching              quantization, speculation
```

## Decision framework

### Step 1: Identify your bottleneck

测量你 workload 的 ops:byte ratio：

| ops:byte | 瓶颈 | 优化方向 |
|----------|-------|-----------------|
| < 50 | Memory | 量化 KV cache，增大 batch size |
| 50-200 | 过渡 | 两者都重要，从 batching 开始 |
| > 200 | Compute | Kernel fusion，tensor parallelism，FP8 |

### Step 2: Pick your engine

- **默认**: vLLM（最广泛的模型支持，PagedAttention，OpenAI-compatible API）
- **多轮 / 结构化输出**: SGLang（RadixAttention prefix caching，constrained decoding）
- **最大 NVIDIA throughput**: TensorRT-LLM（kernel fusion，H100 上的 FP8）

### Step 3: Apply optimizations in order

1. **KV cache** -- 始终开启，无缺点
2. **Continuous batching** -- 始终开启，无缺点（vLLM/SGLang 默认开启）
3. **Prefix caching** -- 如果你有共享 system prompts 则开启（大多数 chatbot 都有）
4. **Quantization** -- KV cache INT8/FP8 以最小质量损失减少 2-4 倍显存
5. **Speculative decoding** -- 当 latency 比 throughput 更重要时添加
6. **Tensor parallelism** -- 当模型无法装入单卡时跨 GPU 拆分

## KV cache 显存公式

```
per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param
total = per_token * sequence_length * num_concurrent_users
```

常见模型（BF16）快速参考：

| Model | Per token | 100 users @ 4K |
|-------|-----------|----------------|
| Llama 3 8B | 32 KB | 12.5 GB |
| Llama 3 70B | 320 KB | 125 GB |
| Llama 3 405B | 504 KB | 197 GB |

## Speculative decoding 检查清单

- Draft model 应比 target 小 5-10 倍（例如 8B 为 70B 起草）
- Acceptance rate > 70% 才有有意义的加速
- 在可预测文本上效果最好（代码、结构化输出、自然语言）
- 在创造性/重采样任务上效果最差（低 temperature 有帮助）
- 对大多数 workload：EAGLE > draft-target > n-gram

## 常见错误

- 在 batch=1 下运行 decode（memory-bound，GPU 计算 95% 空闲）
- 分配 contiguous KV cache 块（使用 PagedAttention，接近零浪费）
- 当 80% 请求共享相同 system prompt 时忽略 prefix caching
- 为模型权重过度配置 GPU 显存，导致 KV cache 无空间可用
- 测量 throughput 但不测量 latency（10 秒 TTFT 的高 throughput 毫无用处）
- 在高 temperature 下使用 speculative decoding（acceptance rate 降到 50% 以下）

## Monitoring checklist

- Time to first token (TTFT): prefill 延迟，交互式使用目标 < 500ms
- Inter-token latency (ITL): decode 速度，流式传输目标 < 50ms
- Throughput (tokens/second): 所有并发用户的总和
- KV cache utilization: 已分配 cache 的使用百分比
- Batch utilization: 每次迭代填充的 batch slot 百分比
- Queue depth: 等待 batch slot 的请求数
