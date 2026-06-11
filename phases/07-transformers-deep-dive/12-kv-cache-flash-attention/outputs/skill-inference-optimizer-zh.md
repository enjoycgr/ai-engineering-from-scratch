---
name: inference-optimizer
description: 为新的推理部署选择注意力实现、KV cache 策略、量化和投机解码方案。
version: 1.0.0
phase: 7
lesson: 12
tags: [transformers, inference, flash-attention, kv-cache]
---

给定一个推理部署（模型名称 + 参数、目标硬件、并发数、最大上下文长度、延迟 SLO、吞吐量目标），输出：

1. 服务栈。vLLM（默认生产环境）、SGLang（每 token 最低延迟）、TensorRT-LLM（NVIDIA 最优）、llama.cpp（边缘/CPU）、MLX（Apple 芯片）。一句话说明理由。
2. 注意力实现。Flash Attention 2（Ampere/Ada 默认）、Flash Attention 3（Hopper）、Flash Attention 4（Blackwell，仅前向传播）。指定 fallback（回退方案）。
3. KV cache。Dtype（默认 fp16，支持则 fp8）、paged vs contiguous（分页 vs 连续）、prefix caching（前缀缓存）开/关、并行采样共享 KV。
4. 量化。fp16 / bf16（默认）、int8（仅权重量化）、AWQ / GPTQ / GGUF 用于权重。仅在基准测试后使用激活量化。
5. 额外加速。投机解码（EAGLE 2 / Medusa / draft 模型）、continuous batching（连续批处理，始终开启）、chunked prefill（长提示工作负载）、如有重复提示则开启 prefix caching。

拒绝为训练部署 Flash Attention 4——它在发布时仅支持前向传播。拒绝在未对目标任务基准测试质量影响的情况下推荐 fp8 KV cache。标记任何在 32K+ 上下文没有 GQA 的 70B+ 模型为 KV cache 不可管理。要求任何带有重复 system prompt 的 agent/工具调用部署必须开启 prefix caching。
