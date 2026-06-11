# 生产环境量化 —— AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> 量化格式不是通用选择 —— 它是硬件、服务引擎和工作负载的函数。GGUF Q4_K_M 或 Q5_K_M 在 CPU 和边缘设备上占据主导，通过 llama.cpp 和 Ollama 交付。当需要在同一基座上使用多 LoRA 时，GPTQ 在 vLLM 内部获胜。AWQ 配合 Marlin-AWQ 内核在 7B 级别模型上实现约 741 tok/s，且在 INT4 中拥有最佳 Pass@1 —— 这是 2026 年数据中心生产的默认选择。FP8 在 Hopper、Ada 和 Blackwell 上保持中间地带 —— 接近无损且广泛支持。NVFP4 和 MXFP4（Blackwell 微缩放）是激进方案，需要逐块验证。两个陷阱会困扰团队：校准数据集必须匹配部署领域，且 KV cache 与权重量化是分开的 —— AWQ 课程中"我的模型现在只有 4 GB"忽略了在生产 batch size 下 10-30 GB 的 KV cache。

**类型:** 学习
**语言:** Python（标准库，跨格式的玩具内存和吞吐量比较）
**前置知识:** Phase 10 · 13（量化基础），Phase 17 · 04（vLLM 服务内部原理）
**时间:** 约 75 分钟

## 学习目标

- 说出 2026 年六种生产量化格式及其最佳适用场景。
- 根据硬件（CPU 与 GPU、Hopper 与 Blackwell）、引擎（vLLM、TRT-LLM、llama.cpp）和工作负载（常规聊天、推理、多 LoRA）选择格式。
- 计算所选格式节省的权重内存和未触及的 KV cache。
- 说出因校准数据集问题导致量化模型在领域流量上性能下降的陷阱。

## 问题背景

量化减少内存和 HBM 带宽，而这正是 decode 所需要的。FP16 70B 模型的权重为 140 GB。将权重量化为 INT4（AWQ 或 GPTQ），模型变为 35 GB —— 可以放入一张 H100 并留有 KV cache 空间，因为在 128 个并发序列、2k 上下文的情况下，仅 KV cache 就达 20-30 GB。

但量化并非免费。激进量化会降低质量，尤其是在推理密集型任务上。不同格式适用于不同引擎。不同硬件原生支持不同精度。2026 年的格式生态是真实存在的，你不能照搬别人的选择 —— 必须根据你的技术栈来选择。

## 核心概念

### 六种格式

| 格式 | 位数 | 最佳场景 | 引擎 |
|------|------|---------|------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、边缘设备、笔记本 | llama.cpp、Ollama |
| GPTQ | 4-8 | vLLM 上的多 LoRA | vLLM、TGI |
| AWQ | 4 | 数据中心 GPU 生产 | vLLM（Marlin-AWQ）、TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 数据中心 | vLLM、TRT-LLM、SGLang |
| MXFP4 | 4 | Blackwell 多用户 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多用户 | TRT-LLM |

### GGUF —— CPU/边缘设备默认选择

GGUF 是一种文件格式，本身不是量化方案 —— 它将 K-quant 变体（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）打包在一个容器中。Q4_K_M 和 Q5_K_M 是生产默认选择 —— 在 4-5 位下接近 BF16 质量。CPU 或边缘设备服务的最佳选择，因为 llama.cpp 是目前最快的 CPU 推理引擎。

在 vLLM 中的吞吐量损失：7B 模型约 93 tok/s —— 该格式未针对 GPU 内核优化。仅在部署目标为 CPU/边缘设备时使用 GGUF。其他情况不用。

### GPTQ —— vLLM 中的多 LoRA

GPTQ 是一种带有校准过程的后训练量化算法。Marlin 内核使其在 GPU 上运行飞快（比非 Marlin GPTQ 快 2.6 倍）。7B 模型约 712 tok/s。

独特优势：GPTQ-Int4 在 vLLM 中支持 LoRA adapter。如果你正在服务一个基座模型加 10-50 个微调变体（每个作为 LoRA），GPTQ 是你的路径。截至 2026 年初，NVFP4 尚不支持 LoRA。

### AWQ —— 数据中心 GPU 默认选择

Activation-aware Weight Quantization（激活感知权重量化）。在量化过程中保护约 1% 最重要的权重。Marlin-AWQ 内核：比朴素实现快 10.9 倍。7B 模型约 741 tok/s，在 INT4 格式中拥有最佳 Pass@1。

除非你需要多 LoRA（GPTQ）或激进的 Blackwell FP4（NVFP4），否则为新的 GPU 服务选择 AWQ。

### FP8 —— 可靠的中间选择

8 位浮点数。接近无损。广泛支持。Hopper Tensor Core 原生加速 FP8。Blackwell 继承。当质量不可妥协时（推理、医疗、代码生成），FP8 是安全的 2026 年默认选择。内存节省是 INT4 的一半，但质量风险远低于 INT4。

### MXFP4 / NVFP4 —— Blackwell 激进方案

Microscaling FP4（微缩放 FP4）。每个权重块都有自己的缩放因子。激进但能在 Blackwell Tensor Core 上硬件加速。相比 FP8，每个 token 的字节数减半 —— 这是 Phase 17 · 07 中的经济优势。

注意事项：
- 截至 2026 年初，尚不支持 LoRA。
- 在推理密集型工作负载上可见质量下降。
- 需要在你的评估集上逐模型验证。

### 校准陷阱

AWQ 和 GPTQ 需要校准数据集 —— 通常是 C4 或 WikiText。对于领域模型（代码、医疗、法律），在通用网页文本上校准会让算法对保护哪些权重做出错误决策。HumanEval 上的 Pass@1 可能下降数个百分点。

解决方法：在领域内数据上校准。通常数百个领域样本就足够了。在发布前在评估集上测试。

### KV cache 陷阱

AWQ 将权重压缩到 4 位。KV cache 是独立的，保持 FP16/FP8。对于使用 AWQ 的 70B 模型：

- 权重：约 35 GB（从 140 GB 压缩到 INT4）。
- 128 并发 × 2k 上下文的 KV cache：约 20 GB。
- 激活值：约 5 GB。
- 总计：约 60 GB —— 可放入 H100 80GB。

天真地认为"我把模型量化到 4 GB"忽略了另外 30-50 GB。要整体规划 HBM。

另外，KV cache 量化（FP8 KV 或 INT8 KV）是另一个具有自身权衡的选择 —— 它直接影响注意力精度，并非免费收益。

### AWQ INT4 对推理有危害

思维链、数学、长上下文代码生成 —— 这些明显受到激进量化的影响。AWQ INT4 在 MATH 上损失约 3-5 个百分点。对于推理密集型工作负载，发布 FP8 或 BF16；接受内存成本。

### 2026 年选择指南

- CPU/边缘设备服务：GGUF Q4_K_M。搞定。
- GPU 服务，常规聊天，无 LoRA：AWQ。
- GPU 服务，多 LoRA：带 Marlin 的 GPTQ。
- 推理工作负载：FP8。
- Blackwell 数据中心，已验证质量：NVFP4 + FP8 KV。
- 不明确：在每个候选格式上运行 1,000 样本评估。

## 动手实践

`code/main.py` 计算内存占用（权重 + KV + 激活值）和六种格式在一系列模型大小下的相对吞吐量。展示 KV cache 在何处占主导、权重视压缩在何处有价值，以及 FP8 在何处是安全选择。

## 交付成果

本课产出 `outputs/skill-quantization-picker.md`。给定硬件、模型大小、工作负载类型和质量容忍度，选择一种格式并生成校准/验证计划。

## 练习

1. 运行 `code/main.py`。对于 128 并发、2k 上下文的 70B 模型，计算每种格式的总 HBM。哪种格式能让你放入一张 H100 80GB？
2. 你有一个 7B 代码模型。选择一种格式并论证。如果你错误估计了质量容忍度，恢复路径是什么？
3. 计算为医疗领域模型校准 AWQ 所需的校准数据集大小。为什么更多数据并不总是更好？
4. 阅读 Marlin-AWQ 内核论文或发布说明。用三句话解释为什么 AWQ 在 7B 上达到 741 tok/s，而原始 GPTQ 仅约 712。
5. 何时应该将 AWQ 权重与 FP8 KV cache 组合使用，而不是将 KV 保持为 BF16？

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| GGUF | "llama.cpp 格式" | 打包 K-quant 变体的文件格式；CPU/边缘设备默认 |
| Q4_K_M | "Q4 K M" | 4 位 K-quant 中等；GGUF 生产默认 |
| GPTQ | "gee pee tee q" | 后训练 INT4 带校准；在 vLLM 中支持 LoRA |
| AWQ | "a w q" | 激活感知 INT4；Marlin 内核；INT4 中最佳 Pass@1 |
| Marlin 内核 | "快速 INT4 内核" | 用于 Hopper 上 INT4 的自定义 CUDA 内核；10 倍加速 |
| FP8 | "八位浮点" | Hopper/Ada/Blackwell 上的安全精度默认 |
| MXFP4 / NVFP4 | "微缩放四位" | Blackwell 4 位 FP，带逐块缩放因子 |
| 校准数据集 | "cal 数据" | 用于选择量化参数的输入文本；必须匹配领域 |
| KV cache 量化 | "KV INT8" | 与权重量化分开的选择；影响注意力精度 |

## 延伸阅读

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 对比基准测试。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 按格式的吞吐量数据。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 逐格式选择指南。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 支持的格式和标志。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 原始 AWQ 公式。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 原始 GPTQ 公式。
