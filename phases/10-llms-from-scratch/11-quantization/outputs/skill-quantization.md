---
name: skill-quantization
description: 根据硬件、质量和延迟约束为部署 LLM 选择合适的 quantization（量化）策略
version: 1.0.0
phase: 10
lesson: 11
tags: [quantization, inference, deployment, optimization, fp8, int4, int8, gptq, awq, gguf]
---

# Quantization（量化）决策框架

部署语言模型时，使用此框架选择合适的数字格式、量化方法和质量验证策略。

## 输入需求

请提供：
- **Model**（名称、参数量、原始精度）
- **Target hardware**（GPU 型号/显存、CPU、Apple Silicon、边缘设备）
- **Latency target**（tokens/second、time to first token）
- **Quality floor**（最大可接受 perplexity 增加量、benchmark delta）
- **Serving pattern**（batch size、最大上下文长度、并发用户数）

## 快速选择

| 你的场景 | 格式 | 方法 | 预期质量损失 |
|---------|------|------|------------|
| H100 GPU，最大吞吐量 | FP8 E4M3 | 原生 H100 转换 | < 0.1% |
| A100/A10，需要 2 倍吞吐量 | INT8 | LLM.int8() 或 SmoothQuant | < 0.5% |
| 单卡 24GB GPU，70B 模型 | INT4 | AWQ 或 GPTQ | 1-3% |
| MacBook / Apple Silicon | INT4 GGUF | Q4_K_M via llama.cpp | 1-2% |
| 移动端 / 边缘设备 | INT4 或 INT3 | QAT + 设备专用 | 2-5% |
| 最大压缩，允许一定损失 | INT2 | QuIP# 或 AQLM | 5-15% |
| 训练（混合精度） | BF16 + FP32 累加 | 原生框架支持 | 0% |

## 按组件选择精度

并非所有 tensor 都应该得到相同处理。

| 组件 | 安全最低精度 | 推荐精度 | 避免 |
|-----------|-------------|-------------|-------|
| FFN 权重 | INT4 | INT4 (AWQ/GPTQ) | 无 QAT 的 INT2 |
| Attention 权重 | INT4 | INT8 或 FP8 | INT2 |
| Embedding 层 | INT8 | FP16（保持原始） | INT4 |
| Output head | INT8 | FP16（保持原始） | INT4 |
| KV cache | FP8 | FP8 或 INT8 | 长上下文下的 INT4 |
| Attention logits | FP16 | FP16 或 BF16 | INT8 |
| Activations（推理） | INT8 | FP8 或 INT8 | INT4 |

## 方法对比

### GPTQ
- **适用场景：** GPU 推理，你需要 Hugging Face 兼容的模型
- **Calibration data：** 128 个 examples，每个 2048 tokens
- **时间：** 70B 模型在 A100 上 30-60 分钟
- **工具：** `auto-gptq`、`exllama`、`exllamav2`
- **优势：** 经过充分测试，Hugging Face 上有庞大的模型库
- **劣势：** 比 AWQ 应用慢，某些模型上质量略低于 AWQ

### AWQ
- **适用场景：** GPU 推理，你需要最佳每比特质量
- **Calibration data：** 128 个 examples
- **时间：** 70B 模型在 A100 上 15-30 分钟
- **工具：** `autoawq`、`vLLM`（原生支持）
- **优势：** 最佳 INT4 质量，应用快速，vLLM 集成
- **劣势：** 模型库比 GPTQ 小

### GGUF
- **适用场景：** CPU 推理、Apple Silicon、llama.cpp 生态系统
- **变体：** Q2_K、Q3_K_S/M/L、Q4_K_S/M、Q5_K_S/M、Q6_K、Q8_0、F16
- **推荐默认：** Q4_K_M（最佳质量/大小平衡）
- **工具：** `llama.cpp`、`ollama`、`LM Studio`
- **优势：** 自包含文件、混合精度、庞大生态系统
- **劣势：** 对 GPU 非最优（为 CPU/Metal 设计）

### SmoothQuant
- **适用场景：** GPU 上的 INT8，需要同时量化权重和 activation
- **核心思想：** 通过 per-channel scaling 将量化难度从 activations 迁移到权重
- **工具：** `smoothquant`、`TensorRT-LLM`
- **优势：** 实现 W8A8（权重和 activations 都用 INT8），2 倍加速
- **劣势：** 仅支持 INT8，不扩展到 INT4

## 质量验证协议

量化后，部署前进行验证：

1. **Perplexity test。** 在 WikiText-2 或你的领域语料上计算。Delta < 0.5 优秀，0.5-1.0 良好，> 2.0 有问题。

2. **Benchmark sweep。** 运行 MMLU（通用）、GSM8K（数学）、HumanEval（代码）。数学和代码对精度损失最敏感。

3. **Output comparison。** 从原始模型和量化模型各生成 100 个回答。使用 LLM-as-judge 计算 win rate。目标：量化模型在 > 90% 的 prompts 上获胜或打平。

4. **Latency measurement。** 在 batch size 1 和你的目标 batch size 下测量 tokens/second。验证加速是否值得质量代价。

5. **Long-context test。** 如果 serve 长上下文（> 4K tokens），在你的最大上下文长度下测试。KV cache quantization 误差随序列长度累积。

## 显存预算计算器

```
Weight memory (GB) = parameters (B) * bits / 8 / 1.073741824
KV cache per token (MB) = 2 * num_layers * d_model * bits / 8 / 1048576
KV cache for context (GB) = kv_per_token * max_context_length / 1024
Activation memory (GB) ~ 1-4 GB（相对恒定，取决于 batch size）
Total = weight_memory + kv_cache + activation_memory + overhead (10-20%)
```

示例：Llama 3 70B 在 INT4、32K 上下文下：
- Weights: 70B * 4 / 8 / 1.07 = 32.6 GB
- KV cache (FP16): 2 * 80 * 8192 * 16 / 8 / 1e9 * 32768 = ~40 GB
- KV cache (FP8): ~20 GB
- Total with FP8 KV: ~55 GB（可装入单张 80GB A100）

## 常见错误

| 错误 | 失败原因 | 修复 |
|---------|-------------|-----|
| 将 embedding 层量化到 INT4 | 第一层误差会放大到整个模型 | 保持 embeddings 在 FP16 或 INT8 |
| 对 INT4 使用 per-tensor scale | 一个 outlier 行会毁掉所有行的精度 | 使用 per-channel 或 per-group scale |
| 不校准 GPTQ/AWQ | 没有代表性数据时 scale factor 错误 | 使用来自你领域的 128 个 examples |
| 所有层使用相同 bit-width | 第一层/最后一层更敏感 | 混合精度：第一层/最后一层用更高 bit |
| 在超长上下文下量化 KV cache | 误差随序列长度二次累积 | KV cache 用 FP8，不用 INT4 |
| 跳过质量验证 | 某些模型量化效果差（尤其在边界处） | 始终运行 perplexity + 任务 eval |

## 部署配方

### Recipe 1: vLLM with AWQ（GPU 服务器）
```
pip install vllm autoawq
vllm serve model-awq --quantization awq --dtype half --max-model-len 8192
```

### Recipe 2: llama.cpp with GGUF（MacBook）
```
./llama-server -m model.Q4_K_M.gguf -c 4096 -ngl 99
```

### Recipe 3: TensorRT-LLM with FP8（H100）
```
trtllm-build --model_dir model --output_dir engine --dtype float16 --use_fp8
```
