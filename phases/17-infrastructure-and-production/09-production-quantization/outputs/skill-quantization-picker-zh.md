---
name: quantization-picker
description: 根据硬件、引擎、工作负载和质量容忍度选择 2026 年量化格式，并生成校准 + 验证计划。
version: 1.0.0
phase: 17
lesson: 09
tags: [quantization, awq, gptq, gguf, fp8, nvfp4, calibration]
---

给定硬件（CPU / H100 / H200 / B200 / GB200，含数量）、引擎（llama.cpp / vLLM / TRT-LLM / SGLang）、模型（大小 + 任务类型 —— 常规聊天 / 推理 / 代码 / 多 LoRA）和质量容忍度（可接受 HumanEval / MATH / MMLU 上 N 个百分点的下降），选择一种量化格式并生成验证计划。

产出：

1. 格式推荐。以下之一：GGUF Q4_K_M、GGUF Q5_K_M、GPTQ-Int4 + Marlin、AWQ-Int4 + Marlin、FP8、NVFP4 + FP8 KV，或叠加组合。通过决策树论证：CPU → GGUF；推理 → FP8；vLLM 多 LoRA → GPTQ；常规 GPU 聊天 → AWQ；Blackwell 已验证 → NVFP4。
2. 内存预算。报告权重 + KV cache（按报告的并发数 × 上下文）+ 激活值。确认是否适合目标 GPU，或指出需要多 GPU。
3. 校准计划。数据集来源（AWQ/GPTQ 需要领域匹配；通用 C4/WikiText 作为最后手段）。样本数（领域 500-2000）。验证集（从校准池中留出 10%）。
4. 验证计划。与任务匹配的评估集：代码用 HumanEval，推理用 MATH/MMLU，聊天用 MT-Bench。BF16 基线 vs 量化。如果下降 ≤ 质量容忍度则发布。
5. KV cache 决策。与权重量化分开。推理推荐 FP8 KV；注意力精度边缘时推荐 BF16 KV；仅在验证后推荐 INT8 KV。
6. 回滚路径。在磁盘上保留 BF16/FP8 权重；如果生产质量下降则标记切换回去。

硬性拒绝：
- 在推理密集型工作负载上推荐 NVFP4 权重而不经过评估集验证。
- 为领域模型在通用网页数据上校准。始终使用领域内数据。
- 在 HBM 预算中忘记 KV cache。始终逐项列出。
- 声称吞吐量数字而不指明内核（Marlin-AWQ 与纯 AWQ 相差 10 倍）。

拒绝规则：
- 如果工作负载本质上是质量边缘的（开放式创意生成、边缘情况推理），拒绝激进 INT4。保持 FP8 或 BF16。
- 如果引擎是 llama.cpp，拒绝 GGUF 以外的任何格式。格式与引擎匹配是基本要求。
- 如果用户无法运行 1,000 样本评估，拒绝。生产中不能盲目量化。

输出：一页量化选择，列出所选格式、HBM 预算、校准计划、验证计划、KV cache 决策和回滚路径。结尾附一段"接下来测量什么"，根据关键风险选择评估集差异、峰值并发下的 KV cache 压力或真实 batch size 下的吞吐量中的一个。
