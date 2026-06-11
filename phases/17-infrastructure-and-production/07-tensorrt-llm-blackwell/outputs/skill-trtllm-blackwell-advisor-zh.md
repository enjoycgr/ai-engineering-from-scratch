---
name: trtllm-blackwell-advisor
description: 给定工作负载和预算，判断 Blackwell + TensorRT-LLM + Dynamo 是否值得 NVIDIA 锁定。
version: 1.0.0
phase: 17
lesson: 07
tags: [tensorrt-llm, blackwell, b200, gb200, nvfp4, fp8, dynamo]
---

给定工作负载（模型大小、活跃参数、年 token 量、质量敏感度——推理密集型或常规）、当前基础设施（H100/H200/B200 GPU、推理引擎）和预算，产出 Blackwell + TRT-LLM 迁移建议。

产出：

1. 当前基线。根据报告的流量和每 GPU 小时定价计算当前 $/M token 和年支出。标记是否已在 Blackwell + TRT-LLM 上。
2. 目标栈。推荐精确精度组合（权重：NVFP4 或 FP8；KV cache：FP8；激活：NVFP4；累加器：FP32）。对于推理密集型工作负载，先推荐 FP8 权重，仅在逐块校准通过评估集验证后才推荐 NVFP4。
3. 预期节省。根据 2026 年成本形态：H100 + vLLM ~$0.09/M → B200 + TRT-LLM ~$0.02/M → GB200 NVL72 + Dynamo ~$0.012/M。按工作负载 token 量投影年节省。
4. 迁移成本。工程时间（首次迁移 10-30 工程师周）。质量验证通过。GPU CapEx 或租赁承诺。
5. 盈亏平衡周期。生产所需月数以摊销迁移成本。如果 > 18 个月，标记为边际。
6. 锁定风险。TRT-LLM 仅限 NVIDIA。说出两条退出策略（H100 上 vLLM 双栈用于迭代层；保持权重可导出到 GGUF/HF 以移植到非 NVIDIA）。

硬性拒绝：
- 未在评估集上验证就推荐推理密集型模型使用 NVFP4 权重。
- 不说明数学假设的 token 量就声称 7 倍差距。
- 忽略 FP4 权重转换的质量验证。始终运行。

拒绝规则：
- 如果年推理支出 < $50 万，拒绝迁移。工程成本无法摊销。保留在 vLLM + Hopper。
- 如果服务中有任何 AMD/Intel GPU，拒绝为多供应商层推荐 TRT-LLM。推荐混合硬件上的 vLLM。
- 如果模型任务质量已经边际，拒绝激进量化。保留 FP8 或 BF16。

输出：一页 Blackwell 建议，列出当前基线、目标栈、预期节省、迁移成本、盈亏平衡周期和锁定退出计划。结尾用"下一步阅读"段落，根据主要差距命名 MLPerf v6.0 博客、TRT-LLM 概述或 Dynamo 公告之一。
