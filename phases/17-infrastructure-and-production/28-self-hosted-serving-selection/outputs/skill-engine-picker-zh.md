---
name: engine-picker
description: 给定硬件、规模和工作负载，选择自托管 LLM 引擎（llama.cpp、Ollama、TGI、vLLM、SGLang）。将 2026 年 TGI 维护模式作为迁移触发器。
version: 1.0.0
phase: 17
lesson: 28
tags: [self-hosted, vllm, sglang, llama-cpp, ollama, tgi, trt-llm, engine-selection]
---

给定硬件（CPU / Apple Silicon / AMD / NVIDIA Hopper / NVIDIA Blackwell）、规模（单用户 / 小团队 / 生产 / 企业）和工作负载（通用聊天 / agentic / RAG / 长上下文 / 代码），产出引擎推荐。

产出：

1. 引擎。命名具体引擎。引用硬件优先、规模其次、工作负载第三的决策树。
2. 为什么不选替代方案。对每个替代引擎，说明为什么不选（TGI 维护模式、AMD 排除 TRT-LLM、Ollama 仅开发）。
3. 流水线。如果是生产环境，命名流水线模式（dev Ollama → staging llama.cpp → prod vLLM/SGLang）并确认权重格式（GGUF 或 HF）全程流通。
4. 生产叠加。在生产规模，指向 Phase 17 · 18（production-stack）、· 17（disaggregated）、· 11（缓存感知路由）进行组合。
5. TGI 迁移。如果现有引擎是 TGI，指定迁移计划和时间线 —— 不紧急但应在 6 个月内开始。
6. 硬件陷阱。指出两个硬性约束：仅 CPU → llama.cpp；AMD → 无 TRT-LLM。

硬性拒绝：
- 2026 年新项目默认选择 TGI。拒绝 —— 维护模式。
- Ollama 用于 >1 并发用户的共享生产。拒绝 —— 吞吐量差距。
- 未确认仅 NVIDIA 就推荐 TRT-LLM。拒绝 —— AMD / 非 NVIDIA 是硬性阻断。

拒绝规则：
- 如果硬件混合（部分 AMD、部分 NVIDIA），要求按集群决策引擎；不要强制单一引擎。
- 如果生产规模工作负载为"未知/通用"，默认 vLLM 并计划在 3 个月流量数据后重新评估。
- 如果团队想要"无 Blackwell 时的每 GPU 最快"且坚持仅 Hopper，确认 —— TRT-LLM 或 vLLM 均可接受。

产出：一页推荐，包含引擎、排除的替代方案、流水线、生产叠加、TGI 迁移姿态。最后附季度审查：当工作负载形状发生实质变化时重新评估引擎选择。
