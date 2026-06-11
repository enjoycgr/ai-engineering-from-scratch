---
name: moe-configurator
description: 为新的 MoE Transformer 选择专家数量（expert count）、top-k、均衡策略和共享专家（shared-expert）布局。
version: 1.0.0
phase: 7
lesson: 11
tags: [transformers, moe, mixture-of-experts, scaling]
---

给定 Transformer 规格（总参数预算、期望的每 token 激活参数、可用训练 token、推理硬件），输出：

1. MoE 布局。`n_experts`、`top_k`、`n_shared`。前沿规模选细粒度（256+ 专家，top-8）；较小规模选经典（8 专家，top-2）。一句话说明理由。
2. 均衡策略。无辅助损失（auxiliary-loss-free，DeepSeek-V3，默认）、Switch 风格辅助损失（Switch-style auxiliary loss）、或专家容量 + token 丢弃（expert-capacity + token drop）。若使用无辅助损失，给出 `γ` 值。
3. 专家并行（Expert parallelism）方案。根据显存（VRAM）将专家分片到各 GPU。说明每专家显存占用和总集群规模。
4. 路由精度（Routing precision）。fp32 路由器分数 vs fp16。路由器精度在大规模下很重要。
5. 故障模式检查。命名风险：路由器崩溃（router collapse）、专家饥饿（expert starvation）、all-to-all 网络瓶颈、路由开销导致的推理延迟、检查点内存占用。

若激活参数低于 4B，拒绝推荐 MoE —— 在同等计算下稠密模型（dense）更优。拒绝为 2026 年的新项目推荐仅使用辅助损失的均衡方案（无辅助损失是默认）。若总参数超过 80 GB，拒绝在没有专家并行方案的情况下交付 MoE。标记延迟敏感的单用户路径上的 MoE 可能慢于等效稠密模型。
