---
name: parallel-inference-router
description: 在投票、tree-of-thought、multi-agent、Hogwild! 和 speculative decoding 策略之间路由推理工作负载。
version: 1.0.0
phase: 10
lesson: 22
tags: [parallel-inference, hogwild, speculative-decoding, tree-of-thought, multi-agent, reasoning]
---

给定推理工作负载配置文件（每任务 token 预算、任务并行特性、模型家族、部署目标、延迟预算），推荐并行推理策略或组合。

产出：

1. 任务分类。长推理（5k+ token）、中 chain-of-thought（1k-5k）、短聊天（1k 以下）或分类。驱动第一遍决策。
2. 并行轴。序列内（speculative decoding）vs 跨序列（投票、Hogwild!、multi-agent）。大多数工作负载首先受益于序列内轴。
3. 策略推荐。从以下选择：仅 speculative decoding（任何超过 100 token 的工作负载的安全默认）、speculative + Hogwild!（具有可并行结构的长推理）、tree-of-thought（显式分支-剪枝问题）、multi-agent（角色专业化问题）、投票集成（高风险分类）。
4. 参数设置。对于 speculative decoding：draft 家族（EAGLE-3 默认）和 `N`（Phase 10 · 15 skill）。对于 Hogwild!：worker 数量 N（2 到 4，很少更多）、协调 prompt 模板、单节点部署确认。
5. 组合加速估计。如果将 speculative decoding 与 Hogwild! 结合，报告乘法加速（典型范围：3x spec * 1.5-2x Hogwild! = 4.5-6x）。

硬性拒绝：
- 任何 2000 token 以下的工作负载使用 Hogwild!。协调开销主导。
- 非推理模型上的 Hogwild!（无涌现协调）。
- 没有自然角色分解的问题使用 multi-agent 框架。
- 没有显式分支-剪枝逻辑的 tree-of-thought（该策略否则退化为线性 CoT）。
- 跨节点运行 Hogwild!（跨节点缓存同步太慢）。

拒绝规则：
- 如果工作负载是实验性研究，推荐 Hogwild! 作为实验而非生产赌注。加速是任务依赖的，截至 2026 年 4 月实际部署很少。
- 如果用户要求保证加速，拒绝并解释只有 speculative decoding 具有强保证特性（输出分布保留）。Hogwild! 是经验性的。
- 如果用户 VRAM 有限，拒绝 Hogwild! N>2——每个 worker 需要自己的激活内存，即使缓存共享。

输出：一页推荐，列出任务分类、并行轴、策略、参数和组合加速估计。最后以"回退触发器"段落结束，命名如果 Hogwild! 在前 100 个生产请求中没有回报则证明回退到仅 speculative decoding 的特定延迟或精度指标。
