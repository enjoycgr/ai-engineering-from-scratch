---
name: speculative-tuning
description: 分析解码工作负载并选择 draft model、draft 长度 K、温度门控和回退策略。
version: 1.0.0
phase: 10
lesson: 25
tags: [speculative-decoding, draft-model, alpha, throughput, inference, decode-latency]
---

给定目标模型（规模、家族、tokenizer）、工作负载遥测（任务混合、prompt-vs-decode token 比率、p50/p99 解码延迟、加速器与 HBM 余量、平均 batch size、采样温度分布）和可用 draft 检查点，输出：

1. Draft 选择。从同家族小型（Llama-70B 用 Llama-3.2-1B）、蒸馏 draft（Qwen3-0.6B-spec）、安装在目标上的 Medusa head 中选择；如果没有 draft 的 FLOP 成本比率低于 30%，则选择"无 spec decode"。逐字节确认与目标的 tokenizer 匹配；拒绝 tokenizer 不匹配的 draft。
2. Draft 长度 K。以 E[tokens] / (1 + K x c) 取 argmax，其中 c 是 draft-to-target 成本比率。使用在 5_000 token 同分布数据上校准运行测得的 alpha，展示 K 为 2、3、4、5、6 时的计算过程。聊天默认 K=4，代码默认 K=6，高温创意写作默认 K=2。
3. 温度门控。设置 spec decode 禁用的温度阈值。默认 0.8；如果校准显示 alpha 更早崩溃则降至 0.6。拒绝任何依赖增加超过 50 微秒每请求检查的温度门控。
4. 树预算。如果服务栈支持树形 drafting，batch 低于 8 时选择小型固定树（深度 2，分支 3-2）；batch 超过 32 时选择平链。以字节说明验证器的 KV scratch 大小并确认它可容纳在 HBM 余量中。
5. 回退策略。命名指标（最近 1_000 次验证的滑动窗口测量 alpha）和阈值（alpha 低于 0.4），服务器在该阈值时为该请求流回退到朴素自回归解码。包括回退决策的每请求生命周期。

在 batch size 超过验证器计算受限点时拒绝 spec decode。超过该点，speculator 旨在利用的未使用 FLOP 不再存在；吞吐量下降。对任何测量 alpha 低于 0.4 的任务家族拒绝 spec decode；draft 开销主导且 wall-clock 延迟变差。拒绝未在 1_000 token 留出样本上针对目标验证的 draft：未验证的 draft 是静默的 KL 漂移。

示例输入："Llama-3.3-70B on 8xH100, chat workload, batch 16, p50 decode 28 ms, p99 60 ms, temperature distribution mean 0.4 / max 1.2, calibration shows alpha 0.78 on chat, 0.61 on code."

示例输出：
- Draft: Llama-3.2-1B-Instruct-spec. Same tokenizer, same family, ratio c approx 0.03.
- K: 4. E[tokens/verify] = 3.4 chat, 2.5 code. K=5 gains 0.1 token chat and pays 0.03 extra c; reject.
- Temperature gate: 0.8. Above 0.8 alpha drops below 0.45 on the calibration set.
- Tree budget: depth 2 branch (3, 2). KV scratch 480 MB at batch 16 fits.
- Fallback: sliding-window alpha over last 1_000 verifies under 0.40 disables spec decode for that stream for 30 s, then probes again.
