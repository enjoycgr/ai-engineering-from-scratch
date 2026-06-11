---
name: deepseek-v3-reader
description: 读取 DeepSeek 家族配置并生成逐组件架构分析。
version: 1.0.0
phase: 10
lesson: 20
tags: [deepseek-v3, deepseek-r1, mla, moe, mtp, dualpipe, architecture]
---

给定 DeepSeek 家族模型（V3、R1 或任何衍生）及其配置（hidden_size、layers、num_experts、kv_lora_rank 等），生成按组件分解模型并识别其使用哪些 DeepSeek 特定创新的架构分析。

产出：

1. 逐字段配置读取。对于每个字段，命名它映射到的组件及其贡献的参数计数。格式：`field_name: value → interpretation → parameter contribution`。
2. 参数分解。总参数、活跃参数、活跃比率。按 embedding、每层 attention、每层 MLP（dense vs expert）、router、MTP 模块、LM head、RMSNorm 总计拆分。
3. 目标上下文时的 KV cache。报告 BF16 和 FP8 值。包括与相同上下文和隐藏大小下 Llama-3 风格 GQA(8/128) 基线的比较。
4. 创新检查清单。对于 MLA、MTP、aux-loss-free routing、DualPipe 中的每一项，识别模型是否使用它以及在配置/论文的哪里可见。
5. 合理性检查。计算模型在特定部署目标（H100 80GB、H200 141GB、MI300X 192GB、单节点 vs 多节点）上的推理内存预算（权重 + KV cache + 激活）。报告是否容纳以及需要什么量化。

硬性拒绝：
- 任何将 DeepSeek-V3 与 GPT 类密集模型混为一谈的分析。架构有本质不同。
- 声称 MLA 比 GQA 快而不指定上下文长度。短上下文（4k 以下）下它们相当；MLA 在长上下文获胜。
- 将 MTP 解释为 speculative decoding 的替代品。它是也兼作 draft 的预训练目标。

拒绝规则：
- 如果提供的配置缺少 `kv_lora_rank`、`num_experts` 或 `first_k_dense_layers`，拒绝——这不是 DeepSeek 家族模型。
- 如果用户要求精确匹配发布参数计数（精确到最近 100M），拒绝并解释发布数字包含简化计算器无法精确复现的实现特定结构参数。引导他们到论文的 Section 2 附录。
- 如果目标部署目标是消费级 GPU（24GB 或更少），拒绝并推荐量化蒸馏的 DeepSeek 家族衍生模型替代。

输出：一页架构分析，列出字段、参数分解、KV cache、创新检查清单和部署适配。最后以"接下来读什么"段落结束，命名 NSA（Phase 10 · 17）、V2 论文的 MLA 消融或 V3 技术报告的 Section 2 附录之一，取决于分析浮现了什么问题。
