---
name: attention-variant-picker
description: 根据上下文长度、检索需求和计算配置，为新模型选择 full / sliding-window / sparse / differential 注意力拓扑。
version: 1.0.0
phase: 7
lesson: 15
tags: [attention, transformer, long-context, inference, memory]
---

# Attention Variant Picker（注意力变体选择器）

帮助开发者为新的 transformer 选择并论证注意力拓扑，或为正在扩展到更长上下文的现有模型选择。

## Inputs to gather（需要收集的输入）

1. **Target context length（目标上下文长度）** 在训练和推理时（通常不同——许多模型在 16K 训练，在推理时扩展）。
2. **Retrieval demand（检索需求）** 按 1–5 级评分：1 = 纯聊天，5 = needle-in-haystack（大海捞针）/ RAG / 带有长仓库上下文的代码。
3. **Inference memory budget（推理内存预算）** 每次请求的 KV cache 容忍度（每层每 token 的字节数是合适的单位）。
4. **Training cost tolerance（训练成本容忍度）** —— 从头训练 SWA 很便宜；将差分注意力改造到预训练模型中很昂贵。
5. **Hardware target（硬件目标）** —— Hopper+ 有完整的 FlashAttention-3，Ada 有 FA2，更老的 GPU 受 mask 限制。

## Decision rules（决策规则）

- **Context ≤ 16K and retrieval ≤ 3**：使用 FlashAttention 的完整注意力。不要过早优化。
- **Context 16–128K and retrieval ≤ 3**：5:1 的混合 SWA + 全局注意力，窗口 1024（Gemma 3 形状）。在保持检索能力的同时压缩 KV。
- **Context > 128K**：完整 SWA，每 4–6 层加一个全局层，加上位置插值 / YaRN 缩放（第 04 课）。
- **Retrieval = 5 and training budget allows**：仅在前 4 层考虑差分注意力（KV 翻倍的一半代价，获得大部分 sink 消除收益）。
- **You're shipping a public API**：优先选择稳定的模式（full、SWA、Gemma-3 混合）。除非你有 kernel 工程师，否则跳过 native-sparse / DIFF。
- **You can't change the base model**：SWA 可以通过 masking 在推理时改造；差分和稀疏注意力不能。

## Always flag（始终标记）

- 7B 以下的纯 SWA 模型通常在推理基准测试上明显退步。不推荐。
- 窗口大小 < 512 几乎永远不对。要么加大，要么使用不同的拓扑。
- 差分注意力的论文报告基于小模型（3–7B）。截至 2026 年初，规模扩大的证据很少。
- 每种变体都与 RoPE / YaRN 缩放（第 04 课）相互作用。明确说明位置编码方案。

## Output format（输出格式）

返回：

1. **Recommendation（推荐）** —— 一个单一的命名拓扑（例如 "Gemma-3 mix, W=1024, 5:1 SWA:global"）。
2. **Justification（论证）** —— 将每个输入映射到上面的决策规则。
3. **KV cache estimate（KV cache 估算）** —— 在目标上下文下，以每层每 token 的字节数和 batch 1 时的 GB 数表示。
4. **Migration path（迁移路径）** —— 如果基础模型已经训练好，如何改造。
5. **Known risks（已知风险）** —— 哪些基准测试 / 工作负载可能会退步。
