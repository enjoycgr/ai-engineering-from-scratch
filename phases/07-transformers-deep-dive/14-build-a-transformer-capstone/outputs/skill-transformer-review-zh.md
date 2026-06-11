---
name: transformer-review
description: 对照第 7 阶段 13 节课的内容，审查从零实现的 Transformer。
version: 1.0.0
phase: 7
lesson: 14
tags: [transformers, review, capstone]
---

给定一个从零实现的 Transformer 代码库（PyTorch / JAX），对照 2026 年默认标准进行审查，并标记缺失或错误的组件：

1. **注意力（Attention）。** 存在因果掩码（causal mask）。按 `sqrt(d_head)` 缩放。多头拆分（multi-head split）正常工作。如果可用，使用 Flash Attention。如果 d_model ≥ 1024，需提及 GQA（Grouped Query Attention，分组查询注意力）。
2. **位置编码（Positional encoding）。** RoPE（Rotary Positional Embedding，旋转位置编码，2026 年首选）或可学习的绝对位置编码（适用于小模型）。将正弦（sinusoidal）标记为历史方案。
3. **Block 结构。** 预归一化（Pre-norm），而非后归一化（post-norm）。RMSNorm（而非 LayerNorm）。SwiGLU FFN（而非 ReLU/GELU）。每个子层（sublayer）都有残差连接（residual）。线性层中舍弃偏置（biases，现代默认）。
4. **训练。** AdamW（或 2026+ 的 Muon），带线性 warmup 的余弦学习率（cosine LR schedule）调度，梯度裁剪（gradient clipping）设为 1.0，bf16 自动混合精度（autocast）。token embedding 与 lm_head 之间的权重绑定（weight tying）。
5. **损失函数。** 在每个位置进行 shift-by-one 交叉熵。如果有 padding 则将其 mask 掉。以固定间隔记录训练损失和验证损失。

如果代码库存在以下任何一项，拒绝通过：无理由使用 post-norm、在 2026 年生产代码中使用 LayerNorm 且无正当理由、解码器自注意力中缺失 causal mask、小型语言模型（LM）中未绑定 embedding。标记以下问题：没有验证集划分、没有梯度裁剪、没有 warmup 时学习率 > 1e-3，或 block_size 超过位置编码范围且无回退方案。建议端到端运行 `python code/main.py`，并检查在 nano 配置下，tinyshakespeare 上的最终验证损失是否低于 2.5。
