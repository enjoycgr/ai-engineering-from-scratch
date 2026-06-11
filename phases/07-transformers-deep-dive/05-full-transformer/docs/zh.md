# 完整 Transformer —— 编码器 + 解码器

> Attention（注意力）是明星。其余一切 —— 残差连接、归一化、前馈网络、交叉注意力 —— 都是让你把它堆深的脚手架。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 7 · 02（自注意力），Phase 7 · 03（多头注意力），Phase 7 · 04（位置编码）
**时间：** 约 75 分钟

## 问题

单层注意力只是一个特征提取器，不是一个完整的模型。每层只做一次矩阵乘法，容量不足以处理语言。你需要深度 —— 而深度没有正确的管道就会崩溃。

2017 年 Vaswani 论文打包了六个设计决策，将单层注意力变成了可堆叠的块。自那以后每一个 Transformer —— 仅编码器（BERT）、仅解码器（GPT）、编码器-解码器（T5）—— 都继承了同一个骨架。到 2026 年，这些块已被优化（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架完全相同。

本课就是这个骨架。后续课程会专门化 —— 06 讲编码器，07 讲解码器，08 讲编码器-解码器。

## 概念

![编码器和解码器块内部结构](../assets/full-transformer.svg)

### 六个组成部分

1. **Embedding（嵌入）+ 位置信号。** Token → 向量。位置信息通过 RoPE（现代）或正弦波（经典）注入。
2. **Self-attention（自注意力）。** 每个位置都能看到所有其他位置。解码器中会加掩码。
3. **Feed-forward network（FFN，前馈网络）。** 逐位置的两层 MLP：`W_2 · activation(W_1 · x)`。默认扩展比率为 4×。
4. **Residual connection（残差连接）。** `x + sublayer(x)`。没有它，梯度在约 6 层之后就会消失。
5. **Layer normalization（层归一化）。** `LayerNorm` 或 `RMSNorm`（现代）。稳定残差流。
6. **Cross-attention（交叉注意力，仅解码器）。** Query（查询）来自解码器，Key（键）和 Value（值）来自编码器输出。

观察一个向量如何流经一个块：注意力在位置之间混合信息，残差将其向前传递，FFN 对其进行变换，归一化保持流的稳定。

```figure
transformer-block
```

### 编码器块（BERT、T5 编码器使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器是双向的。没有掩码。所有位置都能看到所有位置。

### 解码器块（GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个块有三个子层。中间那个 —— 交叉注意力 —— 是信息从编码器流向解码器的唯一通道。在纯解码器架构（GPT）中，交叉注意力被省略，只有掩码自注意力 + FFN。

### Pre-norm vs post-norm

原始论文：`x + sublayer(LN(x))` vs `LN(x + sublayer(x))`。Post-norm 在 2019 年左右失宠 —— 没有精心设计的 warmup 很难训练得很深。Pre-norm（子层**之前**做 `LN`）是 2026 年的默认选择：Llama、Qwen、GPT-3+、Mistral 都使用它。

### 2026 年现代化后的块

Vaswani 2017 年发布的是 LayerNorm + ReLU。现代模型栈把两者都替换掉了。生产环境里的块实际上长这样：

| 组件 | 2017 | 2026 |
|-----------|------|------|
| 归一化 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展比率 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量一致） |
| 位置编码 | 正弦绝对位置 | RoPE |
| 注意力 | 完整 MHA | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 去掉了 LayerNorm 的均值中心化（少一次减法），节省计算量，且经验上至少同样稳定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 的论文中始终比 ReLU/GELU FFN 低约 0.5 的困惑度（ppl）。

### 参数量

对于一个块，`d_model = d`，FFN 扩展比率为 `r`：

- MHA：`4 · d²`（Q、K、V、O 投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3rd²`
- 归一化层：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大致 Llama 3 8B）时，总计：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数每层 × 32 ≈ 7B`（加上嵌入层和输出头）。与公布的参数量一致。

## 构建

### 第一步：基础组件

使用第 03 课中的微型 `Matrix` 类（为独立性复制到本文件）：

- `layer_norm(x, eps=1e-5)` —— 减均值，除以标准差。
- `rms_norm(x, eps=1e-6)` —— 除以 RMS。不减均值。
- `gelu(x)` 和 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整接线见 `code/main.py`。

### 第二步：搭建 2 层编码器和 2 层解码器

把它们堆叠起来。将编码器输出传入每个解码器的交叉注意力。在输出投影前加一层最终的 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 第三步：在玩具样例上运行前向传播

将 6 个 token 的源序列和 5 个 token 的目标序列输入。验证输出形状为 `(5, vocab)`。不训练 —— 本课讲的是架构，不是损失函数。

### 第四步：替换为 RMSNorm + SwiGLU

把 LayerNorm 和 ReLU-FFN 替换为 RMSNorm 和 SwiGLU。确认形状仍然匹配。这就是 2026 年的现代化改造，只需替换一个函数。

## 使用

PyTorch/TF 参考实现：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但大多数 2026 年的生产代码都是自己写块，因为：

- Flash Attention 在注意力内部调用，不是通过 `nn.MultiheadAttention`。
- GQA / MLA 不在标准库参考实现中。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 默认配置。

HF `transformers` 有干净的参考块，你应该读一读：`modeling_llama.py` 是 2026 年仅解码器块的典范实现。大约 500 行，值得通读一遍。

**编码器 vs 解码器 vs 编码器-解码器 —— 如何选择：**

| 需求 | 选择 | 示例 |
|------|------|---------|
| 分类、嵌入、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、对话、代码、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入 → 结构化输出（翻译、摘要） | 编码器-解码器 | T5、BART、Whisper |

仅解码器在语言任务上胜出，因为它扩展最干净，同时处理理解和生成。编码器-解码器在输入有明确的"源序列"身份时仍然最佳（翻译、语音识别、结构化任务）。

## 交付

见 `outputs/skill-transformer-block-reviewer.md`。该 skill 会根据 2026 年默认配置审查新的 Transformer 块实现，并标记缺失部分（pre-norm、RoPE、RMSNorm、GQA、FFN 扩展比率）。

## 练习

1. **简单。** 计算 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 时你的 `encoder_block` 的参数量。通过实现该块并用 `sum(p.numel() for p in block.parameters())` 验证。
2. **中等。** 从 post-norm 切换到 pre-norm。初始化两者，在随机输入上测量 12 层堆叠后的激活范数。Post-norm 的激活应该爆炸；pre-norm 的应该保持有界。
3. **困难。** 在一个玩具复制任务（复制反转后的 `x`）上实现 4 层编码器-解码器。训练 100 步。报告损失。替换为 RMSNorm + SwiGLU + RoPE —— 损失会下降吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Block（块） | "One transformer layer" | 归一化 + 注意力 + 归一化 + FFN 的堆叠，包裹在残差连接中。 |
| Residual（残差） | "Skip connection" | `x + f(x)` 输出；使梯度能够流经深层堆叠。 |
| Pre-norm | "Normalize before, not after" | 现代做法：`x + sublayer(LN(x))`。无需 warmup 技巧就能训练得更深。 |
| RMSNorm | "LayerNorm without the mean" | 除以 RMS；少一个操作，经验上同样稳定。 |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`。在 LM 困惑度上击败 ReLU/GELU。 |
| Cross-attention（交叉注意力） | "How the decoder sees the encoder" | MHA，Q 来自解码器，K/V 来自编码器输出。 |
| FFN expansion（FFN 扩展比率） | "How wide the middle MLP is" | 隐藏层大小与 d_model 的比率，通常为 4（LayerNorm）或 2.6（SwiGLU）。 |
| Bias-free（无偏置） | "Drop the +b terms" | 现代模型栈省略线性层中的偏置项；困惑度略有提升，模型更小。 |

## 延伸阅读

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) —— 原始块规范。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) —— 为什么 pre-norm 在深层击败 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) —— RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) —— SwiGLU 论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) —— 2026 年仅解码器块的典范实现。
