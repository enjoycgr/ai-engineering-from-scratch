# Show-o 与离散 Diffusion 统一模型

> Transfusion 混合连续和离散表示。Show-o (Xie 等人, 2024 年 8 月) 走另一条路：文本 token 使用因果 next-token prediction，图像 token 使用 MaskGIT 精神的 masked discrete diffusion。两者在一个 transformer 内用混合 attention mask。结果在单一主干、每模态一个 tokenizer、单一 loss 公式（next-token 扩展到 masked prediction）上统一 VQA、text-to-image、inpainting 和混合模态生成。本课走过 Show-o 设计——为什么 masked discrete diffusion 是并行、少步图像生成器——并与 Transfusion 和 Emu3 对比。

**类型：** Learn
**语言：** Python (stdlib, masked-discrete-diffusion sampler)
**前置知识：** Phase 12 · 13 (Transfusion)
**时间：** ~120 分钟

## 学习目标

- 解释 masked discrete diffusion：均匀 mask token 然后让 transformer 恢复它们的 schedule。
- 在速度和质量上对比并行图像解码（Show-o、MaskGIT）与自回归图像解码（Chameleon、Emu3）。
- 说出 Show-o 在一个 checkpoint 中处理的三个任务：T2I、VQA、图像 inpainting。
- 挑选 masking schedule（cosine、linear、truncated）并推理其对样本质量的影响。

## 问题

Transfusion 的两 loss 训练有效但有更棘手的动态——连续 diffusion loss 与离散 NTP loss 生活在不同数值尺度上。平衡 loss 权重是超参数搜索。架构有效但复杂。

Show-o 的答案：保持两种模态离散（像 Chameleon），但通过 masked discrete diffusion 并行生成图像而非顺序生成。训练目标变成单一 masked-token-prediction，自然泛化 next-token-prediction。

## 概念

### Masked discrete diffusion (MaskGIT)

原始 Chang 等人 (2022) 的 MaskGIT 技巧优雅。从完全 mask 的图像开始（每个 token 是特殊 `<MASK>` id）。每步并行预测所有 mask token，然后保留最自信的 top-K 预测并重新 mask 其余。~8-16 次迭代后，所有 token 填满。每步 unmask 多少 token 的 schedule 可调——cosine schedule 效果好。

训练简单：从 [0, 1] 均匀采样 masking ratio，应用到图像 VQ token，训练 transformer 恢复被 mask 的。正是 BERT 对文本做的，扩展到图像生成。

### Show-o：一个 transformer，混合 mask

Show-o 把 MaskGIT 放入因果语言模型 transformer。Attention mask 是：

- 文本 token：因果（标准 LLM）。
- 图像 token：图像块内全双向（mask token 预测时可看到每个其他图像 token）。
- 文本到图像：文本关注前面图像，图像关注前面文本。

训练在以下之间交替：
1. 文本序列的标准 NTP。
2. T2I 样本：文本 → 带 mask 图像 token 的图像，masked-token-prediction loss。
3. VQA 样本：图像 → 文本（实际是 NTP）。

统一 loss 是 `<MASK>` token 上的 cross-entropy，覆盖文本 NTP（只有最后 token 是"masked"）和图像 masked-diffusion（随机子集被 mask）。

### 并行采样

Show-o 在 ~16 步生成图像，而非 ~1000（每 token 自回归）或 ~20（diffusion）。每步并行预测所有 mask token；提交 top-K 自信；重复。

对比：
- Chameleon / Emu3（token 上自回归）：N_tokens 次 forward pass，典型每图像 1024-4096。
- Transfusion（连续 diffusion）：~20 步，每步完整 transformer pass。
- Show-o（masked discrete diffusion）：~16 步，每步完整 transformer pass。

Show-o 在类似规模模型上比 Chameleon 更快，大致匹配 Transfusion 步数但每步成本更低（离散 vocab logits vs 连续 MSE loss）。

### 一个 checkpoint 中的任务

Show-o 推理支持四个任务，由 prompt 格式选择：

- 文本生成：标准自回归文本输出。
- VQA：图像进，文本出。
- T2I：文本进，通过 masked discrete diffusion 图像出。
- Inpainting：带部分 token mask 的图像，填充。

Inpainting 能力来自 masked-prediction 训练免费获得。Mask VQ-token 网格区域，喂入其余加文本 prompt，预测 mask token。

### Masking schedule

每步 unmask 多少 token 的 schedule 塑造质量。Show-o 推荐 cosine：

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

第 0 步，所有 token mask（ratio 1.0）。第 T 步，无 mask。Cosine 将质量集中在 prediction 最丰富的 mid-range ratio。Linear schedule 也有效但更快平台期。

### Show-o2

Show-o2 (2025 后续, arXiv 2506.15564) 扩展 Show-o：更大 LLM 基础、更好 tokenizer、改进 mask schedule。相同架构模式。

### Show-o 的位置

在 2026 年分类中：

- 离散 token + NTP：Chameleon、Emu3。推理简单但慢。
- 离散 token + masked diffusion：Show-o、MaskGIT、LlamaGen、Muse。并行采样，仍受 tokenizer 有损。
- 连续 + diffusion：Transfusion、MMDiT、DiT。最高质量，训练更复杂。
- VLM 中连续 + flow matching：JanusFlow、InternVL-U。最新。

按任务挑选：需要 T2I + inpainting + VQA 在一个开放模型中且速度合理时选 Show-o；质量至上且能负担两 loss 管道时选 Transfusion。

## 使用它

`code/main.py` 模拟 Show-o 采样：

- 16 个 VQ token 的玩具网格。
- 基于 prompt 和当前 unmask token 预测 logits 的 mock "transformer"。
- Cosine schedule 上 8 步并行 mask 采样。
- 打印中间状态（mask 模式演化）和最终 token。

运行它，逐步观察 mask 溶解。

## 交付它

本课产生 `outputs/skill-unified-gen-model-picker.md`。给定需要理解（VQA、标题生成）和生成（T2I、inpainting）且受开放权重约束的产品，在 Show-o 家族、Transfusion/MMDiT 家族和 Emu3 / Chameleon 家族之间挑选，附具体权衡。

## 练习

1. Masked discrete diffusion 在 ~16 步采样。为什么不是 1？第 0 步 unmask 一切会崩溃什么？

2. Inpainting 随 masked diffusion 免费获得。提出 Show-o inpainting 击败专家模型的产品用例（真实或假设）。

3. Cosine schedule vs linear schedule：追踪 T=8 时每步 unmask token 数。哪个更平衡？

4. 512x512 Show-o 图像是 1024 token。词汇 K=16384 时，模型发出 1024 * log2(16384) = 14,336 位（~1.75 KiB）数据。Stable Diffusion 输出 512*512*24 位 = 6,291,456 位（~768 KiB）原始像素。压缩比是多少，买到什么质量？

5. 阅读 LlamaGen (arXiv:2406.06525)。LlamaGen 的类条件自回归图像模型与 Show-o 的 masked 方法有何不同？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Masked discrete diffusion | "MaskGIT-style" | 训练预测 mask token；推理时迭代 unmask 最自信预测 |
| Cosine schedule | "Unmask schedule" | 推理步上 mask ratio 衰减；将质量增长集中在 mid-range |
| Parallel decoding | "All tokens at once" | 每步一次 forward pass 预测完整 mask token 序列，然后提交 top-K |
| Hybrid attention | "Causal + bidirectional" | 跨文本 token 因果、图像块内双向的 mask |
| Inpainting | "Fill-in generation" | 条件于带部分 token mask 的图像，预测缺失；从训练目标免费获得 |
| Commitment rate | "Top-K per step" | 每迭代声明"完成"的 token 数；控制推理 vs 质量权衡 |

## 延伸阅读

- [Xie 等人 — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang 等人 — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun 等人 — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang 等人 — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
