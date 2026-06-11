# LLaVA 与视觉指令微调

> LLaVA (2023 年 4 月) 是地球上被复制最多的多模态架构。它用 2 层 MLP 取代 BLIP-2 的 Q-Former，用朴素 token 拼接取代 Flamingo 的门控 cross-attention，并在仅由 GPT-4 从纯文本标题生成的 158k 视觉指令轮次上训练。任何在 2023 到 2026 年间构建 VLM 的从业者都构建了某种 LLaVA 变体。LLaVA-1.5 添加了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 在一个配方中统一了图像、多图像和视频。本课阅读配方，实现 projector，并解释"为什么更简单赢了。"

**类型：** Build
**语言：** Python (stdlib, projector + instruction-template builder)
**前置知识：** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**时间：** ~180 分钟

## 学习目标

- 构建一个 2 层 MLP projector，把 ViT patch embedding (dim 1024) 映射到 LLM 的 embedding dim (dim 4096)。
- 走过 LLaVA 两阶段配方：(1) 558k 标题对的 projector 对齐，(2) 158k GPT-4 生成轮次的视觉指令微调。
- 构造一个带图像 token 占位符、system prompt 和 user/assistant 轮次的 LLaVA 格式 prompt。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上获胜。

## 问题

BLIP-2 的 Q-Former（Lesson 12.03）把图像压缩到 32 个 token。干净、高效、基准测试好。但它有两个问题。

第一，Q-Former 是可训练的，但它的 loss 不是最终任务。阶段 1 训练 ITC+ITM+ITG。阶段 2 训练 LM loss。Query 学习某种中间表示，然后 LLM 必须解码。瓶颈中丢失了信息。

第二，Q-Former 占 188M 参数，在 LLaVA 2023 年的规模下，你必须与目标 LLM 共同设计它。换 LLM，重训 Q-Former。换视觉编码器，重训。每种组合都是一个独立的 R&D 项目。

LLaVA 的答案简单得令人尴尬：取 ViT 的 576 个 patch token，每个通过一个 2 层 MLP（`1024 → 4096 → 4096`），把所有 576 个倒进 LLM 的输入序列。没有瓶颈。没有阶段 1 的奇怪目标预训练。只在直接 LM loss 上训练 MLP。

数据从哪里来？LLaVA 的第二个洞察：用 GPT-4（纯文本）生成指令数据。给 GPT-4 一张图像的 COCO 标题和边界框数据，让它生成对话、描述和复杂推理问题。158k 指令-响应对，免费。无人工标注。

结果：一个在 8 A100 上跑一天的 VLM，在 MMMU 上击败 Flamingo，并发布了一个社区可以扩展的开源 checkpoint。到 2023 年底，它已经催生了 50+ 分支。

## 概念

### 架构

LLaVA-1.5 at 13B：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段 1 冻结，阶段 2 可选解冻）。
- Projector：2 层 MLP，GELU 激活，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来 Llama-3.1-8B）。

图像 + 文本 prompt 的 forward pass：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占据 LLM 上下文的 576 个 token。在 2048 上下文下，留给文本 1472 个 token。在 32k 上下文下，这是舍入误差。

### 阶段 1：projector 对齐

冻结 ViT。冻结 LLM。只训练 2 层 MLP。数据集：558k 图像-标题对（LAION-CC-SBU）。Loss：以投影图像 token 为条件的标题语言建模。

Batch 128 下单个 epoch，几小时完成。Projector 学习把 ViT 空间映射到 LLM 空间。无任务特定监督。

### 阶段 2：视觉指令微调

解冻 projector（仍然可训练）。解冻 LLM（通常完全解冻，有时 LoRA）。在 158k 视觉指令轮次上训练。

指令数据是诀窍。Liu 等人通过以下方式生成：
1. 取一张 COCO 图像。
2. 提取文本描述（5 个人类标题 + 边界框列表）。
3. 发给 GPT-4，三个 prompt 模板：
   - 对话："Generate a back-and-forth dialogue between a user and assistant about this image."
   - 详细描述："Give a rich, detailed description of the image."
   - 复杂推理："Ask a question that requires reasoning about the image, then answer it."
4. 解析 GPT-4 的输出为（instruction, response）对。

这一切都不直接触碰图像——只有文本描述。GPT-4 产生看似合理的图像内容幻觉。有些噪声，但它有效：158k 轮次足以解锁对话。

### 为什么社区复制这个

- 没有阶段 1 特有的 loss 需要调。全程 LM loss。
- Projector 几小时训练，不是几天。
- LLM 可以换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重训 projector。
- 视觉指令数据流水线使用 GPT-4，对新领域廉价可重新生成。

### LLaVA-1.5 和 LLaVA-NeXT

LLaVA-1.5 (2023 年 10 月) 添加了：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的 system prompt。
- 2048 → 32k 上下文。

LLaVA-NeXT (2024 年 1 月) 添加了：
- AnyRes：把高分辨率图像分成 2x2 或 1x3 的 336x336 裁剪网格，加一个全局低分辨率缩略图。每个裁剪变成 576 个 token；总计每张图像约 2880 个视觉 token。OCR 和图表任务大幅提升。
- 更好的指令数据混合物，含 ShareGPT4V（高质量 GPT-4V 标题）。
- 更强基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

Lesson 12.08 深度覆盖 OneVision。简短版：相同 projector，但用一个课程训练，覆盖单图像、多图像和视频在一个模型中，共享视觉 token 预算。

### 与 Q-Former 对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每张图像视觉 token | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 188M + LM | 40M + LM |
| 阶段 1 loss | ITC+ITM+ITG | 仅 LM |
| LLM 即插即用 | 需要重训 | 最小重训即可更换 |
| 多图像 | 笨拙 | 自然（拼接） |
| 视频 | 笨拙 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简单性和 token 灵活性上获胜。Q-Former 在 token 预算上获胜。到 2023 年底，token 预算不再是绑定约束（LLM 上下文增长到 32k-128k+），简单性主导。

### Prompt 格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是一个占位 token。在 tokenization 之前，它被 576 个视觉 token（AnyRes 时 2880 个）取代。Tokenizer 看到比它训练时稍长的序列，但 LLM 处理这个新颖输入，因为阶段 1 教过它了。

### 参数经济学

LLaVA-1.5-7B 分解：
- CLIP ViT-L/14 @ 336：303M（阶段 1 冻结，阶段 2 常解冻）。
- Projector（2x linear）：约 22M 可训练。
- Llama-7B：7B。
- 总计：7.3B 参数。阶段 2 可训练：完整 7B + 22M projector。

阶段 2 训练成本：约 8xA100 20 小时。这是关键数字——一天，一个节点，可复现。这就是 LLaVA 传播的原因。

## 使用它

`code/main.py` 实现：

1. 玩具规模的 2 层 MLP projector（dim 16 → 32 → 32）纯 Python。
2. Prompt 构建流水线：system prompt + `<image>` 被 N 个投影 token 取代 + user turn + assistant generation placeholder。
3. 576-token 视觉块在 LLM 上下文中占比的可视化（2k / 32k / 128k 上下文消耗百分比）。

## 交付它

本课产生 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 家族 checkpoint，它运行一个 10-prompt vibes-eval 套件（3 个标题生成、3 个 VQA、2 个推理、2 个拒绝）并报告人类可读的记分卡。不是基准；一个烟雾测试，确认 projector 和 LLM 连接良好。

## 练习

1. 计算 2 层 MLP projector 在 `1024 → 4096 → 4096` 时的可训练参数。含 GELU 和 bias，它占 LLaVA-13B 的多少比例？

2. 为"拒绝"案例构造一个 LLaVA prompt——图像包含一个私人个体。写出预期的 assistant 响应。为什么 LLaVA 应该 zero-shot 拒绝这个，需要什么训练数据来强化拒绝？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的视觉 token 数量。与基础 336x336 的 576 token 对比。

4. LLaVA 阶段 1 projector 用标题上的 LM loss 训练。如果你跳过阶段 1 直接进入阶段 2（视觉指令微调）会发生什么？引用 Prismatic VLMs ablation（arXiv:2402.07865）作为答案。

5. LLaVA-Instruct-150k 用 COCO 标题通过 GPT-4 生成指令。对于一个新领域（医学 X 光、卫星图像），描述生成领域指令的四步数据流水线。每一步可能出什么错？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Projector | "MLP bridge" | 2 层 MLP 带 GELU，把 ViT dim 映射到 LLM dim |
| Image token | "<image> placeholder" | Prompt 标记，在推理前被 N 个投影视觉 token 取代 |
| Visual instruction tuning | "LLaVA stage 2" | 在 GPT-4 生成的（图像、指令、响应）三元组上训练 |
| Stage 1 alignment | "Projector pretraining" | 冻结 ViT 和 LLM，在标题上用 LM loss 训练 projector |
| AnyRes | "Multi-crop tiling" | 把高分辨率图像分成瓦片网格，拼接每个瓦片的视觉 token |
| LLaVA-Instruct | "GPT-4-generated" | 从 COCO 标题 + GPT-4 合成的 158k 指令-响应对 |
| Vision encoder freeze | "Backbone locked" | 阶段 1 CLIP 权重不更新，阶段 2 有时也不 |
| ShareGPT4V | "Better captions" | GPT-4V 生成的 1M 密集标题，用于更高质量对齐 |
| VQA | "Visual question answering" | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | "Design-space paper" | Karamcheti 2024 系统测试 projector 和数据选择的 ablation |

## 延伸阅读

- [Liu 等人 — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu 等人 — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen 等人 — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集标题数据集。
- [Karamcheti 等人 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间 ablations。
- [Li 等人 — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图像、多图像、视频。
