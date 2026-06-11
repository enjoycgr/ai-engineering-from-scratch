# InternVL3：原生多模态预训练

> InternVL3 之前的每个开放 VLM 遵循相同三步配方：拿一个在数万亿文本 token 上训练的文本 LLM，装上一个视觉编码器，然后微调接缝。这有效但有对齐债务——文本 LLM 花掉了完整预训练预算在纯文本上，不原生理解视觉 token。当你事后加视觉，LLM 必须重新学习如何把视觉输入与其文本推理关联而不遗忘文本。InternVL3 (Zhu 等人, 2025 年 4 月) 拒绝事后方法：一次预训练，文本和多模态从第一步起就交错。结果在 MMMU-Pro 上 78B 参数开放匹配 Gemini 2.5 Pro。本课阅读原生预训练的案例以及当你这样做时什么改变。

**类型：** Learn
**语言：** Python (stdlib, training-corpus mixer)
**前置知识：** Phase 12 · 05, Phase 12 · 07 (recipes)
**时间：** ~120 分钟

## 学习目标

- 解释为什么事后 VLM 训练积累对齐债务，引用三个可测量症状（灾难性遗忘、答案漂移、视觉-文本不一致）。
- 描述 InternVL3 的原生预训练语料混合及为什么 text : interleaved : caption 的比例重要。
- 对比 V2PE（可变视觉位置编码）与 Qwen2-VL 的 M-RoPE。
- 说出 Visual Resolution Router (ViR) 和 Decoupled Vision-Language (DvD) 部署优化。

## 问题

事后 VLM 训练是默认。LLaVA、BLIP-2、Qwen-VL、Idefics——都拿一个已预训练的 LLM（Llama、Vicuna、Qwen、Mistral）并加视觉。训练阶段通常看起来像这样：

1. 冻结 LLM + 冻结视觉编码器 + 可训练 projector，在标题对上训练以对齐 embedding。
2. 解冻 LLM，在指令数据（LLaVA-Instruct、ShareGPT4V）上训练。
3. 可选任务特定微调。

三种对齐债务症状出现：

- 灾难性遗忘。事后 VLM 遗忘纯文本技能。GSM8K 分数掉 5-10 分。Hellaswag 分数掉。纯文本 agent 退化。
- 答案漂移。相同视觉问题的微小措辞得到不同答案。视觉编码器与 LLM 的连接比 LLM 自身 token 更弱。
- 视觉-文本不一致。VLM 能正确描述图像然后回答与其自身描述矛盾的问题。视觉 token 不像文本那样参与 LLM 的内部一致性检查。

这些症状有充分记录。MM1.5 Section 4 量化它们。LLaVA-OneVision 的 ablations 暗示它们。原生预训练是答案。

## 概念

### 原生多模态预训练

InternVL3 从头训练语料上训练，该语料从第一步起就是原生多模态。混合是：

- 40% 纯文本数据（FineWeb、Proof-Pile-2 等）
- 35% 交错图像-文本数据（OBELICS、MMC4 风格）
- 20% 成对图像-标题数据
- 5% 视频-文本数据

视觉 token、文本 token 和跨模态交互都从第一个梯度步骤参与相同 loss。无对齐预训练，无 projector 冻结阶段，无灾难性遗忘需要恢复。

基础模型训练是单阶段。指令微调跟随，但基础模型已把视觉 token 理解为头等公民。

### V2PE（可变视觉位置编码）

Qwen2-VL 使用固定轴分配的 M-RoPE。InternVL3 引入 V2PE：位置编码按模态类型（文本、图像、视频）变化，带可学习缩放。实践中：

- 文本 token 获得 1D 位置（文本索引）。
- 图像 patch 获得 2D 位置（行、列）。
- 视频帧获得 3D 位置（时间、行、列）。

三者共享相同 RoPE 频率基础，但每频带的 hidden-dim 分配是学习参数而非固定分割。在预训练期间自由交易时序 vs 空间频率分辨率。

V2PE 的 ablation 声称：相同计算下比 M-RoPE 在视频基准上高 1-2 分。不是革命，但更干净。

### Visual Resolution Router (ViR)

部署优化。并非所有图像都需要全分辨率编码。低细节单物体照片在 1280px 原生编码时浪费 token。ViR 是一个小分类器，预测回答问题所需的最小分辨率，在编码之前。

路由有三个层级：低分辨率（256 token）、中等（576）、高（2048+）。在生产流量中 60% 的查询，低或中等足够。净效果：相等质量下 2-3 倍吞吐。

### Decoupled Vision-Language deployment (DvD)

服务大型 VLM 时，视觉编码器每张图像运行一次但 LLM 为每个输出 token 自回归运行。两个组件有不同瓶颈（视觉 = conv + attention 的 GPU 内存带宽；LLM = KV cache）。DvD 把它们拆到不同 GPU 并流式传输。

对 8B + 400M 编码器模型，DvD 大致翻倍每节点吞吐 vs 共置。

### 单阶段 vs 多阶段质量

InternVL3 的主要基准声明：78B 参数下匹配 Gemini 2.5 Pro 的 MMMU-Pro。38B 下匹配 GPT-4o。8B 下领先开放 8B 排行榜。全在单阶段预训练 + 指令微调配方上。

对齐债务假说可测量：InternVL3-8B 比 Qwen2.5-VL-7B 每单位视觉基准增益丢失更少的文本基准分数（MMLU、GSM8K）。模型更通才因为训练是一块而非两块。

### InternVL3.5 和 InternVL-U

InternVL3.5 (2025 年 8 月) 扩展配方。相同原生预训练方法，更多数据，更多参数。MMMU 改进是增量的。

InternVL-U (2026) 添加统一生成——通过相同主干顶部的 MMDiT head 输出图像。"U"代表"Understanding + generation"，追逐 Transfusion 风格统一模型（Lesson 12.13）。相同原生预训练主干支持理解和生成 head。

### 原生预训练的权衡

原生预训练不免费：

- 计算。从头训练新 VLM 成本与训练文本 LLM 相同——数百万 GPU 小时。事后适配重用现有 LLM 权重，节省大部分成本。
- 数据。大规模交错图像-文本语料罕见。OBELICS 1.41 亿文档；MMC4 5.71 亿。纯文本发货为 15T token。多模态预训练数据稀缺是硬约束。
- 基础 LLM 重用。原生预训练放弃了稍后换入新 LLM 的选项。事后让你通过仅重训练 adapter 换 Llama-3.1 为 Llama-4。

InternVL3 的赌注：对齐债务比重用损失更糟。基准支持声明。生产成本阻止未来实验室廉价复制。事后 VLM 将继续存在因为它们对大多数项目更便宜。

## 使用它

`code/main.py` 是一个训练语料混合器和 ViR 路由器模拟器。它：

- 取目标语料混合（%文本、%交错、%标题、%视频）并计算每模态预期步数。
- 模拟 batch 查询上的 ViR 路由（分布：50% 低细节、30% 中等、20% 高细节）并报告平均 token 数。
- 给定编码器 vs LLM FLOPs 报告 DvD 吞吐估计。
- 打印事后 vs 原生预训练在参数、计算、数据和预期对齐债务症状上的并排对比。

## 交付它

本课产生 `outputs/skill-native-vs-posthoc-auditor.md`。给定一个提议的 VLM 训练计划，它审计是否走原生或事后，标记对齐债务风险，并推荐语料混合。每当你调整新开放 VLM 项目并需要挑选训练策略时使用。

## 练习

1. 估计 InternVL3-8B（原生预训练）和 LLaVA-OneVision-7B（事后）之间的计算差。GPU 小时比大约？什么解释了差距？

2. InternVL3 报告 40% 文本 / 35% 交错 / 20% 标题 / 5% 视频。如果你的目标任务是视频重，提出新比例并论证为什么基础模型仍需要大量文本和标题数据。

3. 阅读 MM1.5 Section 4 关于遗忘。说出事后训练显示最大退化的精确基准。退化花了多少代价？

4. ViR 把 60% 流量路由到低分辨率编码。哪些查询类型它误路由（需要高分辨率时送到低分辨率）？提出三种路由器失败模式。

5. DvD 把视觉和 LLM 拆到不同 GPU。什么流量模式下 DvD 伤害而非帮助吞吐？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Native multimodal pretraining | "From scratch together" | 文本 + 图像 + 视频 token 从第 1 步就参与 loss，不是事后装上 |
| Alignment debt | "Post-hoc penalty" | 把视觉装到冻结 LLM 上带来的文本技能可测量退化和答案一致性 |
| V2PE | "Variable visual pos encoding" | 每模态可学习位置编码分配；InternVL3 的 M-RoPE 继任者 |
| ViR | "Resolution router" | 在编码前预测每查询所需最小分辨率的小型分类器，节省推理 token |
| DvD | "Decoupled deployment" | 视觉编码器在一个 GPU，LLM 在另一个，流式传输；大型 VLM 翻倍吞吐 |
| InternVL-U | "Unified understanding + generation" | 2026 后续，通过 MMDiT head 在原生预训练主干上添加图像生成 |
| Interleaved corpus | "OBELICS / MMC4" | 带自然阅读顺序文本和图像的文档；原生预训练的原材料 |

## 延伸阅读

- [Chen 等人 — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu 等人 — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang 等人 — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
