# 开源权重 VLM 配方：什么真正重要

> 2024-2026 年开源权重 VLM 文献是一片 ablation table 的森林。Apple 的 MM1 测试了 13 种图像编码器、连接器和数据混合的组合。Allen AI 的 Molmo 证明详细的人类标题击败了 GPT-4V 蒸馏。Cambrian-1 运行了 20+ 编码器对比。Idefics2 形式化了五轴设计空间。Prismatic VLMs 在受控基准上对比了 27 个训练配方。在所有噪音中，一小套结果跨论文成立：图像编码器比连接器架构更重要，数据混合比两者都更重要，详细的人类标题击败了蒸馏合成数据。本课阅读这些表格，让你不必亲自读。

**类型：** Learn + lab
**语言：** Python (stdlib, ablation table parser + recipe picker)
**前置知识：** Phase 12 · 05 (LLaVA baseline)
**时间：** ~180 分钟

## 学习目标

- 说出五轴 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率方案。
- 阅读 MM1 / Idefics2 / Cambrian-1 ablation table 并预测哪个旋钮移动给定基准。
- 给定计算预算和任务混合，为新 VLM 挑选配方（编码器、连接器、数据、分辨率）。
- 解释为什么详细的人类标题在相同 token 计数下击败 GPT-4V 蒸馏。

## 问题

数百个开源权重 VLM 存在。"好"与"最先进"之间的大部分差距不是架构。是数据、分辨率方案和编码器选择。知道当你的模型表现不佳时先转哪个旋钮，可以省掉一个 500 万 GPU 小时的错误。

2023 波（LLaVA-1.5、InstructBLIP、MiniGPT-4）运行在标题对预训练 + LLaVA-Instruct-150k 上。好基线。MMMU 约 35% 封顶。

2024 波（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）运行详尽 ablations。结果令人惊讶且实用。

## 概念

### 五轴设计空间

Idefics2 (Laurençon 等人, 2024) 命名了轴：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch size、分辨率和预训练目标上不同。
2. 连接器。MLP（2-4 层）、Q-Former（32 query + cross-attn）、Perceiver Resampler（64 query）、C-Abstractor（卷积 + 双线性 pooling）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主导参数成本。
4. 训练数据。标题对（CC3M、LAION）、交错（OBELICS、MMC4）、指令（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率方案。固定 224/336/448、AnyRes、原生动态。训练期间 ramped 或恒定。

每个生产 VLM 在每个轴上做选择。MMMU 分数的大部分方差由轴 1、4、5 解释——不是你选的哪个连接器。

### 轴 1：编码器 > 连接器

MM1 Section 3.2 表明：从 CLIP ViT-L/14 换到 SigLIP SO400m/14 增加 3+ MMMU 分。连接器从 MLP 换到 Perceiver Resampler 增加不到 1 分。Idefics2 复现：SigLIP > CLIP，Q-Former ≈ MLP ≈ Perceiver 在相同 token 计数下。

Cambrian-1 的 "Cambrian Vision Encoders Match-Up" (Tong 等人, 2024) 在视觉中心基准 (CV-Bench) 上运行 20+ 编码器。排行榜顶部是 DINOv2 和 SigLIP 的混合；CLIP 是中游；ImageBind 和 ViT-MAE 更低。从 CLIP ViT-L 到 DINOv2 ViT-g/14 的 gap 在 CV-Bench 上约 5-7 分。

2026 年开源 VLM 默认编码器是语义 + 密集特征的 SigLIP 2 SO400m/14，有时拼接 DINOv2 ViT-g/14 特征（Cambrian 的 "Spatial Vision Aggregator" 这样做）。

### 轴 2：连接器设计是平局

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出相同结论：在固定视觉 token 计数下，连接器架构几乎不重要。2 层 MLP 在 mean-pooled patch 上表现与相同 token 预算下的 32-query Q-Former 在 1 分以内。

重要的是 token 计数。更多视觉 token = 更多 LLM 计算 = 更好性能到一个点，然后收益递减。每张图像 64 token 对 OCR 太少。576-1024 token 对大多数开源 VLM 是最佳点。2048+ 只对文档和图表有帮助。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 无论图像分辨率都限制 token 在 32-64；MLP 发出所有 patch token。高分辨率输入时 Q-Former 节省 LLM 上下文；低分辨率时差异是噪声。

### 轴 3：LLM 大小设置天花板

LLM 从 7B 翻倍到 13B 在每个 VLM 论文上可靠增加 2-4 MMMU 分。70B 时大多数基准饱和。VLM 的多模态推理天花板是 LLM 的文本推理天花板——视觉编码器只能喂它，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上碾压：语言大脑巨大。7B VLM 无法通过巧妙的连接器设计替代 70B VLM。

### 轴 4：数据——详细人类标题击败蒸馏

Molmo + PixMo (Deitke 等人, 2024) 是每个 2024 年人都该读的结果。Allen AI 让人类标注员用 1-3 分钟密集语音转文字描述图像，产出 712K 密集标题图像。训练数据中没有任何 GPT-4V 蒸馏。

Molmo-72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。差异不是架构——是标题质量。详细的人类标题每张图像包含 5-10 倍于短网络标题的信息，并在 GPT-4V 蒸馏产生幻觉的地方保持事实接地。

ShareGPT4V (Chen 等人, 2023) 和 Cauldron (Idefics2) 遵循相同剧本，混合人类 + GPT-4V 标题。趋势清晰：对 2026 前沿，标题密度 > 标题数量 > 蒸馏便利性。

### 轴 5：分辨率及其方案

Idefics2 的 ablations：384 -> 448 增加 1-2 分。448 -> 980 带图像拆分 (AnyRes) 在 OCR 基准上再增加 3-5。平坦分辨率训练在中等准确率上饱和；分辨率 ramping（从 224 开始，结束于 448 或原生）训练更快且最终更高。

Cambrian-1 运行分辨率 vs token 权衡：固定计算下，你可以低分辨率更多 token 或高分辨率更少 token。高分辨率对 OCR 获胜；低分辨率更多 token 对一般场景理解获胜。

2026 年生产配方：阶段 1 在 384 固定训练，阶段 2 对 OCR 重任务动态分辨率到 1280。

### Prismatic 受控对比

Prismatic VLMs (Karamcheti 等人, 2024) 是控制了所有轴的论文。相同 13B LLM、相同指令数据、相同评估——一次只变一个轴。结果：

- 每张图像视觉 token 数解释约 60% 方差。
- 编码器选择解释约 20%。
- 连接器架构解释约 5%。
- 其他一切（数据混合、scheduler、LR）剩余约 15%。

这是一个粗略分解，但它是文献中"我该先 ablate 什么"最干净的答案。

### 2026 年选择器

给定证据，2026 年新项目的默认开源 VLM 配方：

- 编码器：SigLIP 2 SO400m/14 原生分辨率带 NaFlex，如需分割/定位则拼接 DINOv2 ViT-g/14 密集特征。
- 连接器：patch token 上的 2 层 MLP。除非 token 受限，跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 选成本，70B 选质量，按目标延迟挑。
- 数据：PixMo + ShareGPT4V + Cauldron，补充任务特定指令数据。
- 分辨率：动态（min 256，max 1280 长边像素）。
- 方案：阶段 1 对齐（仅 projector），阶段 2 全微调，阶段 3 任务特定微调。

每个默认都追溯到本课末尾引用论文中的测量 ablation。

## 使用它

`code/main.py` 是一个 ablation table parser 和 recipe picker。它编码 MM1 和 Idefics2 ablation table（浓缩版）并让你查询：

- "给定预算 X 和任务 Y，哪个配方获胜？"
- "如果我把 SigLIP 换 CLIP 在 7B Llama 上，预期 MMMU 差多少？"
- "哪个轴应该先 ablate 以获得 80% 置信度答案？"

输出是排名的 recipe 列表，带预期基准差和 "先 ablate" 建议。

## 交付它

本课产生 `outputs/skill-vlm-recipe-picker.md`。给定目标任务混合、计算预算和延迟目标，它发出完整配方（编码器、连接器、LLM、数据混合、分辨率方案）并引用证明每个选择的 ablation。阻止工程师每次新 VLM 项目开始时重新发明 Idefics2 ablation table。

## 练习

1. 阅读 MM1 Section 3.2。固定 2B LLM 在 5000 万图像预算下，哪个编码器获胜？13B LLM 时答案会翻转吗？为什么？

2. Cambrian-1 发现拼接 DINOv2 + SigLIP 在视觉中心基准上击败单独任一，但在 MMMU 上不加信号。预测哪些基准获益、哪些持平。

3. 你的目标是 2B LLM 上的移动 UI agent。挑选编码器、连接器、分辨率和数据混合。用特定 ablation table 证明每个选择。

4. Molmo 发布 4B 和 72B 模型。4B 与闭源 7B VLM 竞争；72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。这告诉你关于 LLM 大小天花板假说的什么？

5. 设计一个 ablation table 以在 7B VLM 上隔离数据混合质量与编码器质量。最少多少次训练运行？提出四个轴设置。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Ablation | "Turning one knob" | 在恰好一个设计空间轴上不同的多次训练，保持其他一切恒定 |
| Connector | "Bridge" / "projector" | 把视觉编码器输出映射进 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| Detailed human caption | "Dense caption" | 多句人类撰写描述（通常 80-300 token），比网络 alt 文本丰富 |
| Distillation | "GPT-4V captions" | 由更强专有 VLM 生成的训练数据；方便但易继承幻觉 |
| AnyRes / dynamic res | "High-res path" | 通过 tiling 或 M-RoPE 喂给编码器大于原生分辨率的图像的策略 |
| Resolution ramp | "Curriculum" | 从低分辨率开始并增加的训练方案，加速对齐学习 |
| Vision-centric bench | "CV-Bench / BLINK" | 强调细粒度视觉感知而非语言重推理的评估 |
| PixMo | "Molmo's data" | Allen AI 的 712K 密集标题图像数据集；人类语音转录成密集标题 |

## 延伸阅读

- [McKinzie 等人 — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon 等人 — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke 等人 — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong 等人 — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti 等人 — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
