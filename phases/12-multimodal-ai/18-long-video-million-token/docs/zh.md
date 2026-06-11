# 百万 Token 上下文的长视频理解

> 1 小时 4K 视频在 24 FPS，patch 并 embedding 后，产生约 6000 万 token。2 小时播客逐字稿是 30,000 token。完整蓝光故事片，即使激进 pool，也是数十万 token。Google 的 Gemini 1.5 (2024 年 3 月) 用 1000 万 token 上下文开启了这一时代，在小时长视频上做可靠的 needle-in-a-haystack 召回。LWM (Liu 等人, 2024 年 2 月) 展示了 ring attention 的扩展路径。LongVILA 和 Video-XL 进一步扩展了摄入。VideoAgent 用 agentic 检索交换原始上下文。每种方法是计算、召回和工程复杂度上的不同权衡。本课并排阅读它们。

**类型：** Build
**语言：** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**前置知识：** Phase 12 · 17 (video temporal tokens)
**时间：** ~180 分钟

## 学习目标

- 计算长形式视频在不同 FPS 和 pool 下的总视觉 token 数。
- 解释三种扩展路径：暴力上下文 (Gemini 1.5)、ring attention (LWM)、token 压缩 (LongVILA / Video-XL)。
- 对比原始上下文视频 VLM 与 agentic-检索视频 VLM (VideoAgent) 在准确率和延迟上。
- 为 30 分钟视频设计 needle-in-a-haystack 测试并在特定分钟测量召回。

## 问题

Qwen2.5-VL 大小 patch 在 384 原生分辨率下单帧约 ~729 token。3x3 pool 下每帧 81 token。30 分钟片段在 1 FPS 是 1800 帧 = 145,800 token。2025 开放 VLM 可做，紧张。2 FPS 下 291,600 token——只有最大上下文能装。

2 小时电影在 1 FPS 是 583k token。超出大多数 2026 开放模型；需要 Gemini 2.5 Pro 或更激进 pool。

三条扩展路径出现。

## 概念

### 路径 1：暴力上下文 (Gemini 1.5, Claude Opus)

向问题扔硬件。扩展上下文到数百万 token，一次 forward pass 处理一切。

Gemini 1.5 Pro 首发 1M token；Gemini 1.5 Ultra 到 10M；Gemini 2.5 Pro 2026 年可靠处理小时视频。论文 (arXiv:2403.05530) 记录 needle-in-a-haystack 召回在 ~9.5M token 内 99.7%。

工程：自定义 attention 实现带内存层级（局部 + 全局 + 稀疏）加 MoE expert 路由用于长上下文效率。未完整公开。非开源。

### 路径 2：Ring attention (LWM, LongVILA)

Ring attention 以"环"跨设备分布长序列，每设备持有一块。跨完整序列的 attention 通过每设备发送其块到环中的下一个、计算 partial attention、并聚合来发生。

LWM (Liu 等人, 2024) 这样训练 1M token 上下文模型。训练计算随上下文线性扩展，而非二次——attention 的二次打击被摊销到环的设备上。

LongVILA (arXiv:2408.10188) 将模式适配到 VLM。1400 帧视频在 192 token/帧 = 268k 上下文，在 8 路并行上用 ring attention 训练。

### 路径 3：Token 压缩 (Video-XL, LongVA)

比暴力上下文更便宜：在 LLM 看到序列前压缩。

Video-XL (arXiv:2409.14485) 使用视觉摘要 token：每 N 帧片段产生一个关注所有 N 的"摘要"token。推理时 LLM 每片段看到一个摘要 token，大幅缩小上下文。

LongVA 通过"长上下文迁移"技术将 LLM 上下文从 200k 扩展到 2M。在长上下文文本上训练，通过共享表示迁移到长上下文视频。

Token 压缩交易特定时间戳的召回以换取可扩展性。模型知道大致发生了什么但有时错过精确帧。

### 路径 4：Agentic 检索 (VideoAgent)

不将完整视频喂给 LLM。相反，将视频当作数据库并用 LLM 查询它。

VideoAgent (arXiv:2403.10517)：

1. LLM 读问题。
2. LLM 请求检索工具获取相关片段（"给我有猫的片段"）。
3. 工具返回匹配时间戳。
4. LLM 通过 VLM 读取这些片段。
5. LLM 组合答案或请求跟进查询。

这是应用于长视频的 LLM-as-agent 模式。更便宜推理（只编码相关片段），更难工程（检索质量成为瓶颈）。

### Needle-in-a-haystack 基准

标准长上下文测试：在视频随机点插入唯一视觉或文本标记，然后问需要召回它的查询。

指标：跨视频长度和标记位置的 Recall@k。

Gemini 2.5 Pro 在 90 分钟视频上评分 >99% 召回。开放 72B 模型 (Qwen2.5-VL-72B, InternVL3-78B) 在 30 分钟评分 ~85-90% 并在 60 分钟后退化。

VideoAgent 在 2+ 小时上可匹配或击败原始上下文模型，如果工具好。

### 挑选哪条路径

15 分钟片段在前沿准确率：开放 72B + 原生上下文通常有效。选 Qwen2.5-VL-72B。

30 分钟到 1 小时内容：LongVILA 或 Video-XL 用于开放；Gemini 2.5 Pro 用于封闭。质量栏重要——前沿走封闭。

2+ 小时内容：VideoAgent 或类似检索模式。替代：总结为更小片段并喂分层摘要。

### 2026 年生产模式

实践中，生产长视频流水线是混合的：

1. 在整个视频上运行动态 FPS 采样 + 激进 pool（获得 100k-token 全局表示）。
2. 传给 72B VLM 做全局摘要。
3. 如果用户问详细问题，用摘要作为索引运行 agentic 检索。

这结合暴力上下文做全局理解和检索做局部细节。

## 使用它

`code/main.py`：

- 计算 1 分钟到 3 小时视频在不同 FPS + pool 下的 token 预算。
- 模拟 needle-in-a-haystack 运行：在随机时间戳注入标记，问问题，评分召回。
- 包括 agentic-检索路由器模拟器，挑选特定片段喂给下游 VLM。

运行预算表并感受规模差距。

## 交付它

本课产生 `outputs/skill-long-video-strategy-planner.md`。给定视频持续时间和查询复杂度，它在暴力上下文、压缩和 agentic 检索之间挑选，并计算延迟 + 质量预期。

## 练习

1. 45 分钟讲座在 1 FPS、81 token/帧。总 token？适配哪些模型上下文？

2. 设计 needle-in-a-haystack 测试：你在哪分钟注入标记，精确查询格式是什么？

3. 对比 1 小时视频上暴力上下文 Qwen2.5-VL-72B (80k 上下文) 与 VideoAgent (Claude 3.5 + 检索)。哪个在召回上获胜？哪个在延迟上获胜？

4. Ring attention 的内存成本在序列长度和设备数上线性扩展。解释为什么以及如果去掉环旋转阶段什么会失败。

5. 阅读 Gemini 1.5 Section 5 关于 needle-in-a-haystack。论文在 1M vs 10M token 边界发现召回什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Brute context | "Just more tokens" | 将 LLM 上下文扩展到数百万 token；一次 pass 处理一切 |
| Ring attention | "LWM-style parallel" | 分布式 attention 模式，每设备持有一块并旋转 |
| Token compression | "Summary tokens" | 通过学习的压缩器在 LLM 前减少每片段 token |
| Needle-in-haystack | "NIH test" | 在随机点插入唯一标记，测试时让模型召回它 |
| Agentic retrieval | "LLM as query planner" | LLM 请求检索工具获取相关片段，通过 VLM 读取，组合答案 |
| VideoAgent | "Retrieval pattern for video" | 经典 agentic-检索设计：问题 -> 工具 -> 片段 -> 答案 |

## 延伸阅读

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu 等人 — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue 等人 — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu 等人 — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang 等人 — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
