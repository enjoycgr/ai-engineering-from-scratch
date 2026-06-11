---
name: modality-bridge-picker
description: 给定 token 预算、质量目标和训练计算量，为 VLM 配置推荐 Q-Former vs MLP projector vs Perceiver resampler。
version: 1.0.0
phase: 12
lesson: 03
tags: [blip2, qformer, vlm, modality-bridge, architecture]
---

给定视觉编码器的每张图像 token 数、LLM 的上下文预算、每张 prompt 的目标图像数和训练计算预算，推荐用哪种模态桥并用参数量和 token 经济学证明。

产出：

1. Token 预算审计。报告视觉编码器的每张图像原始 token、每种桥选项后的每张图像 token，以及在声明的每张 prompt 图像数下消耗的 LLM 上下文比例。
2. 桥对比。对 Q-Former（32 token，~188M 参数）、MLP projector（所有 patch，~20M 参数）和 Perceiver resampler（K 可学习 query 经 N 层 cross-attention，可变），给出参数、质量代理和训练成本估算。
3. 推荐。基于声明约束的单一最佳选择，附一行理由。当约束矛盾时标记（高质量 + 紧 token 预算 + 低训练计算）。
4. 两阶段训练轨迹。如果选 Q-Former，概述阶段 1 的 ITC + ITM + ITG loss 和阶段 2 的 LM loss。为每个命名一个代表性数据集（COCO、LAION、Visual Genome）。
5. Ablation 清单。调用者在锁定桥之前应运行的五个实验（query 数量、两阶段 vs 单阶段、projector 深度、freeze 计划、微调子集）。

硬性拒绝：
- 任何忽略 token 预算的推荐。"用 MLP" 配每张图像 576 token 在 10 张图像进入 4k 上下文时失败。
- 声称 Q-Former 严格优于 MLP。在单图像高质量任务且上下文无限时，MLP 获胜。
- 将 Perceiver resampler 等同于 Q-Former。Flamingo 在每一层 LLM 应用它；BLIP-2 只应用一次。

拒绝规则：
- 如果调用者要求一个能处理视频的桥而不指定多少帧和帧率，拒绝——视频桥与单图像桥的区别在于规格，不只是规模。
- 如果范围内的 LLM 是与视觉塔一起从头训练的（early-fusion，Chameleon 风格），拒绝——Lesson 12.11 单独覆盖该情况。
- 如果未声明训练计算量，拒绝并询问调用者是否能负担 BLIP-2 阶段 2（约几百 A100 小时）或只能负担 projector-only 训练。

输出：一页桥推荐，含 token 数学、参数量、推荐架构、训练大纲和 ablation 清单。以 "what to read next" 段落结尾，指向 Lesson 12.04 (Flamingo) 了解 everywhere cross-attention、Lesson 12.05 (LLaVA) 了解仅 MLP，或 Lesson 12.07 (ablations) 了解数据 vs 架构权衡。
