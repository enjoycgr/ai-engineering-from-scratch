---
name: gated-bridge-diagnostic
description: 识别开放 VLM 配置中的 Flamingo 谱系设计元素并诊断 freezing / gating 问题。
version: 1.0.0
phase: 12
lesson: 04
tags: [flamingo, idefics, openflamingo, gated-cross-attention, interleaved-inputs]
---

给定一个开放 VLM checkpoint 及其配置（层结构、cross-attention 计划、gate 参数化、训练配方），识别它使用哪些 Flamingo 谱系元素并诊断 gating 设置错误的常见症状。

产出：

1. 谱系清单。标记 (Perceiver resampler Y/N、门控 cross-attn 频率 M、tanh vs sigmoid gate、alpha init 值、LLM freeze depth) 的存在。
2. 交错输入支持。解析模型期望的 prompt 格式；确认或否认多图像、视频和 few-shot in-context prompting 的支持。
3. 视觉 token 预算。计算每张图像成本：K latents x N cross-attn 插入点。与相同图像数下 BLIP-2 风格单输入桥对比。
4. Gate 诊断。给定训练 loss 曲线或基准退化，建议 gate 是否开得太快（丢失文本能力）、太慢（未能使用视觉输入）或校准错误（视觉 token 竞争而非增强）。
5. 修复配方。具体参数修复：如果文本退化将 alpha 初始化更接近 0、提高 gate 参数的学习率、或在前 N 步冻结 gate。

硬性拒绝：
- 未经检查 resampler 和 gate 计划就将任何开放 VLM 视为 "a Flamingo"。Idefics2 去掉了 resampler；不加限定地标记为 Flamingo 谱系是错误的。
- 假设零初始化总能撑过训练。一些开源复现使用小的非零初始化，以初始稳定性换取更快收敛。
- 声称门控 cross-attention 对所有任务都严格优于单个 BLIP-2 桥。在小 LLM 的单图像 VQA 上，额外的 cross-attn 层是纯成本。

拒绝规则：
- 如果 checkpoint 的训练配方未公开，拒绝并解释为什么 gate 诊断需要知道 gate schedule。
- 如果调用者要求与 Gemini 或 Claude（专有）对比，拒绝——它们的 gating 机制未公开。
- 如果范围内的 VLM 是 early-fusion 模型（Chameleon、Emu3），拒绝——gating 只适用于 adapter 风格 VLM。

输出：一页诊断，含谱系清单、交错输入能力矩阵、token 预算、gate 诊断和具体修复配方。以 "what to read next" 段落结尾，指向 Lesson 12.05 (LLaVA) 了解替代 projector 方法或 Lesson 12.11 (Chameleon) 了解 early-fusion 逃生舱。
