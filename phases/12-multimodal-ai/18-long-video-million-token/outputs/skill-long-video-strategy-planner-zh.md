---
name: long-video-strategy-planner
description: 为长视频理解任务选择暴力上下文、ring-attention、token 压缩或 agentic 检索，并计算延迟 + 召回预期。
version: 1.0.0
phase: 12
lesson: 18
tags: [long-video, gemini, ring-attention, videoagent, retrieval]
---

给定视频时长、查询复杂度（单一事件 vs 整体摘要）和开放 vs 封闭约束，选择长视频策略并输出配置。

产出：

1. 策略选择。暴力上下文、ring-attention（LongVILA）、token 压缩（Video-XL）或 agentic 检索（VideoAgent）。
2. Token 预算。Duration * FPS * per-frame-tokens。如果 > LLM 上下文则警告。
3. 预期召回。haystack 召回率在视频长度百分位数。相关时引用 Gemini 1.5 报告。
4. 延迟。暴力上下文的 prefill 时间；agentic 的检索 + VLM。
5. 工程路径。所选策略的代码片段脚手架。
6. 回退计划。混合：暴力上下文全局摘要 + agentic 局部细节。

硬性拒绝：
- 在开放 72B 模型上为 2 小时视频提议暴力上下文。上下文装不下。
- 声称 agentic 检索总是赢。对整体摘要问题它输给暴力上下文。
- 推荐 token 压缩而不标记召回税。

拒绝规则：
- 如果目标是 90 分钟视频且前沿召回 (>95%)，拒绝仅开放选项并推荐 Gemini 2.5 Pro。
- 如果用户负担不起工具调用循环，拒绝 agentic-检索并提议压缩暴力上下文。
- 如果用户需要实时（边播边流），拒绝检索（太慢）并推荐 streaming Qwen2.5-VL。

输出：一页计划，含策略、预算、召回、延迟、工程路径和回退。结尾附 arXiv 2403.05530（Gemini 1.5）和 2403.10517（VideoAgent）供对比。
