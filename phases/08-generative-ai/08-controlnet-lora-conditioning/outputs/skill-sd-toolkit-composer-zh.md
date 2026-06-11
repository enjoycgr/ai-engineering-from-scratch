---
name: sd-toolkit-composer
description: 在给定输入集合之上，将 ControlNets、LoRAs 和 IP-Adapters 组合到 SD / Flux base 上。
version: 1.0.0
phase: 8
lesson: 08
tags: [controlnet, lora, ip-adapter, diffusion]
---

给定一个任务（目标图像）、输入（prompt、参考图像、姿态 / 深度 / 涂鸦 / 分割、主体身份）和 base model（SDXL、SD3.5、Flux.1-dev），输出：

1. ControlNet stack（ControlNet 栈）。哪些 ControlNets（canny / openpose / depth / scribble / seg / lineart / tile），什么权重，什么顺序。最大权重总和 <= 1.5。
2. LoRA stack（LoRA 栈）。命名的 LoRAs、rank、alpha。当 alpha > 1.5 或多个 LoRA 针对同一概念时发出警告。
3. IP-Adapter。None、plain 或 FaceID 变体；权重 0.4-0.8 典型值。
4. Text prompt + negative prompt（文本提示词 + 负提示词）。关键词顺序、token budget、negative scaffolding。
5. Sampler + CFG + seed。Euler A / DPM-Solver++ / LCM；CFG scale 与 base 绑定。可复现的 seed protocol。
6. QA checklist。ControlNet drift、LoRA over-saturation、IP-Adapter identity leak、解剖问题的视觉检查。

拒绝将 SD 1.5 LoRA 堆叠到 SDXL base 上（维度不匹配）。拒绝以权重 1.0 各运行 3+ ControlNets（特征碰撞）。当用户有 SDXL 或 Flux 的 GPU 预算时，标记任何 SD 1.5 推荐。当 LoRA 身份训练在 < 10 张图像上时，标记为可能过拟合。
