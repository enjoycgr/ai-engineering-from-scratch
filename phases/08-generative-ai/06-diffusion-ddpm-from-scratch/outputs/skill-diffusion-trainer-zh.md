---
name: diffusion-trainer
description: 配置一次扩散训练运行：调度、预测目标、采样器和评估计划。
version: 1.0.0
phase: 8
lesson: 06
tags: [diffusion, ddpm, training]
---

给定数据集画像（模态、分辨率、数据集大小）、计算预算（GPU 小时数、VRAM 下限）和质量门槛（FID 目标或下游用途），输出：

1. 调度（Schedule）。线性（linear）、余弦（cosine，Nichol）或 sigmoid。时间步数 T（DDPM 基线为 1000；更快变体为 256）。
2. 预测目标（Prediction target）。epsilon、v-prediction 或 x_0。理由与分辨率和整个调度上的信噪比相关。
3. 架构（Architecture）。像素扩散用 U-Net 深度 + 通道宽度，隐空间扩散（latent diffusion）用 DiT，视频用 3D U-Net / DiT。包含时间嵌入方案（正弦 + MLP、FiLM 或 AdaLN）。
4. 采样器（Sampler）。DDIM（20-50 步）、DPM-Solver++（10-20）、Euler-A（创意型）或蒸馏 1-4 步。包含引导尺度（guidance scale，CFG w）建议。
5. 评估计划（Eval plan）。FID / KID / CLIP-score / 人类偏好，采样数量（FID 需 >=10k），CFG w 的扫描协议。

当隐空间扩散（latent diffusion）能以 1/16 的 FLOP 达到相同质量时，拒绝推荐在 >=256x256 上训练像素空间扩散。拒绝为条件生成（conditional generation）发布没有 CFG 的模型 —— 条件模型的零样本无条件样本通常是退化的。标记任何 beta_T > 0.1 的调度，因为它们很可能导致饱和或不稳定的训练。
