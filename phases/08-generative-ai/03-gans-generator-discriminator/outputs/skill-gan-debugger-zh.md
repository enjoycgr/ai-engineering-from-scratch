---
name: gan-debugger
description: 从损失曲线和样本网格诊断失败的 GAN 训练；给出单行修复方案。
version: 1.0.0
phase: 8
lesson: 03
tags: [gan, adversarial, debugging]
---

给定一个失败的 GAN 运行（D 和 G 损失曲线、样本网格、数据集大小、优化器配置），输出：

1. 诊断。一个根因，选自：模式崩溃 (mode collapse)、D 太强、D 太弱、梯度消失 (vanishing gradient)、批量归一化 (batch norm) 泄露、D 过拟合、学习率不匹配、初始化不良。
2. 证据。指向损失曲线或样本中征兆的指针（例如 "D(fake) < 0.05 在第 500 步 = D 太强"）。
3. 修复。一个具体的改动。例如：`lr_D = lr_G / 2`，用实例归一化 (instance norm, IN) 替换 BN，对 D 添加谱归一化 (spectral norm)，切换到 lambda=10 的 WGAN-GP，批量大小减半，向 D 输入添加 0.1 高斯噪声。
4. 重跑协议。要尝试的随机种子、重新评估前的步数、接受标准（例如 "FID 在第 20k 步降至基线以下"）。
5. 回退方案。如果修复一次重跑未能成功，接下来尝试什么。通常是：切换架构（StyleGAN、R3GAN）或切换范式（diffusion、flow matching），如果数据集过于多样。

当 D 已经饱和时，拒绝建议增加 G 的学习率。当真正的失败是 D 时，拒绝向 G 添加正则化 (regularization)——先修复 D。标记任何在 100 步内出现训练崩溃的运行为可能是初始化不良或学习率爆炸，而非深层算法问题。
