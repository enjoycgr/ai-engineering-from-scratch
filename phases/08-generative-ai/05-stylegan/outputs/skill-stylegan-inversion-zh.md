---
name: stylegan-inversion
description: 为真实照片选择预训练 StyleGAN 的反演和编辑管线。
version: 1.0.0
phase: 8
lesson: 05
tags: [stylegan, inversion, editing]
---

给定一张真实照片 + 预训练 StyleGAN 检查点（FFHQ-1024、StyleGAN-XL、自定义微调）以及目标编辑（年龄、微笑、姿态、头发、身份保留），输出：

1. 反演方法。e4e（快、保真度低）、ReStyle（迭代编码器）、HyperStyle（超网络）、PTI（关键调优）或直接 W-优化。一句话说明原因，关联保真度 vs 速度。
2. 目标空间。W、W+ 或 StyleSpace。权衡：W = 最解耦但保真度最低，W+ = 每层 w，StyleSpace = 通道级。
3. 编辑方向。命名方向来源：InterFaceGAN（基于 SVM）、StyleSpace 通道、GANSpace PCA 或学习分类器。
4. 保真度预算。身份漂移前的 LPIPS 阈值；回退启发式。
5. 评估。ID 相似度（ArcFace 余弦）、与原始图像的 LPIPS、编辑强度（目标属性分类器分数）。

拒绝任何直接在 Z 中编辑的管线（纠缠）。拒绝大范围编辑（&gt;1.5 sigma in W）而不进行身份检查。标记需要开放域编辑的请求（例如"把他变成卡通"）——那些需要扩散 + IP-Adapter，而不是 StyleGAN。
