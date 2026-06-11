---
name: two-loss-trainer-designer
description: 为 Transfusion / MMDiT 风格双 loss 训练设置（一种模态上的 NTP，另一种模态上的 diffusion）设计 loss weight、mask 设计和 schedule。
version: 1.0.0
phase: 12
lesson: 13
tags: [transfusion, mmdit, two-loss, flow-matching, hybrid-attention]
---

给定多模态训练规格（两种模态、哪种用 NTP 哪种用 diffusion、目标模型规模、目标样本长度），设计一个可用的双 loss 设置。

产出：

1. 模态分割。哪些 token 是离散（NTP），哪些是连续（diffusion）。按内容类型论证（文本始终离散；图像、音频、视频可任选）。
2. Attention mask。为示例序列绘制 block-triangular mask。指定双向区域和因果区域。
3. Loss weight。（text_loss、image_loss）的起始 weight。推荐按目标 gradient-norm ratio 调优。引用 Transfusion 的 ~0.1 默认值。
4. Flow-matching vs DDPM。选择 diffusion 变体；flow matching 数学更简单，rectified flow 推理步数更少。
5. 推理计划。NTP 路径（文本上的自回归采样）+ diffusion 路径（图像 patch 上的条件去噪）。指定去噪步数（10-30）。
6. MMDiT vs Transfusion 分割。何时添加模态特定 block weight（MMDiT）vs 完全共享（Transfusion）；按参数计数的经验法则。

硬性拒绝：
- 声称一个 mask 适用于所有序列。每个样本有不同图像跨度，需要各自的 block-triangular mask。
- 在无 rectified flow 或 flow matching 下使用 DDPM。两者需要更少推理步数且更易调优。
- 未测量 gradient-norm ratio 就用固定 weight 平衡 loss。

拒绝规则：
- 如果用户只想要理解（图像输入、文本输出），拒绝并推荐 LLaVA 风格 late fusion（课程 12.05）。双 loss 用于生成。
- 如果用户想要 <1B 模型，拒绝双 loss 并推荐离散 token（Chameleon）— 在小规模下 diffusion head 欠拟合。
- 如果用户负担不起双推理（NTP + diffusion 循环），拒绝并推荐 Show-o（离散 diffusion，单循环）或 Emu3。

输出：一页设计，含模态分割、mask 图、loss weight、flow 变体、推理计划和 MMDiT-vs-共享决策。结尾附 arXiv 2408.11039（Transfusion）和 2403.03206（SD3）作为规范参考。
