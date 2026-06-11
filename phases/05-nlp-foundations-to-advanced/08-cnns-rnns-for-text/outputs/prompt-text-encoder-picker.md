---
name: text-encoder-picker
description: 根据给定约束集选择文本编码器架构。
phase: 5
lesson: 08
---

给定约束（任务、数据量、延迟预算、部署目标、计算预算），输出：

1. 编码器架构（Encoder architecture）：TextCNN、BiLSTM、BiLSTM-CRF、transformer fine-tune（transformer 微调），或"使用预训练 transformer 作为冻结编码器 + 小型分类头"。
2. Embedding 输入（Embedding input）：随机初始化、冻结的 GloVe / fastText，或上下文相关的 transformer embedding（上下文嵌入）。
3. 五行训练配方（Training recipe）：optimizer、learning rate（学习率）、batch size（批量大小）、epoch（轮次）、regularization（正则化）。
4. 一个监控信号（One monitoring signal）。RNN/CNN 模型：检查按序列长度划分的准确率，以发现长距离依赖失败。Transformer 微调：注意学习率过高时 fine-tuning（微调）会崩溃；在前 100 步内检查训练损失。

当标注样本少于约 500 条时，拒绝推荐 fine-tuning transformer，除非先证明 TextCNN / BiLSTM 基线已经饱和。标记边缘部署（手机、微控制器、浏览器）需要把架构决策放在一切之前。
