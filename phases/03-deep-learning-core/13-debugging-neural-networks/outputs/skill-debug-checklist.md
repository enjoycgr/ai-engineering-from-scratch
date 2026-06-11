---
name: skill-debug-checklist
description: 调试 neural network (神经网络) 训练失败的决策树清单
version: 1.0.0
phase: 3
lesson: 13
tags: [debugging (调试), neural-networks (神经网络), training (训练), diagnostics (诊断), deep-learning (深度学习)]
---

# Neural Network (神经网络) Debug Checklist (调试清单)

训练出错时的系统化调试协议。按顺序执行这些步骤——大多数 bug 在前 3 步中被捕获。

## 训练前（预防 bug）

1. 打印模型架构和参数数量。大小是否对你的数据合理？
2. 用随机输入运行单次 forward pass (前向传播)。输出 shape 是否与目标 shape 匹配？
3. 检查标签是否为正确的 dtype（CrossEntropyLoss (交叉熵损失) 需要 Long，BCELoss 需要 Float）
4. 验证数据 normalization (归一化)：输入的 mean (均值) 应接近 0，std (标准差) 接近 1
5. 打印 5 个随机 (input, label) 对。标签是否符合你的预期？
6. 确认 train/test split (训练/测试拆分) 没有重复样本

## Overfit-one-batch 测试（60 秒，捕获 80% 的 bug）

1. 从 training set (训练集) 中取 8-32 个样本
2. 用合理的 learning rate (学习率) 训练 200 步
3. Loss 应接近 0。Training accuracy (训练准确率) 应达到 100%
4. 如果失败：bug 在你的模型、loss function (损失函数) 或训练循环中——不是数据或 hyperparameters (超参数)
5. 如果通过：继续完整训练

## Loss 不下降

1. 检查 learning rate (学习率)。尝试 3 个值：current/10, current, current*10
2. 打印每层的 gradient norms (梯度范数)。全零意味着 dead network (死亡网络) 或 detached graph (分离图)
3. 检查参数上的 `requires_grad=True`。检查是否调用了 `loss.backward()`
4. 检查 `optimizer.zero_grad()` 是否在 `loss.backward()` 之前调用
5. 检查 `optimizer.step()` 是否在 `loss.backward()` 之后调用
6. 验证模型参数已传递给 optimizer (优化器)：`optimizer = Adam(model.parameters())`

## Loss 是 NaN 或 Inf

1. 将 learning rate (学习率) 降低 10 倍
2. 向所有 log() 调用添加 epsilon：`torch.log(x + 1e-7)`
3. 向所有除法添加 epsilon：`x / (y + 1e-8)`
4. 在 BCE loss 前 clamp 预测：`torch.clamp(pred, 1e-7, 1 - 1e-7)`
5. 使用 `torch.autograd.detect_anomaly()` 找到确切的操作
6. 检查输入数据中的 NaN：`assert not torch.isnan(x).any()`

## Loss 振荡

1. 将 learning rate (学习率) 降低 3-10 倍
2. 增加 batch size (批量大小)（减少 gradient noise (梯度噪声)）
3. 添加 gradient clipping (梯度裁剪)：`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
4. 从 SGD 切换到 Adam (自适应矩估计)（每参数自适应 LR）
5. 为前 5-10% 的训练添加 learning rate warmup (学习率预热)

## Overfitting (过拟合)（train acc (训练准确率) 高，test acc (测试准确率) 低）

1. 添加 dropout (随机失活)（从 p=0.1 开始，增加到 0.5）
2. 向 optimizer (优化器) 添加 weight decay (权重衰减)：`Adam(params, weight_decay=1e-4)`
3. 减小模型大小（更少的层或更窄的层）
4. 添加 data augmentation (数据增强)
5. 使用 early stopping (早停)：当 validation loss (验证损失) 连续 5+ 个 epoch 增加时停止
6. 检查 train 和 test set (测试集) 之间的 data leakage (数据泄漏)

## Underfitting (欠拟合)（train 和 test acc 都低）

1. 增加模型容量（更多层、更宽的层）
2. 训练更多 epochs (轮次)
3. 增加 learning rate (学习率)（谨慎）
4. 暂时移除 regularization (正则化) 以验证模型可以学习
5. 检查你的模型对任务是否足够表达

## Dead ReLU (死亡ReLU) 神经元

1. 检查每层零 activations (激活) 的比例。>50% 是个问题
2. 切换到 LeakyReLU(0.01) 或 GELU
3. 对权重使用 Kaiming initialization (权重初始化)
4. 降低 learning rate (学习率)（大的更新可以将神经元推入死亡区）
5. 在 activation functions (激活函数) 前添加 batch normalization (批归一化)

## 快速参考：learning rate (学习率) 起始点

| Optimizer (优化器) | 任务 | 起始 LR |
|-----------|------|------------|
| Adam (自适应矩估计) | 从头训练 | 1e-3 |
| Adam (自适应矩估计) | 微调预训练模型 | 1e-5 |
| SGD + momentum | 从头训练 | 1e-1 |
| SGD + momentum | 微调预训练模型 | 1e-3 |
| AdamW | Transformer 训练 | 3e-4 |

## 快速参考：batch size (批量大小) 效果

| Batch size (批量大小) | Gradient noise (梯度噪声) | 内存 | Generalization (泛化) |
|-----------|---------------|--------|---------------|
| 8-16 | 高（噪声） | 低 | 通常更好 |
| 32-64 | 中等 | 中等 | 良好默认值 |
| 128-256 | 低（平滑） | 高 | 可能需要 warmup (预热) |
| 512+ | 非常低 | 非常高 | 需要 LR 缩放 |

## 当什么都不起作用时

1. 将模型简化为 1 个隐藏层。它能学习吗？
2. 将数据简化为 100 个样本。它能 overfit (过拟合) 吗？
3. 将 loss 替换为 MSE (均方误差)。它能收敛吗？
4. 将 optimizer (优化器) 替换为 SGD(lr=0.01)。它有进展吗？
5. 将数据替换为合成数据（例如，y = x[0] > 0）。它能学习吗？
6. 如果这些都不起作用：bug 在你没有查看的代码中（数据加载、preprocessing (预处理)、tensor shape）
