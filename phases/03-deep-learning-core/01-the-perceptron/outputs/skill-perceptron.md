---
name: skill-perceptron
description: 理解感知器（perceptron）模式，以及何时使用单层架构、何时使用多层架构
version: 1.0.0
phase: 3
lesson: 1
tags: [perceptron, neural-networks, classification, deep-learning]
---

# 感知器模式（The Perceptron Pattern）

感知器（perceptron）计算输入的 weighted sum（加权和）加上 bias（偏置），然后应用 step function（阶跃函数）产生二分类输出。它是神经网络的基本单元。

```
output = step(w1*x1 + w2*x2 + ... + wn*xn + bias)
```

## 单个感知器足够的情况

- 问题是 linearly separable（线性可分）的：一条直线（或超平面）可以分开两类
- 逻辑门：AND gate（与门）、OR gate（或门）、NOT gate（非门）、NAND gate（与非门）
- 简单的阈值决策：“分数是否高于 X？”
- 数据聚成两个不重叠区域的二分类器

## 需要多层网络的情况

- 问题不是 linearly separable 的：没有一条直线能分开两类
- XOR gate（异或门）和奇偶校验问题
- 任何需要“是这个但不是那个”的推理任务（条件组合）
- 现实世界分类：图像、文本、音频——几乎都是非线性的

## 决策检查清单

1. 绘制或检查你的数据。你能用一条直线在两类之间画出边界吗？
   - 是：单个感知器即可工作
   - 否：你至少需要两层
2. 问题能否分解为更简单线性决策的 AND/OR 组合？
   - 这种分解告诉你最小网络结构
   - XOR = (A OR B) AND (NOT (A AND B)) = 2 层中的 3 个感知器
3. 对于多于两类的问题，每类需要一个输出节点

## 训练规则

```
error = expected - predicted
weight_new = weight_old + learning_rate * error * input
bias_new = bias_old + learning_rate * error
```

如果预测正确，什么都不变。如果错误，weight 会移动以减少误差。这仅适用于单层感知器。多层网络需要 backpropagation（反向传播）。

## 常见错误

- 试图用单个感知器学习非线性模式（它永远不会收敛）
- learning rate（学习率）设置太高（weight 振荡）或太低（训练耗时过长）
- 忘记 bias 项（没有它，decision boundary（决策边界）必须穿过原点）
- 混淆感知器收敛（对 linearly separable 数据有保证）与一般神经网络的收敛（不保证）
