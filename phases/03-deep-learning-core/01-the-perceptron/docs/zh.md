# 感知器（Perceptron）

> 感知器（perceptron）是神经网络的“原子”。把它拆开，你会发现 weight（权重）、bias（偏置）和一个决策。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 学习目标

- 用 Python 从零实现一个感知器（perceptron），包括 weight update rule（权重更新规则）和 step activation function（阶跃激活函数）
- 解释为什么单个感知器只能解决 linearly separable（线性可分）问题，并演示 XOR 失败的情况
- 通过组合 OR gate（或门）、NAND gate（与非门）和 AND gate（与门）构建 multi-layer perceptron（多层感知器）来解决 XOR
- 使用 sigmoid 激活函数和 backpropagation（反向传播）训练一个两层网络，自动学习 XOR

## 问题

你已经了解向量和点积。你知道矩阵可以把输入变换成输出。但机器是如何*学习*该使用哪种变换的呢？

感知器回答了这个问题。它是最简单的学习机器：接收若干输入，乘以 weight（权重），加上 bias（偏置），做出一个二分类决策。然后调整。仅此而已。每一个神经网络都是这一思想层层堆叠的结果。

理解感知器意味着理解“学习”在代码中的真正含义：不断调整数字，直到输出与现实相符。

## 概念

### 一个神经元，一个决策

感知器接收 n 个输入，将每个输入乘以对应的 weight（权重），求和，加上 bias（偏置），然后将结果传入 activation function（激活函数）。

```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```

step function（阶跃函数）非常直接：如果 weighted sum（加权和）加上 bias 大于等于 0，输出 1；否则输出 0。

```
step(z) = 1  if z >= 0
           0  if z < 0
```

这是一个 linear classifier（线性分类器）。weight 和 bias 定义了一条直线（或高维空间中的超平面），将输入空间分成两个区域。

### 决策边界（Decision Boundary）

对于两个输入，感知器在二维空间中画出一条直线：

```
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```

直线一侧的所有点输出 0，另一侧输出 1。训练过程会移动这条直线，直到它能正确分开两类。

### 学习规则（Learning Rule）

感知器学习规则很简单：

```
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```

如果预测正确，error = 0，什么都不变。如果预测为 0 但应该是 1，weight 增加。如果预测为 1 但应该是 0，weight 减少。learning rate（学习率）控制每次调整的幅度。

### XOR 问题

这里是感知器的局限。看看这些逻辑门：

```
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```

AND 和 OR 是 linearly separable（线性可分）的：你可以画一条直线把 0 和 1 分开。XOR 不是。没有任何一条直线能把 [0,1] 和 [1,0] 与 [0,0] 和 [1,1] 分开。

```
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```

这是一个根本性的限制。单个感知器只能解决 linearly separable 问题。Minsky 和 Papert 在 1969 年证明了这一点，几乎让神经网络研究停滞了十年。

解决方案：把感知器堆叠成层。multi-layer perceptron（多层感知器）可以通过组合两个线性决策来形成非线性决策，从而解决 XOR。

## 动手实现

### 步骤 1：Perceptron 类

```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```

### 步骤 2：训练逻辑门

```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```

### 步骤 3：观察 XOR 失败

```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```

它永远不会收敛。这是单个感知器无法学习 XOR 的硬核证明。

### 步骤 4：用两层网络解决 XOR

技巧：XOR = (x1 OR x2) AND NOT (x1 AND x2)。组合三个感知器：

```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```

四个用例全部正确。将感知器堆叠成层可以创造出单个感知器无法产生的 decision boundary（决策边界）。

### 步骤 5：训练一个两层网络

步骤 4 手动设置了 weight。这对 XOR 有效，但对不知道正确 weight 的真实问题不适用。解决方案：用 sigmoid 替换 step function，并通过 backpropagation 自动学习 weight。

```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```

与步骤 4 相比有两个关键区别。首先，sigmoid 替代了 step function —— 它是平滑的，因此梯度存在。其次，`train` 方法将误差从输出层传播回隐藏层，按照每个 weight 对误差的贡献比例进行调整。这就是 20 行代码实现的 backpropagation（反向传播）。

这是通往第 03 课的桥梁。`d_output` 和 `hidden_deltas` 背后的数学是将链式法则应用于网络图。我们将在那里正式推导它。

## 使用现成的库

你刚才从零构建的所有内容，只需一行导入即可使用：

```python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```

五行代码。你写的 30 行 `Perceptron` 类做的是同样的事情。sklearn 版本增加了收敛检查、多种 loss function（损失函数）和稀疏输入支持 —— 但核心循环完全相同：weighted sum（加权和）、step function、误差时更新 weight。

真正的差距体现在规模上。生产级网络中的变化：

- step function 变成 sigmoid、ReLU 或其他平滑的 activation function
- weight 通过 backpropagation（第 03 课）自动学习
- 层数更深：3 层、10 层、100+ 层
- 相同的原则仍然成立：每一层根据前一层的输出创建新的特征

单个感知器只能画直线。把它们堆叠起来，你可以画出任何形状。

## 产出

本课产出：
- `outputs/skill-perceptron.md` - 一个技能文档，涵盖何时需要单层架构、何时需要多层架构

## 练习

1. 在 NAND gate 上训练一个感知器（NAND 是通用门 —— 任何逻辑电路都可以仅用 NAND 构建）。验证其 weight 和 bias 是否形成了有效的 decision boundary。
2. 修改 Perceptron 类，使其在每个 epoch 跟踪 decision boundary（w1*x1 + w2*x2 + b = 0）。打印训练 AND gate 时这条线是如何移动的。
3. 构建一个 3 输入感知器，仅当至少 2 个输入为 1 时输出 1（多数表决函数）。它是 linearly separable 的吗？为什么？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| Perceptron | “一个假神经元” | 一个 linear classifier（线性分类器）：输入与 weight 的点积，加上 bias，再通过 step function |
| Weight | “输入有多重要” | 一个乘数，缩放每个输入对决策的贡献 |
| Bias | “阈值” | 一个常数，平移 decision boundary，让感知器即使在输入全为零时也能激活 |
| Activation function | “把值压扁的东西” | 应用于 weighted sum 之后的函数 —— 感知器用 step function，现代网络用 sigmoid/ReLU |
| Linearly separable | “你可以画一条线把它们分开” | 一个数据集，可以用单个超平面完美分开两类 |
| XOR problem | “感知器做不到的事” | 证明单层网络无法学习非线性可分函数 |
| Decision boundary | “分类器切换的地方” | 将输入空间分成两类的超平面 w*x + b = 0 |
| Multi-layer perceptron | “真正的神经网络” | 感知器逐层堆叠，每一层的输出作为下一层的输入 |

## 延伸阅读

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) —— 一切的起源，原始论文
- Minsky & Papert, "Perceptrons" (1969) —— 证明了 XOR 无法被单层网络解决，并让感知器研究停滞了十年的著作
- Michael Nielsen, "Neural Networks and Deep Learning", Chapter 1 (http://neuralnetworksanddeeplearning.com/) —— 免费在线阅读，对感知器如何组合成网络的最佳可视化解释
