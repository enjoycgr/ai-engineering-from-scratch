# 构建你自己的 Mini-Framework (迷你框架)

> 你已经构建了神经元、层、网络、反向传播、激活函数、损失函数、优化器、正则化、初始化和学习率调度。都是独立的模块。现在把它们组装成一个框架。不是 PyTorch，不是 TensorFlow，是你自己的。

**类型:** 构建
**语言:** Python
**前置知识:** Phase 03 全部课程 (Lessons 01-09)
**时间:** ~120 分钟

## 学习目标

- 构建一个完整的深度学习框架 (~500 行)，包含 Module、Linear、ReLU、Sigmoid、Dropout、BatchNorm、Sequential、loss functions (损失函数)、optimizers (优化器) 和 DataLoader
- 解释 Module (模块) 抽象（forward、backward、parameters）以及为什么需要 train/eval 模式切换
- 将所有组件连接成一个可工作的训练循环，在 circle classification (圆形分类) 上训练一个 4 层网络
- 将框架的每个组件映射到对应的 PyTorch 等效实现（nn.Module、nn.Sequential、optim.Adam、DataLoader）

## 问题

你有十节课的构建模块分散在不同文件中。一个 `Value` 类在这里，一个训练循环在那里，权重初始化在另一个文件，学习率调度又在另一个。要训练一个网络，你得从五个不同课程中复制粘贴并手动连接。

这就是框架要解决的问题。PyTorch 提供 `nn.Module`、`nn.Sequential`、`optim.Adam`、`DataLoader` 以及将它们连接起来的训练循环模式。TensorFlow 提供 `keras.Layer`、`keras.Sequential`、`keras.optimizers.Adam`。这些不是魔法，它们是组织模式，让你无需每次都重新发明底层机制就能定义、训练和评估网络。

你将用约 500 行 Python 构建同样的东西。不用 numpy，没有外部依赖。一个可以定义任何前馈网络、用 SGD 或 Adam 训练、批处理数据、应用 dropout 和 batch normalization (批归一化)、使用任意激活函数、并调度学习率的框架。

完成后，当你写 PyTorch 的 `model = nn.Sequential(...)` 时，你会完全理解背后发生了什么。你会理解为什么 `model.train()` 和 `model.eval()` 存在。你会理解为什么 `optimizer.zero_grad()` 是一个单独的调用。你会理解所有这一切，因为是你亲手构建了它们。

## 概念

### Module (模块) 抽象

PyTorch 中的每个 layer (层) 都继承自 `nn.Module`。Module 有三个职责：

1. **forward()** -- 给定输入计算输出
2. **parameters()** -- 返回所有可训练的 weight (权重)
3. **backward()** -- 计算 gradient (梯度)（PyTorch 中由 autograd (自动微分) 处理，我们的框架中显式实现）

Linear layer (线性层) 是一个 Module。ReLU 激活是一个 Module。Dropout layer 是一个 Module。Batch normalization layer 是一个 Module。它们都有相同的接口。

### Sequential (顺序容器)

`nn.Sequential` 用于串联 Modules。Forward pass (前向传播)：数据依次流经 Module 1、Module 2、Module 3。Backward pass (反向传播)：逆序遍历。容器本身也是一个 Module——它有 forward()、parameters() 和 backward()。这就是组合模式：一系列 Modules 本身也是一个 Module。

### 训练模式 vs 评估模式

Dropout 在训练时随机将神经元置零，但在评估时全部通过。Batch normalization (批归一化) 在训练时使用 batch 统计量，在评估时使用 running average (滑动平均)。`train()` 和 `eval()` 方法切换这种行为。每个 Module 都有一个 `training` 标志。

### Optimizer (优化器)

Optimizer 使用 gradient (梯度) 更新 parameter (参数)。SGD (随机梯度下降): `param -= lr * grad`。Adam (自适应矩估计): 维护 momentum (动量) 和 variance (方差) 估计，然后更新。Optimizer 不了解网络架构——它只看到 parameter 和 gradient 的扁平列表。

### DataLoader (数据加载器)

Batching (批处理) 重要有两个原因。首先，对于大规模问题你无法把整个数据集放入内存。其次，mini-batch gradient descent (小批量梯度下降) 提供的噪声有助于逃离局部最小值。DataLoader 将数据分成 batch (批次)，并可选地在每个 epoch (轮次) 之间 shuffle (打乱)。

### 框架架构

```mermaid
graph TD
    subgraph "Modules (模块)"
        Linear["Linear (线性层)<br/>W*x + b"]
        ReLU["ReLU (修正线性单元)<br/>max(0, x)"]
        Sigmoid["Sigmoid (S型函数)<br/>1/(1+e^-x)"]
        Dropout["Dropout<br/>random zero mask (随机置零掩码)"]
        BatchNorm["BatchNorm (批归一化)<br/>normalize activations (归一化激活值)"]
    end

    subgraph "Containers (容器)"
        Sequential["Sequential (顺序容器)<br/>chains modules (串联模块)"]
    end

    subgraph "Loss Functions (损失函数)"
        MSE["MSELoss<br/>(pred - target)^2"]
        BCE["BCELoss<br/>binary cross-entropy (二元交叉熵)"]
    end

    subgraph "Optimizers (优化器)"
        SGD["SGD (随机梯度下降)<br/>param -= lr * grad"]
        Adam["Adam (自适应矩估计)<br/>adaptive moments"]
    end

    subgraph "Data (数据)"
        DataLoader["DataLoader (数据加载器)<br/>batching + shuffle (批处理 + 打乱)"]
    end

    Sequential --> |"contains (包含)"| Linear
    Sequential --> |"contains (包含)"| ReLU
    Sequential --> |"forward/backward (前向/反向)"| MSE
    SGD --> |"updates (更新)"| Sequential
    DataLoader --> |"feeds (输入)"| Sequential
```

### 训练循环

```mermaid
sequenceDiagram
    participant DL as DataLoader (数据加载器)
    participant M as Model (模型)
    participant L as Loss (损失)
    participant O as Optimizer (优化器)

    loop Each Epoch (每轮)
        DL->>M: batch of inputs (一批输入)
        M->>M: forward pass (前向传播, 逐层)
        M->>L: predictions (预测)
        L->>L: compute loss (计算损失)
        L->>M: backward pass (反向传播, 梯度)
        M->>O: parameters + gradients (参数 + 梯度)
        O->>M: updated parameters (更新后的参数)
        O->>O: zero gradients (梯度清零)
    end
```

### Module 层级结构

```mermaid
classDiagram
    class Module {
        +forward(x)
        +backward(grad)
        +parameters()
        +train()
        +eval()
    }

    class Linear {
        -weights (权重)
        -biases (偏置)
        +forward(x)
        +backward(grad)
    }

    class ReLU {
        +forward(x)
        +backward(grad)
    }

    class Sequential {
        -modules[] (模块列表)
        +forward(x)
        +backward(grad)
        +parameters()
    }

    Module <|-- Linear
    Module <|-- ReLU
    Module <|-- Sequential
    Sequential *-- Module
```

## 构建它

### 第 1 步: Module 基类

每个 layer 实现的抽象接口。

```python
class Module:
    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
```

### 第 2 步: Linear Layer (线性层)

基础构建块。存储 weight (权重) 和 bias (偏置)，前向计算 Wx + b，反向计算 weight/input gradient (权重/输入梯度)。

```python
import math
import random


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params
```

### 第 3 步: Activation Modules (激活模块)

ReLU、Sigmoid 和 Tanh 作为 Modules。每个缓存 backward pass 所需的内容。

```python
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        return [g * m for g, m in zip(grad, self.mask)]


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = []
        for v in x:
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output

    def backward(self, grad):
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output

    def backward(self, grad):
        return [g * (1 - o * o) for g, o in zip(grad, self.output)]
```

### 第 4 步: Dropout Module

训练时随机将元素置零。将剩余元素缩放 1/(1-p) 以保持期望值不变。评估时不做任何操作。

```python
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]
```

### 第 5 步: BatchNorm Module (批归一化模块)

将 batch 中每个 feature (特征) 的激活值归一化为零均值和单位方差。为 eval 模式维护 running statistics (滑动统计量)。

```python
class BatchNorm(Module):
    def __init__(self, size, momentum=0.1, eps=1e-5):
        super().__init__()
        self.size = size
        self.gamma = [1.0] * size
        self.beta = [0.0] * size
        self.gamma_grads = [0.0] * size
        self.beta_grads = [0.0] * size
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = momentum
        self.eps = eps
        self.x_norm = None
        self.std_inv = None
        self.batch_input = None

    def forward_batch(self, batch):
        batch_size = len(batch)
        output_batch = []

        if self.training:
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in var]

            self.x_norm = []
            self.batch_input = batch
            for sample in batch:
                normed = [(sample[j] - mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

        return output_batch

    def forward(self, x):
        result = self.forward_batch([x])
        return result[0]

    def backward(self, grad):
        if self.x_norm is None:
            return grad
        for j in range(self.size):
            self.gamma_grads[j] += self.x_norm[0][j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]

    def parameters(self):
        params = []
        for j in range(self.size):
            params.append((self.gamma, j, None, self.gamma_grads))
            params.append((self.beta, j, None, self.beta_grads))
        return params
```

### 第 6 步: Sequential Container (顺序容器)

串联 modules。Forward 从左到右，backward 从右到左。

```python
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def eval(self):
        self.training = False
        for module in self.modules:
            module.eval()
```

### 第 7 步: Loss Functions (损失函数)

MSE 和 Binary Cross-Entropy (二元交叉熵)。每个返回 loss 值并提供 backward() 返回 gradient。

```python
class MSELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        n = len(predicted)
        self.loss = sum((p - t) ** 2 for p, t in zip(predicted, target)) / n
        return self.loss

    def backward(self):
        n = len(self.predicted)
        return [2 * (p - t) / n for p, t in zip(self.predicted, self.target)]


class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            p = max(eps, min(1 - eps, p))
            self.loss += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        self.loss /= n
        return self.loss

    def backward(self):
        eps = 1e-7
        n = len(self.predicted)
        grads = []
        for p, t in zip(self.predicted, self.target):
            p = max(eps, min(1 - eps, p))
            grads.append((-t / p + (1 - t) / (1 - p)) / n)
        return grads
```

### 第 8 步: SGD 和 Adam Optimizers (优化器)

两者都接收 parameter 列表并使用 gradient 更新 weight。

```python
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [0.0] * len(parameters)
        self.v = [0.0] * len(parameters)

    def step(self):
        self.t += 1
        for idx, (container, i, j, grad_container) in enumerate(self.params):
            if j is not None:
                g = grad_container[i][j]
            else:
                g = grad_container[i]

            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g

            m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
            v_hat = self.v[idx] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

            if j is not None:
                container[i][j] -= update
            else:
                container[i] -= update

    def zero_grad(self):
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0
```

### 第 9 步: DataLoader (数据加载器)

将数据分成 batch，可选地在每个 epoch 之间 shuffle。

```python
class DataLoader:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs, targets

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size
```

### 第 10 步: 在 Circle Classification (圆形分类) 上训练一个 4 层网络

将所有东西连接起来。定义模型、选择 loss、选择 optimizer、运行训练循环。

```python
def make_circle_data(n=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], [label]))
    return data


def train():
    random.seed(42)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Linear(16, 16),
        ReLU(),
        Linear(16, 8),
        ReLU(),
        Linear(8, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    data = make_circle_data(500)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]

    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(100):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            batch_loss = 0
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                batch_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

                predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
                if predicted_class == t[0]:
                    total_correct += 1
                total_samples += 1

            total_loss += batch_loss

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%")

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    test_accuracy = correct / len(test_data) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_data)})")

    return model, test_accuracy
```

## 使用它

以下是你刚才构建的内容的 PyTorch 等效实现：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_predictions = model(test_inputs)
```

结构完全相同。`Sequential`、`Linear`、`ReLU`、`Sigmoid`、`BCELoss`、`Adam`、`zero_grad`、`backward`、`step`、`train`、`eval`。每个概念都是一一对应的。区别在于 PyTorch 自动处理 autograd (无需在每个 module 中实现 backward())、在 GPU 上运行、并经过多年优化。但骨架是一样的。

现在当你看到 PyTorch 代码时，你确切知道每一行背后发生了什么。这种理解就是本课的全部意义。

## 交付物

本课产出：
- `outputs/prompt-framework-architect.md` -- 一个使用框架抽象设计神经网络架构的 prompt

## 练习

1. 添加一个 `SoftmaxCrossEntropyLoss` 类用于多分类。对预测值应用 Softmax (软最大值)，计算 cross-entropy (交叉熵) loss，并处理组合的 backward pass。在 3-class spiral dataset (3 类螺旋数据集) 上测试它。

2. 在 optimizer 中实现 learning rate scheduling (学习率调度)：添加 `set_lr()` 方法并接入 Lesson 09 的 cosine schedule (余弦调度)。用 warmup + cosine 训练 circle classifier (圆形分类器) 并与恒定 LR 比较。

3. 给 Sequential 添加 `save()` 和 `load()` 方法，将所有 weight 序列化到 JSON 文件并加载回来。验证加载的模型与原始模型产生相同的预测。

4. 在 Adam optimizer 中实现 weight decay (权重衰减, L2 正则化)。添加一个 `weight_decay` 参数，每步将 weight 向零收缩。比较 decay=0 和 decay=0.01 的训练效果。

5. 将逐样本训练循环替换为正确的 mini-batch gradient accumulation (小批量梯度累积)：在 batch 中所有样本上累积 gradient，然后除以 batch size (批量大小) 并执行一次 optimizer step。测量这是否改变收敛速度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Module (模块) | "一个层" | 框架中的基础抽象——任何具有 forward()、backward() 和 parameters() 的东西 |
| Sequential (顺序容器) | "按顺序堆叠层" | 一个串联 modules 的容器，forward 按顺序应用，backward 逆序应用 |
| Forward pass (前向传播) | "运行网络" | 按顺序将输入通过每个 module 计算输出 |
| Backward pass (反向传播) | "计算梯度" | 将 loss gradient 反向传播通过每个 module 以计算 parameter gradient |
| Parameters (参数) | "可训练权重" | 网络中 optimizer 可以更新的所有值——weight 和 bias |
| Optimizer (优化器) | "更新权重的东西" | 使用 gradient 更新 parameter 的算法，实现 SGD、Adam 或其他规则 |
| DataLoader (数据加载器) | "喂数据的东西" | 将数据集拆分为 batch 的迭代器，可选地在 epoch 之间 shuffle |
| Training mode (训练模式) | "model.train()" | 一个标志，启用 dropout 等随机行为，batch normalization 使用 batch 统计量 |
| Evaluation mode (评估模式) | "model.eval()" | 一个标志，禁用 dropout，batch normalization 使用 running statistics (滑动统计量) |
| Zero grad (梯度清零) | "清空梯度" | 在计算下一 batch 的 gradient 之前，将所有 parameter gradient 重置为零 |

## 延伸阅读

- Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019) -- 描述 PyTorch 设计决策的论文
- Chollet, "Deep Learning with Python, Second Edition" (2021) -- 第 3 章用相同的 module/layer 抽象讲解 Keras 内部原理
- Johnson, "Tiny-DNN" (https://github.com/tiny-dnn/tiny-dnn) -- 一个仅头文件的 C++ 深度学习框架，用于理解框架内部机制
