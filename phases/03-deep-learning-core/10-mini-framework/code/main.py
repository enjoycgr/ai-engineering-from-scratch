import math
import random


# Module (模块) 基类：所有层和容器的抽象接口
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


# Linear layer (线性层)：仿射变换 y = W*x + b
class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        # Kaiming 初始化：std = sqrt(2 / fan_in)，适用于 ReLU
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[random.gauss(0, std) for _ in range(fan_in)] for _ in range(fan_out)]
        self.biases = [0.0] * fan_out
        self.weight_grads = [[0.0] * fan_in for _ in range(fan_out)]
        self.bias_grads = [0.0] * fan_out
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.input = None

    def forward(self, x):
        # 缓存输入用于 backward pass (反向传播)
        self.input = x
        output = []
        for i in range(self.fan_out):
            val = self.biases[i]
            for j in range(self.fan_in):
                val += self.weights[i][j] * x[j]
            output.append(val)
        return output

    def backward(self, grad):
        # 计算 weight gradient (权重梯度)、bias gradient (偏置梯度) 和 input gradient (输入梯度)
        input_grad = [0.0] * self.fan_in
        for i in range(self.fan_out):
            self.bias_grads[i] += grad[i]
            for j in range(self.fan_in):
                self.weight_grads[i][j] += grad[i] * self.input[j]
                input_grad[j] += grad[i] * self.weights[i][j]
        return input_grad

    def parameters(self):
        # 返回所有可训练的 parameter (参数) 及其对应的 gradient (梯度)
        params = []
        for i in range(self.fan_out):
            for j in range(self.fan_in):
                params.append((self.weights, i, j, self.weight_grads))
            params.append((self.biases, i, None, self.bias_grads))
        return params


# ReLU (修正线性单元)：max(0, x)
class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        # 缓存 mask 用于 backward pass
        self.mask = [1.0 if v > 0 else 0.0 for v in x]
        return [max(0.0, v) for v in x]

    def backward(self, grad):
        # 梯度通过 mask：正数位置原样传递，负数位置置零
        return [g * m for g, m in zip(grad, self.mask)]


# Sigmoid (S型函数)：1 / (1 + exp(-x))
class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = []
        for v in x:
            # 裁剪防止 exp 溢出
            v = max(-500, min(500, v))
            self.output.append(1.0 / (1.0 + math.exp(-v)))
        return self.output

    def backward(self, grad):
        # sigmoid 的导数：o * (1 - o)
        return [g * o * (1 - o) for g, o in zip(grad, self.output)]


# Tanh 激活函数
class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = [math.tanh(v) for v in x]
        return self.output

    def backward(self, grad):
        # tanh 的导数：1 - tanh^2(x)
        return [g * (1 - o * o) for g, o in zip(grad, self.output)]


# Dropout：训练时随机置零，评估时直通
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x
        # 倒置 Dropout：缩放保留元素以保持期望值
        self.mask = [0.0 if random.random() < self.p else 1.0 / (1 - self.p) for _ in x]
        return [v * m for v, m in zip(x, self.mask)]

    def backward(self, grad):
        if self.mask is None:
            return grad
        return [g * m for g, m in zip(grad, self.mask)]


# BatchNorm (批归一化)：归一化激活值，加速训练
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
            # 计算 batch 均值
            mean = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            # 计算 batch 方差
            var = [0.0] * self.size
            for sample in batch:
                for j in range(self.size):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in var]

            # 归一化并应用 gamma/beta 缩放平移
            self.x_norm = []
            self.batch_input = batch
            for sample in batch:
                normed = [(sample[j] - mean[j]) * self.std_inv[j] for j in range(self.size)]
                self.x_norm.append(normed)
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

            # 更新 running statistics (滑动统计量)
            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            # 评估模式：使用 running statistics
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            for sample in batch:
                normed = [(sample[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
                output = [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]
                output_batch.append(output)

        return output_batch

    def forward(self, x):
        if self.training:
            # 单样本训练时更新 running_mean（用于兼容逐样本训练）
            for j in range(self.size):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * x[j]

            self.std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            self.x_norm = [(x[j] - self.running_mean[j]) * self.std_inv[j] for j in range(self.size)]
            return [self.gamma[j] * self.x_norm[j] + self.beta[j] for j in range(self.size)]
        else:
            std_inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            normed = [(x[j] - self.running_mean[j]) * std_inv[j] for j in range(self.size)]
            return [self.gamma[j] * normed[j] + self.beta[j] for j in range(self.size)]

    def backward(self, grad):
        if self.x_norm is None:
            return grad
        x_norm = self.x_norm if not isinstance(self.x_norm[0], list) else self.x_norm[0]
        for j in range(self.size):
            self.gamma_grads[j] += x_norm[j] * grad[j]
            self.beta_grads[j] += grad[j]
        return [grad[j] * self.gamma[j] * self.std_inv[j] for j in range(self.size)]

    def parameters(self):
        params = []
        for j in range(self.size):
            params.append((self.gamma, j, None, self.gamma_grads))
            params.append((self.beta, j, None, self.beta_grads))
        return params


# Sequential (顺序容器)：串联多个 Module
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x

    def backward(self, grad):
        # Backward pass (反向传播)：从最后一层到第一层
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

    def count_parameters(self):
        return len(self.parameters())


# MSELoss (均方误差损失)：用于回归任务
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


# BCELoss (二元交叉熵损失)：用于二分类任务
class BCELoss:
    def __call__(self, predicted, target):
        self.predicted = predicted
        self.target = target
        eps = 1e-7
        n = len(predicted)
        self.loss = 0
        for p, t in zip(predicted, target):
            # 裁剪防止 log(0)
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


# SGD (随机梯度下降) Optimizer (优化器)
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        # 使用 gradient (梯度) 更新 parameter (参数)
        for container, i, j, grad_container in self.params:
            if j is not None:
                container[i][j] -= self.lr * grad_container[i][j]
            else:
                container[i] -= self.lr * grad_container[i]

    def zero_grad(self):
        # 清零所有 gradient
        for container, i, j, grad_container in self.params:
            if j is not None:
                grad_container[i][j] = 0.0
            else:
                grad_container[i] = 0.0


# Adam (自适应矩估计) Optimizer (优化器)
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

            # 更新一阶矩 (momentum) 和二阶矩 (variance) 估计
            self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * g * g

            # 偏差修正
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


# DataLoader (数据加载器)：将数据分成 batch (批次)，可选 shuffle (打乱)
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


# 生成 circle classification (圆形分类) 数据集
def make_circle_data(n=500, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        # 点在圆内标签为 1，否则为 0
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], [label]))
    return data


# 使用 Adam 训练 4 层网络
def train_framework():
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

    print(f"Model: 4 linear layers (2->16->16->8->1)")
    print(f"Total parameters: {model.count_parameters()}")
    print(f"Optimizer: Adam (lr=0.01)")
    print(f"Loss: Binary Cross-Entropy")
    print(f"Data: 500 samples (80/20 train/test split)")
    print()

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
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                total_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

                predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
                if predicted_class == t[0]:
                    total_correct += 1
                total_samples += 1

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 10 == 0 or epoch == 99:
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%")

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    test_accuracy = correct / len(test_data) * 100
    print(f"\n  Test Accuracy: {test_accuracy:.1f}% ({correct}/{len(test_data)})")

    return model, test_accuracy


# 使用 SGD 训练相同架构
def train_with_sgd():
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
    optimizer = SGD(model.parameters(), lr=0.1)

    data = make_circle_data(500)
    split = int(len(data) * 0.8)
    train_data = data[:split]
    test_data = data[split:]
    loader = DataLoader(train_data, batch_size=16, shuffle=True)

    model.train()

    for epoch in range(100):
        total_loss = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                loss = criterion(pred, t)
                total_loss += loss

                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()
                total_samples += 1

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    return correct / len(test_data) * 100


# 使用 Dropout 训练
def train_with_dropout():
    random.seed(42)

    model = Sequential(
        Linear(2, 16),
        ReLU(),
        Dropout(0.3),
        Linear(16, 16),
        ReLU(),
        Dropout(0.3),
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
        for batch_inputs, batch_targets in loader:
            for x, t in zip(batch_inputs, batch_targets):
                pred = model.forward(x)
                criterion(pred, t)
                optimizer.zero_grad()
                grad = criterion.backward()
                model.backward(grad)
                optimizer.step()

    model.eval()
    correct = 0
    for x, t in test_data:
        pred = model.forward(x)
        predicted_class = 1.0 if pred[0] >= 0.5 else 0.0
        if predicted_class == t[0]:
            correct += 1
    return correct / len(test_data) * 100


# 打印样本预测结果
def sample_predictions(model, data):
    test_points = [
        ([0.0, 0.0], "inside"),
        ([0.5, 0.5], "inside"),
        ([1.0, 0.0], "inside"),
        ([-0.3, 0.3], "inside"),
        ([1.5, 1.5], "outside"),
        ([0.0, 1.8], "outside"),
        ([-1.5, -1.0], "outside"),
        ([2.0, 0.0], "outside"),
    ]

    print("\n  Sample Predictions:")
    for point, expected in test_points:
        pred = model.forward(point)
        predicted_region = "inside" if pred[0] >= 0.5 else "outside"
        status = "OK" if predicted_region == expected else "WRONG"
        print(f"    ({point[0]:5.1f}, {point[1]:5.1f}) -> {pred[0]:.4f} ({predicted_region:7s}, expected {expected:7s}) {status}")


if __name__ == "__main__":
    print("=" * 70)
    print("MINI FRAMEWORK (迷你框架) -- Phase 3 Capstone")
    print("=" * 70)
    print()

    print("-" * 70)
    print("EXPERIMENT 1: Adam Optimizer (4-layer network)")
    print("-" * 70)
    model, adam_acc = train_framework()
    sample_predictions(model, None)

    print("\n" + "-" * 70)
    print("EXPERIMENT 2: SGD Optimizer (same architecture)")
    print("-" * 70)
    sgd_acc = train_with_sgd()
    print(f"  SGD Test Accuracy: {sgd_acc:.1f}%")

    print("\n" + "-" * 70)
    print("EXPERIMENT 3: With Dropout (p=0.3)")
    print("-" * 70)
    dropout_acc = train_with_dropout()
    print(f"  Dropout Test Accuracy: {dropout_acc:.1f}%")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Adam (no dropout):     {adam_acc:.1f}%")
    print(f"  SGD (no dropout):      {sgd_acc:.1f}%")
    print(f"  Adam + Dropout(0.3):   {dropout_acc:.1f}%")

    print("\n" + "=" * 70)
    print("FRAMEWORK COMPONENTS (框架组件)")
    print("=" * 70)
    print(f"  Modules (模块):    Linear, ReLU, Sigmoid, Tanh, Dropout, BatchNorm")
    print(f"  Containers (容器): Sequential")
    print(f"  Losses (损失):     MSELoss, BCELoss")
    print(f"  Optimizers (优化器): SGD, Adam")
    print(f"  Data (数据):       DataLoader (batching + shuffle)")
    print(f"  Total:      ~500 lines of pure Python")
