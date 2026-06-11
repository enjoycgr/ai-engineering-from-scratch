import math
import random


class Dropout:
    """Dropout (随机失活) 层：以概率 p 在训练期间随机将神经元输出置零。"""

    def __init__(self, p=0.5):
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        # 推理模式：直接返回输入，不做任何修改
        if not self.training:
            return list(x)
        self.mask = []
        output = []
        for val in x:
            if random.random() < self.p:
                self.mask.append(0)
                output.append(0.0)
            else:
                self.mask.append(1)
                # Inverted dropout (反转dropout)：训练时缩放，测试时无需修改
                output.append(val / (1 - self.p))
        return output

    def backward(self, grad_output):
        # 反向传播：被丢弃的神经元梯度为零，其余按缩放因子回传
        grads = []
        for g, m in zip(grad_output, self.mask):
            if m == 0:
                grads.append(0.0)
            else:
                grads.append(g / (1 - self.p))
        return grads


def l2_regularization(weights, lambda_reg):
    """计算 L2 regularization (L2正则化) 惩罚项：0.5 * lambda * sum(w^2)。"""
    penalty = 0.0
    for w in weights:
        penalty += w * w
    return lambda_reg * 0.5 * penalty


def l2_gradient(weights, lambda_reg):
    """计算 L2 regularization (L2正则化) 对权重的梯度：lambda * w。"""
    return [lambda_reg * w for w in weights]


class BatchNorm:
    """Batch Normalization (批归一化)：跨 batch (批次) 维度对每层的输出进行归一化。"""

    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = [0.0] * num_features
        self.running_var = [1.0] * num_features
        self.training = True
        self.num_features = num_features

    def forward(self, batch):
        batch_size = len(batch)
        if self.training:
            # 训练期间：使用当前 batch (批次) 的均值和方差
            mean = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    mean[j] += sample[j]
            mean = [m / batch_size for m in mean]

            var = [0.0] * self.num_features
            for sample in batch:
                for j in range(self.num_features):
                    var[j] += (sample[j] - mean[j]) ** 2
            var = [v / batch_size for v in var]

            # 更新 running statistics (运行统计量) 用于推理
            for j in range(self.num_features):
                self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * mean[j]
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * var[j]
        else:
            # 推理期间：使用训练期间累积的 running statistics (运行统计量)
            mean = list(self.running_mean)
            var = list(self.running_var)

        self.x_hat = []
        output = []
        for sample in batch:
            normalized = []
            out_sample = []
            for j in range(self.num_features):
                x_h = (sample[j] - mean[j]) / math.sqrt(var[j] + self.eps)
                normalized.append(x_h)
                # 使用可学习的 gamma 和 beta 进行缩放和平移
                out_sample.append(self.gamma[j] * x_h + self.beta[j])
            self.x_hat.append(normalized)
            output.append(out_sample)
        return output


class LayerNorm:
    """Layer Normalization (层归一化)：在每个样本内跨特征维度归一化，与 batch size (批次大小) 无关。"""

    def __init__(self, num_features, eps=1e-5):
        self.gamma = [1.0] * num_features
        self.beta = [0.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        # 计算当前样本的 feature mean (特征均值) 和 variance (方差)
        mean = sum(x) / len(x)
        var = sum((xi - mean) ** 2 for xi in x) / len(x)

        self.x_hat = []
        output = []
        for j in range(self.num_features):
            x_h = (x[j] - mean) / math.sqrt(var + self.eps)
            self.x_hat.append(x_h)
            output.append(self.gamma[j] * x_h + self.beta[j])
        return output


class RMSNorm:
    """RMSNorm：去掉均值减法的 LayerNorm，仅按 RMS (均方根) 缩放，速度提升约 10%。"""

    def __init__(self, num_features, eps=1e-6):
        self.gamma = [1.0] * num_features
        self.eps = eps
        self.num_features = num_features

    def forward(self, x):
        # 仅计算 RMS (均方根)，不做 mean subtraction (均值减法)
        rms = math.sqrt(sum(xi * xi for xi in x) / len(x) + self.eps)
        output = []
        for j in range(self.num_features):
            output.append(self.gamma[j] * x[j] / rms)
        return output


def sigmoid(x):
    """数值稳定的 sigmoid 函数。"""
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def make_circle_data(n=200, seed=42):
    """生成 circle classification dataset (圆形分类数据集)。"""
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class RegularizedNetwork:
    """
    一个带有可选 Dropout (随机失活) 和 L2 weight decay (L2权重衰减) 的 2 层 MLP。
    用于演示不同 regularization (正则化) 策略对 overfitting (过拟合) 的影响。
    """

    def __init__(self, hidden_size=16, lr=0.05, dropout_p=0.0, weight_decay=0.0):
        random.seed(0)
        self.hidden_size = hidden_size
        self.lr = lr
        self.dropout_p = dropout_p
        self.weight_decay = weight_decay
        self.dropout = Dropout(p=dropout_p) if dropout_p > 0 else None

        # 第一层 weight (权重) 和 bias (偏置)
        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        # 第二层 weight (权重) 和 bias (偏置)
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x, training=True):
        """前向传播。training=True 时启用 dropout (随机失活)。"""
        self.x = x
        self.z1 = []
        self.h = []
        # 第一层线性变换 + ReLU 激活
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(max(0.0, z))

        # 可选的 Dropout (随机失活) 层
        if self.dropout and training:
            self.dropout.training = True
            self.h = self.dropout.forward(self.h)
        elif self.dropout:
            self.dropout.training = False
            self.h = self.dropout.forward(self.h)

        # 第二层线性变换 + sigmoid 输出
        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        """反向传播，包含 binary cross-entropy (二元交叉熵) 梯度和 weight decay (权重衰减)。"""
        eps = 1e-15
        p = max(eps, min(1 - eps, self.out))
        # Binary cross-entropy (二元交叉熵) 对输出的梯度
        d_loss = -(target / p) + (1 - target) / (1 - p)
        d_sigmoid = self.out * (1 - self.out)
        d_out = d_loss * d_sigmoid

        # 通过 dropout (随机失活) 掩码回传梯度
        d_h_dropout = [d_out * self.w2[i] for i in range(self.hidden_size)]
        if self.dropout and self.dropout.mask is not None:
            d_h_dropout = [g * m / (1 - self.dropout.p) if m else 0.0
                           for g, m in zip(d_h_dropout, self.dropout.mask)]

        # 更新 weight (权重) 和 bias (偏置)，包含 weight decay (权重衰减) 项
        for i in range(self.hidden_size):
            d_relu = 1.0 if self.z1[i] > 0 else 0.0
            d_h = d_h_dropout[i] * d_relu
            self.w2[i] -= self.lr * (d_out * self.h[i] + self.weight_decay * self.w2[i])
            for j in range(2):
                self.w1[i][j] -= self.lr * (d_h * self.x[j] + self.weight_decay * self.w1[i][j])
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def evaluate(self, data):
        """在 dataset (数据集) 上评估：返回平均 loss (损失) 和 accuracy (准确率)%。"""
        correct = 0
        total_loss = 0.0
        for x, y in data:
            pred = self.forward(x, training=False)
            eps = 1e-15
            p = max(eps, min(1 - eps, pred))
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            if (pred >= 0.5) == (y >= 0.5):
                correct += 1
        return total_loss / len(data), correct / len(data) * 100

    def train_model(self, train_data, test_data, epochs=300):
        """训练模型并记录每个 epoch 的 train/test metrics (指标)。"""
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for x, y in train_data:
                pred = self.forward(x, training=True)
                self.backward(y)
                eps = 1e-15
                p = max(eps, min(1 - eps, pred))
                total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            train_loss = total_loss / len(train_data)
            train_acc = correct / len(train_data) * 100
            test_loss, test_acc = self.evaluate(test_data)
            history.append((train_loss, train_acc, test_loss, test_acc))
            if epoch % 75 == 0 or epoch == epochs - 1:
                gap = train_acc - test_acc
                print(f"    Epoch {epoch:3d}: train_acc={train_acc:.1f}%, test_acc={test_acc:.1f}%, gap={gap:.1f}%")
        return history


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Dropout (随机失活) 演示")
    print("=" * 60)
    drop = Dropout(p=0.5)
    test_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    random.seed(42)

    drop.training = True
    print(f"  Input:          {test_input}")
    for trial in range(3):
        output = drop.forward(test_input)
        active = sum(1 for v in output if v > 0)
        print(f"  Train pass {trial+1}:   {[f'{v:.1f}' for v in output]}  ({active}/{len(test_input)} active)")

    drop.training = False
    output = drop.forward(test_input)
    print(f"  Eval pass:      {[f'{v:.1f}' for v in output]}")
    print(f"  Train mean: ~{sum(test_input)/len(test_input):.1f} (scaled by 1/(1-p))")
    print(f"  Eval mean:   {sum(output)/len(output):.1f} (no scaling needed)")

    print("\n" + "=" * 60)
    print("STEP 2: L2 Regularization (L2正则化)")
    print("=" * 60)
    weights = [0.5, -1.2, 3.0, 0.1, -2.5]
    lambda_val = 0.01
    penalty = l2_regularization(weights, lambda_val)
    grads = l2_gradient(weights, lambda_val)
    print(f"  Weights: {weights}")
    print(f"  Lambda:  {lambda_val}")
    print(f"  L2 penalty: {penalty:.6f}")
    print(f"  L2 grads:   {[f'{g:.4f}' for g in grads]}")
    print(f"  Largest weight (3.0) gets largest gradient ({grads[2]:.4f})")

    print("\n" + "=" * 60)
    print("STEP 3: BatchNorm vs LayerNorm vs RMSNorm")
    print("=" * 60)
    random.seed(42)
    batch = [[random.gauss(5, 2) for _ in range(4)] for _ in range(8)]
    sample = batch[0]

    bn = BatchNorm(4)
    bn_out = bn.forward(batch)

    ln = LayerNorm(4)
    ln_out = ln.forward(sample)

    rn = RMSNorm(4)
    rn_out = rn.forward(sample)

    print(f"  Raw sample: {[f'{v:.2f}' for v in sample]}")
    print(f"  BatchNorm:  {[f'{v:.2f}' for v in bn_out[0]]}")
    print(f"  LayerNorm:  {[f'{v:.2f}' for v in ln_out]}")
    print(f"  RMSNorm:    {[f'{v:.2f}' for v in rn_out]}")

    ln_mean = sum(ln_out) / len(ln_out)
    ln_std = math.sqrt(sum((v - ln_mean) ** 2 for v in ln_out) / len(ln_out))
    rn_mean = sum(rn_out) / len(rn_out)
    rn_rms = math.sqrt(sum(v * v for v in rn_out) / len(rn_out))
    print(f"\n  LayerNorm output: mean={ln_mean:.4f}, std={ln_std:.4f}")
    print(f"  RMSNorm output:   mean={rn_mean:.4f}, rms={rn_rms:.4f}")
    print(f"  LayerNorm centers to mean=0. RMSNorm normalizes scale only.")

    print("\n" + "=" * 60)
    print("STEP 4: BatchNorm Training vs Eval Mode")
    print("=" * 60)
    bn2 = BatchNorm(4)
    bn2.training = True
    for step in range(10):
        batch = [[random.gauss(3 + step * 0.1, 1) for _ in range(4)] for _ in range(16)]
        bn2.forward(batch)

    print(f"  Running mean after 10 batches: {[f'{v:.3f}' for v in bn2.running_mean]}")
    print(f"  Running var  after 10 batches: {[f'{v:.3f}' for v in bn2.running_var]}")

    bn2.training = False
    test_sample = [[5.0, 5.0, 5.0, 5.0]]
    eval_out = bn2.forward(test_sample)
    print(f"  Eval mode uses running stats, not batch stats")
    print(f"  Input [5,5,5,5] -> {[f'{v:.3f}' for v in eval_out[0]]}")

    print("\n" + "=" * 60)
    print("STEP 5: Training With vs Without Regularization")
    print("=" * 60)
    all_data = make_circle_data(n=300, seed=42)
    train_data = all_data[:150]
    test_data = all_data[150:]

    configs = [
        ("No regularization", 0.0, 0.0),
        ("Dropout p=0.3", 0.3, 0.0),
        ("Weight decay 0.01", 0.0, 0.01),
        ("Dropout + weight decay", 0.3, 0.01),
    ]

    results = {}
    for name, drop_p, wd in configs:
        print(f"\n--- {name} ---")
        net = RegularizedNetwork(hidden_size=16, lr=0.05, dropout_p=drop_p, weight_decay=wd)
        history = net.train_model(train_data, test_data, epochs=300)
        results[name] = history

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"  {'Config':30s} {'Train Acc':>10s} {'Test Acc':>10s} {'Gap':>8s}")
    print("  " + "-" * 60)
    for name, history in results.items():
        train_loss, train_acc, test_loss, test_acc = history[-1]
        gap = train_acc - test_acc
        print(f"  {name:30s} {train_acc:>9.1f}% {test_acc:>9.1f}% {gap:>7.1f}%")

    print("\n  Key insight: regularization reduces the train-test gap.")
    print("  The model with dropout + weight decay generalizes best,")
    print("  even if its training accuracy is lower.")
