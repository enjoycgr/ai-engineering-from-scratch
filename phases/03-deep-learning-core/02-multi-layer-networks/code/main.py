import math
import random


def sigmoid(x):
    # 将输入限制在 [-500, 500] 范围内，防止 math.exp 溢出
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class Layer:
    """一个全连接层：保存 weight matrix（权重矩阵）和 bias vector（偏置向量），执行 forward pass（前向传播）。"""

    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            # 随机初始化 weight（权重），范围 [-1, 1]
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            # 默认 bias（偏置）初始化为 0
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        """执行 forward pass（前向传播）：线性变换 + sigmoid 激活。"""
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            # 计算该 neuron（神经元）的加权和
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            # 加上 bias（偏置）
            z += self.biases[neuron_idx]
            # 应用 sigmoid activation function（激活函数）
            self.last_output.append(sigmoid(z))
        return self.last_output


class Network:
    """一个 multi-layer network（多层网络）：按顺序堆叠 Layer，链式执行 forward pass（前向传播）。"""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        """将输入逐层推送，完成整个 forward pass（前向传播）。"""
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def count_parameters(self):
        """计算可训练 parameter（参数）的总数：所有 weight（权重）和 bias（偏置）之和。"""
        total = 0
        for layer in self.layers:
            for neuron_weights in layer.weights:
                total += len(neuron_weights)
            total += len(layer.biases)
        return total


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 1: XOR with hand-tuned 2-2-1 network")
    print("=" * 60)

    # 手工调优的 hidden layer（隐藏层）：第一个 neuron（神经元）近似 OR，第二个近似 NAND
    hidden = Layer(
        n_inputs=2,
        n_neurons=2,
        weights=[[20.0, 20.0], [-20.0, -20.0]],
        biases=[-10.0, 30.0],
    )

    # output layer（输出层）：将 hidden layer（隐藏层）的特征组合成 AND，即 XOR
    output = Layer(
        n_inputs=2,
        n_neurons=1,
        weights=[[20.0, 20.0]],
        biases=[-30.0],
    )

    xor_net = Network([hidden, output])

    xor_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]

    all_correct = True
    for inputs, expected in xor_data:
        result = xor_net.forward(inputs)
        predicted = 1 if result[0] >= 0.5 else 0
        status = "OK" if predicted == expected else "WRONG"
        if predicted != expected:
            all_correct = False
        print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected}) {status}")

    print(f"\nXOR solved: {all_correct}")
    print(f"Parameters: {xor_net.count_parameters()}")

    print()
    print("=" * 60)
    print("DEMO 2: Circle classification with 2-8-1 network")
    print("=" * 60)

    random.seed(42)

    # 生成圆形分类数据集：半径 0.5，圆心在原点
    data = []
    for _ in range(200):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        label = 1 if (x * x + y * y) < 0.25 else 0
        data.append(([x, y], label))

    inside_count = sum(1 for _, label in data if label == 1)
    outside_count = len(data) - inside_count
    print(f"  Dataset: {len(data)} points ({inside_count} inside, {outside_count} outside)")

    random.seed(7)
    circle_net = Network([
        Layer(n_inputs=2, n_neurons=8),
        Layer(n_inputs=8, n_neurons=1),
    ])

    correct = 0
    for inputs, expected in data:
        result = circle_net.forward(inputs)
        predicted = 1 if result[0] >= 0.5 else 0
        if predicted == expected:
            correct += 1

    print(f"  Accuracy with random weights: {correct}/{len(data)} ({100 * correct / len(data):.1f}%)")
    print(f"  Parameters: {circle_net.count_parameters()}")
    print(f"  (Random weights give poor accuracy -- training needed)")

    print()
    print("=" * 60)
    print("DEMO 3: Forward pass internals on XOR")
    print("=" * 60)

    # 展示 XOR 的 forward pass（前向传播）内部细节：hidden layer（隐藏层）和 output layer（输出层）的输出
    for inputs, expected in xor_data:
        xor_net.forward(inputs)
        h = xor_net.layers[0].last_output
        o = xor_net.layers[1].last_output
        print(f"  Input: {inputs}")
        print(f"    Hidden: [{h[0]:.6f}, {h[1]:.6f}]")
        print(f"    Output: {o[0]:.6f} -> {'1' if o[0] >= 0.5 else '0'} (expected: {expected})")

    print()
    print("=" * 60)
    print("DEMO 4: Parameter count for classic architectures")
    print("=" * 60)

    architectures = [
        ("2-3-1 (this lesson)", [2, 3, 1]),
        ("2-8-1 (circle)", [2, 8, 1]),
        ("784-256-128-10 (MNIST)", [784, 256, 128, 10]),
        ("784-512-256-128-10 (deep MNIST)", [784, 512, 256, 128, 10]),
    ]

    for name, sizes in architectures:
        layers = []
        for i in range(1, len(sizes)):
            layers.append(Layer(n_inputs=sizes[i - 1], n_neurons=sizes[i]))
        net = Network(layers)
        print(f"  {name}: {net.count_parameters():,} parameters")
