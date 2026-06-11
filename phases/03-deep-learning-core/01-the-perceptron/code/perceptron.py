class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        # weight (权重) 初始化为 0，bias (偏置) 初始化为 0
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        # 计算 weighted sum (加权和) 并加上 bias，然后通过 step activation function (阶跃激活函数)
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        # 使用 perceptron learning rule (感知器学习规则) 训练
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    # 更新 weight：w_i = w_i + learning_rate (学习率) * error * x_i
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")


def test_gate(name, n_inputs, data):
    print(f"=== {name} ===")
    p = Perceptron(n_inputs)
    p.train(data)
    print(f"  Weights: {p.weights}, Bias: {p.bias}")
    for inputs, expected in data:
        result = p.predict(inputs)
        status = "OK" if result == expected else "WRONG"
        print(f"  {inputs} -> {result} (expected {expected}) {status}")
    print()


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

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

test_gate("AND Gate", 2, and_data)
test_gate("OR Gate", 2, or_data)
test_gate("NOT Gate", 1, not_data)

print("=== XOR Gate (single perceptron - will fail) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
print()


def xor_network(x1, x2):
    # 手动设置 weight 和 bias，用 OR + NAND + AND 组成 multi-layer perceptron (多层感知器) 解决 XOR
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
    return and_neuron.predict([hidden1, hidden2])


print("=== XOR Gate (multi-layer network - works) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
print()


class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        # hidden layer (隐藏层) 的 weight 和 bias
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        # output layer (输出层) 的 weight 和 bias
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        # forward pass (前向传播)：计算 hidden layer 和 output layer 的输出
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        # 使用 backpropagation (反向传播) 训练两层网络
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                # 输出层梯度：sigmoid 导数 * error
                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    # 隐藏层梯度：链式法则
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                # 更新 output layer 的 weight 和 bias
                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                # 更新 hidden layer 的 weight 和 bias
                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]

            if epoch % 2000 == 0:
                print(f"  Epoch {epoch}, error: {total_error:.4f}")


print("=== XOR Gate (trained 2-layer network with backpropagation) ===")
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
print()
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
