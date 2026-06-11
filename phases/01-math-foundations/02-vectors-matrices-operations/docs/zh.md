# 向量、矩阵与运算

> 每个神经网络本质上都是矩阵乘法，只是多了几步。

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (线性代数直觉)
**Time:** ~60 分钟

## 学习目标

- 构建一个 Matrix 类，支持 element-wise (逐元素) 运算、matrix multiplication (矩阵乘法)、transpose (转置)、determinant (行列式) 和 inverse (逆矩阵)
- 区分 element-wise multiplication (逐元素乘法) 与 matrix multiplication (矩阵乘法)，并解释各自适用场景
- 仅使用从零构建的 Matrix 类实现单层 dense (全连接) 神经网络层：`relu(W @ x + b)`
- 解释 broadcasting (广播) 规则以及 neural network framework (神经网络框架) 中 bias addition (偏置加法) 的工作原理

## 问题

你想构建一个神经网络。你读到这样一行代码：

```
output = activation(weights @ input + bias)
```

这里的 `@` 是 matrix multiplication (矩阵乘法)。`weights` 是矩阵，`input` 是向量。如果你不知道这些运算在做什么，这行代码就是魔法。但如果你知道，它就是单层 forward pass (前向传播) 的完整三步骤。

模型处理的每张图片都是 pixel value (像素值) 矩阵。每个 word embedding (词嵌入) 都是向量。每个神经网络的每一层都是矩阵变换。你无法在不精通矩阵运算的情况下构建 AI 系统，就像你无法在不理解变量的情况下编写代码。

本课程从零构建这种精通能力。

## 概念

### 向量：有序数字列表

向量是具有方向和大小的数字列表。在 AI 中，向量代表数据点、特征或参数。

```
v = [3, 4]        -- 一个 2D 向量
w = [1, 0, -2]    -- 一个 3D 向量
```

二维向量 `[3, 4]` 指向平面上的坐标点 (3, 4)。它的长度（模长）为 5（3-4-5 三角形）。

### 矩阵：数字网格

矩阵是一个二维网格。行和列。一个 m x n 矩阵有 m 行和 n 列。

```
A = | 1  2  3 |     -- 2x3 矩阵 (2 行, 3 列)
    | 4  5  6 |
```

在 neural networks (神经网络) 中，weight matrix (权重矩阵) 将 input vector (输入向量) 变换为 output vector (输出向量)。一个有 784 个输入和 128 个输出的层使用一个 128×784 的 weight matrix (权重矩阵)。

### 为什么形状很重要

Matrix multiplication (矩阵乘法) 有一条严格的规则：`(m x n) @ (n x p) = (m x p)`。内维度必须匹配。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

内维度: 784 = 784  -- 合法
```

如果你在 PyTorch 中遇到 shape mismatch (形状不匹配) 错误，原因就在这里。

### 运算速查表

| 运算 | 作用 | 神经网络中的应用 |
|-----------|-------------|-------------------|
| Addition (加法) | 逐元素组合 | 给输出添加 bias (偏置) |
| Scalar multiply (标量乘法) | 缩放每个元素 | Learning rate (学习率) * gradients (梯度) |
| Matrix multiply (矩阵乘法) | 变换向量 | Layer forward pass (层前向传播) |
| Transpose (转置) | 翻转行列 | Backpropagation (反向传播) |
| Determinant (行列式) | 单一数值摘要 | 检查可逆性 |
| Inverse (逆矩阵) | 撤销变换 | 求解线性方程组 |
| Identity matrix (单位矩阵) | 无操作矩阵 | 初始化、residual connections (残差连接) |

### Element-wise (逐元素) vs Matrix multiplication (矩阵乘法)

这个区别经常让初学者困惑。

Element-wise (逐元素)：相乘匹配的位置。两个矩阵形状必须相同。

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Matrix multiplication (矩阵乘法)：行与列的 dot product (点积)。内维度必须匹配。

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

不同的运算，不同的结果，不同的规则。

### Broadcasting (广播)

当你将 bias vector (偏置向量) 加到输出矩阵时，形状不匹配。Broadcasting (广播) 会拉伸较小的数组以匹配。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting 将向量拉伸到每一行:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

每个现代框架都自动执行此操作。理解它可以在形状看起来错误但代码运行正常时避免困惑。

## 从零构建

### 第一步：Vector 类

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 第二步：带核心运算的 Matrix 类

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 第三步：查看效果

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 第四步：与神经网络建立联系

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

这就是单个 dense layer (全连接层)：`output = relu(W @ x + b)`。每个神经网络的每个 dense layer (全连接层) 都在做完全相同的事情。

## 使用它

NumPy 可以用更少的代码行、快数个数量级地完成上述所有操作。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 中的 `@` 运算符调用 `__matmul__`。NumPy 使用 C 和 Fortran 编写的优化 BLAS 例程来实现它。相同的数学，快 100 倍。

NumPy 中的 Broadcasting (广播):

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 自动将一维 bias (偏置) 广播到所有行。这就是每个 neural network framework (神经网络框架) 中 bias addition (偏置加法) 的工作原理。

## 交付物

本课程产出一个用于通过几何直觉教授矩阵运算的 prompt (提示词)。请参见 `outputs/prompt-matrix-operations.md`。

这里构建的 Matrix 类是我们在第三阶段第 10 课构建的迷你神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 计算 `A @ A.inverse_2x2()` 并确认得到单位矩阵。用三个不同的 2x2 矩阵尝试。当 determinant (行列式) 为零时会发生什么？

2. **实现 3x3 逆矩阵。** 使用 adjugate method (伴随矩阵法) 扩展 Matrix 类以计算 3x3 矩阵的逆矩阵。对照 NumPy 的 `np.linalg.inv` 测试。

3. **构建一个两层网络。** 仅使用你的 Matrix 类（不使用 NumPy），创建一个两层神经网络：输入 (3) -> 隐藏层 (4) -> 输出 (2)。初始化随机权重，运行 forward pass (前向传播)，并验证所有形状正确。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------------|
| Vector (向量) | "一个箭头" | 一个有序数字列表。在 AI 中：高维空间中的一个点。 |
| Matrix (矩阵) | "一张数字表格" | 一个线性变换。将向量从一个空间映射到另一个空间。 |
| Matrix multiply (矩阵乘法) | "把数字乘起来" | 第一个矩阵的每一行与第二个矩阵的每一列的 dot product (点积)。顺序很重要。 |
| Transpose (转置) | "翻转它" | 交换行和列。将 m x n 矩阵变成 n x m。在 backpropagation (反向传播) 中至关重要。 |
| Determinant (行列式) | "从矩阵算出来的某个数" | 衡量矩阵在 2D 中缩放面积或在 3D 中缩放体积的程度。为零意味着变换压扁了一个维度。 |
| Inverse (逆矩阵) | "撤销矩阵" | 可以反转变换的矩阵。只有当 determinant (行列式) 不为零时才存在。 |
| Identity matrix (单位矩阵) | "无聊的矩阵" | 矩阵版本的乘以 1。用于 residual connections (残差连接) (ResNets)。 |
| Broadcasting (广播) | "魔法形状修复" | 通过沿缺失维度重复来拉伸较小的数组以匹配较大的数组。 |
| Element-wise (逐元素) | "普通乘法" | 相乘匹配的位置。两个数组必须形状相同（或可广播）。 |

## 延伸阅读

- [3Blue1Brown: 线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) - 涵盖此处所有运算的视觉直觉
- [NumPy broadcasting 文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的确切规则
- [Stanford CS229 线性代数复习](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 针对机器学习线性代数的简洁参考
