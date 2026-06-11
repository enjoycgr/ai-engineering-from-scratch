# Linear Regression (线性回归)

> Linear regression (线性回归) 在你的数据中画出最佳直线。它是机器学习的 "hello world"。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Linear Algebra, Calculus, Optimization), Phase 2 Lesson 1
**Time:** ~90 分钟

## 学习目标

- 推导 mean squared error (均方误差) 的 gradient descent (梯度下降) 更新规则，并从零实现 linear regression (线性回归)
- 从计算复杂度和适用场景角度比较 gradient descent (梯度下降) 和 normal equation (正规方程)
- 构建一个带 feature scaling (特征缩放) 的 multiple linear regression (多元线性回归) 模型，并解释学习到的权重
- 解释 Ridge regression (岭回归) (L2 regularization (正则化)) 如何通过惩罚大权重来防止 overfitting (过拟合)

## 问题

你有数据：房屋面积和售价。你想根据面积预测新房的价格。你可以在散点图上目测，但你需要一个公式。你需要一条最拟合数据的直线，这样你就可以代入任意面积得到价格预测。

Linear regression (线性回归) 给你这条线。更重要的是，它引入了完整的 ML 训练循环：定义模型、定义 cost function (损失函数)、优化参数。每个 ML 算法都遵循这个模式。在这里用最简单的情况掌握它，你将在任何地方认出它。

这不仅用于简单问题。Linear regression (线性回归) 在生产系统中用于需求预测、A/B 测试分析、金融建模，以及作为每个回归任务的基线。

## 概念

### 模型

Linear regression (线性回归) 假设输入 (x) 和输出 (y) 之间存在线性关系：

```
y = wx + b
```

- `w` (weight/slope): x 增加 1 时 y 变化多少
- `b` (bias/intercept): x = 0 时 y 的值

对于多个输入 (features)，扩展为：

```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

或向量形式：`y = w^T * x + b`

目标：找到 w 和 b 的值，使得预测的 y 在所有训练样本上尽可能接近实际的 y。

### Cost Function (损失函数) (Mean Squared Error (均方误差))

如何衡量 "尽可能接近"？你需要一个单一数字来捕捉预测有多错误。最常见的选择是 Mean Squared Error (MSE) (均方误差)：

```
MSE = (1/n) * sum((y_predicted - y_actual)^2)
```

为什么用平方？两个原因。首先，它对大错误的惩罚比小错误更重（误差为 10 比误差为 1 严重 100 倍，不是 10 倍）。其次，平方函数处处光滑可导，这使得优化变得直接。

Cost function (损失函数) 创建一个曲面。对于单个权重 w 和偏置 b，MSE 曲面看起来像一个碗（凸抛物面）。碗底就是 MSE 最小的地方。训练就是找到这个碗底。

### Gradient Descent (梯度下降)

Gradient descent (梯度下降) 通过向山下迈步来找到碗底。

```mermaid
flowchart TD
    A[Initialize w and b randomly] --> B[Compute predictions: y_hat = wx + b]
    B --> C[Compute cost: MSE]
    C --> D[Compute gradients: dMSE/dw, dMSE/db]
    D --> E[Update parameters]
    E --> F{Cost low enough?}
    F -->|No| B
    F -->|Yes| G[Done: optimal w and b found]
```

Gradients 告诉你两件事：每个参数向哪个方向移动，以及移动多少。

对于 MSE，y_hat = wx + b：

```
dMSE/dw = (2/n) * sum((y_hat - y) * x)
dMSE/db = (2/n) * sum(y_hat - y)
```

更新规则：

```
w = w - learning_rate (学习率) * dMSE/dw
b = b - learning_rate (学习率) * dMSE/db
```

Learning rate (学习率) 控制步长。太大：越过最小值并发散。太小：训练永远进行。典型起始值：0.01、0.001 或 0.0001。

### Normal Equation (正规方程) (闭式解)

对于 linear regression (线性回归)  specifically，有一个直接公式可以在不迭代的情况下给出最优权重：

```
w = (X^T * X)^(-1) * X^T * y
```

这通过矩阵求逆一步解出 w。对于小数据集它工作得很好。对于大数据集（数百万行或数千个特征），gradient descent (梯度下降) 更受青睐，因为矩阵求逆在特征数量上是 O(n^3)。

### Multiple Linear Regression (多元线性回归)

有多个特征时，模型变为：

```
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

一切工作相同：MSE 是 cost function (损失函数)，gradient descent (梯度下降) 同时更新所有权重。唯一的区别是你拟合的是一个超平面而不是一条线。

Feature scaling (特征缩放) 在这里很重要。如果一个特征范围是 0 到 1，另一个范围是 0 到 1,000,000，gradient descent (梯度下降) 会挣扎，因为 cost surface 变得细长。训练前标准化特征（减去均值，除以标准差）。

### Polynomial Regression (多项式回归)

如果关系不是线性的怎么办？你仍然可以通过创建 polynomial features (多项式特征) 来使用 linear regression (线性回归)：

```
y = w1*x + w2*x^2 + w3*x^3 + b
```

这仍然是 "线性" 回归，因为模型在权重 (w1, w2, w3) 上是线性的。你只是使用了 x 的非线性特征。

高阶多项式可以拟合更复杂的曲线，但有 overfitting (过拟合) 的风险。一个 10 次多项式会通过 10 个数据点中的每一个，但在新数据上预测很差。

### R-squared (R平方)

MSE 告诉你有多错，但数字取决于 y 的尺度。R-squared (R^2) 给出一个与尺度无关的度量：

```
R^2 = 1 - (sum of squared residuals) / (sum of squared deviations from mean)
    = 1 - SS_res / SS_tot
```

- R^2 = 1.0: 完美预测
- R^2 = 0.0: 模型不比每次都预测均值更好
- R^2 < 0.0: 模型比预测均值更差

### Regularization (正则化) 预览 (Ridge Regression (岭回归))

当你有很多特征时，模型可以通过分配大权重来 overfitting (过拟合)。Ridge regression (岭回归) (L2 regularization (正则化)) 添加一个惩罚项：

```
Cost = MSE + lambda * sum(w_i^2)
```

惩罚项阻止大权重。Hyperparameter (超参数) lambda 控制权衡：更高的 lambda 意味着更小的权重和更多的 regularization (正则化)。这在后面的课程中深入讲解。现在，知道它存在以及为什么有帮助。

## 动手实现

### 步骤 1：生成样本数据

```python
import random
import math

random.seed(42)

TRUE_W = 3.0
TRUE_B = 7.0
N_SAMPLES = 100

X = [random.uniform(0, 10) for _ in range(N_SAMPLES)]
y = [TRUE_W * x + TRUE_B + random.gauss(0, 2.0) for x in X]

print(f"Generated {N_SAMPLES} samples")
print(f"True relationship: y = {TRUE_W}x + {TRUE_B} (+ noise)")
print(f"First 5 points: {[(round(X[i], 2), round(y[i], 2)) for i in range(5)]}")
```

### 步骤 2：用 gradient descent (梯度下降) 从零实现 linear regression (线性回归)

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.w = 0.0
        self.b = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        cost = sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n
        return cost

    def compute_gradients(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        dw = (2 / n) * sum((pred - actual) * x for pred, actual, x in zip(predictions, y, X))
        db = (2 / n) * sum(pred - actual for pred, actual in zip(predictions, y))
        return dw, db

    def fit(self, X, y, epochs=1000, print_every=200):
        for epoch in range(epochs):
            dw, db = self.compute_gradients(X, y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | w: {self.w:.4f} | b: {self.b:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


print("=== Training Linear Regression (Gradient Descent) ===")
model = LinearRegression(learning_rate=0.005)
model.fit(X, y, epochs=1000, print_every=200)
print(f"\nLearned: y = {model.w:.4f}x + {model.b:.4f}")
print(f"True:    y = {TRUE_W}x + {TRUE_B}")
print(f"R-squared: {model.r_squared(X, y):.4f}")
```

### 步骤 3：Normal equation (正规方程) (闭式解)

```python
class LinearRegressionNormal:
    def __init__(self):
        self.w = 0.0
        self.b = 0.0

    def fit(self, X, y):
        n = len(X)
        x_mean = sum(X) / n
        y_mean = sum(y) / n
        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))
        self.w = numerator / denominator
        self.b = y_mean - self.w * x_mean
        return self

    def predict(self, X):
        return [self.w * x + self.b for x in X]

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


print("\n=== Normal Equation (Closed-Form) ===")
model_normal = LinearRegressionNormal()
model_normal.fit(X, y)
print(f"Learned: y = {model_normal.w:.4f}x + {model_normal.b:.4f}")
print(f"R-squared: {model_normal.r_squared(X, y):.4f}")
```

### 步骤 4：Multiple linear regression (多元线性回归)

```python
class MultipleLinearRegression:
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.cost_history = []

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        n = len(y)
        return sum((pred - actual) ** 2 for pred, actual in zip(predictions, y)) / n

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


random.seed(42)
N = 100
X_multi = []
y_multi = []
for _ in range(N):
    size = random.uniform(500, 3000)
    bedrooms = random.randint(1, 5)
    age = random.uniform(0, 50)
    price = 50 * size + 10000 * bedrooms - 1000 * age + 50000 + random.gauss(0, 20000)
    X_multi.append([size, bedrooms, age])
    y_multi.append(price)


def standardize(X):
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = []
    for j in range(n_features):
        variance = sum((X[i][j] - means[j]) ** 2 for i in range(len(X))) / len(X)
        stds.append(variance ** 0.5)
    X_scaled = []
    for i in range(len(X)):
        row = [(X[i][j] - means[j]) / stds[j] if stds[j] > 0 else 0 for j in range(n_features)]
        X_scaled.append(row)
    return X_scaled, means, stds


y_mean_val = sum(y_multi) / len(y_multi)
y_std_val = (sum((yi - y_mean_val) ** 2 for yi in y_multi) / len(y_multi)) ** 0.5
y_scaled = [(yi - y_mean_val) / y_std_val for yi in y_multi]

X_scaled, x_means, x_stds = standardize(X_multi)

print("\n=== Multiple Linear Regression (3 features) ===")
print("Features: house size, bedrooms, age")
multi_model = MultipleLinearRegression(n_features=3, learning_rate=0.01)
multi_model.fit(X_scaled, y_scaled, epochs=1000, print_every=200)

print(f"\nWeights (standardized): {[round(w, 4) for w in multi_model.weights]}")
print(f"Bias (standardized): {multi_model.bias:.4f}")
print(f"R-squared: {multi_model.r_squared(X_scaled, y_scaled):.4f}")
```

### 步骤 5：Polynomial regression (多项式回归)

```python
class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.01):
        self.degree = degree
        self.weights = [0.0] * degree
        self.bias = 0.0
        self.lr = learning_rate

    def make_features(self, X):
        return [[x ** (d + 1) for d in range(self.degree)] for x in X]

    def predict(self, X):
        features = self.make_features(X)
        return [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]

    def fit(self, X, y, epochs=1000, print_every=200):
        features = self.make_features(X)
        n = len(y)
        for epoch in range(epochs):
            predictions = [sum(w * f for w, f in zip(self.weights, row)) + self.bias for row in features]
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            for j in range(self.degree):
                grad = (2 / n) * sum(errors[i] * features[i][j] for i in range(n))
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                cost = sum(e ** 2 for e in errors) / n
                print(f"  Epoch {epoch:4d} | Cost: {cost:.6f}")
        return self

    def r_squared(self, X, y):
        predictions = self.predict(X)
        y_mean = sum(y) / len(y)
        ss_res = sum((actual - pred) ** 2 for actual, pred in zip(y, predictions))
        ss_tot = sum((actual - y_mean) ** 2 for actual in y)
        return 1 - (ss_res / ss_tot)


random.seed(42)
X_poly = [x / 10.0 for x in range(0, 50)]
y_poly = [0.5 * x ** 2 - 2 * x + 3 + random.gauss(0, 1.0) for x in X_poly]

x_max = max(abs(x) for x in X_poly)
X_poly_norm = [x / x_max for x in X_poly]
y_poly_mean = sum(y_poly) / len(y_poly)
y_poly_std = (sum((yi - y_poly_mean) ** 2 for yi in y_poly) / len(y_poly)) ** 0.5
y_poly_norm = [(yi - y_poly_mean) / y_poly_std for yi in y_poly]

print("\n=== Polynomial Regression (degree 2 vs degree 5) ===")
print("True relationship: y = 0.5x^2 - 2x + 3")

print("\nDegree 2:")
poly2 = PolynomialRegression(degree=2, learning_rate=0.1)
poly2.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly2.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 5:")
poly5 = PolynomialRegression(degree=5, learning_rate=0.1)
poly5.fit(X_poly_norm, y_poly_norm, epochs=2000, print_every=500)
print(f"  R-squared: {poly5.r_squared(X_poly_norm, y_poly_norm):.4f}")

print("\nDegree 2 fits the true curve well. Degree 5 fits training data slightly better")
print("but risks overfitting on new data.")
```

### 步骤 6：Ridge regression (岭回归) (L2 regularization (正则化))

```python
class RidgeRegression:
    def __init__(self, n_features, learning_rate=0.01, alpha=1.0):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = learning_rate
        self.alpha = alpha

    def predict_single(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, X):
        return [self.predict_single(x) for x in X]

    def fit(self, X, y, epochs=1000, print_every=200):
        n = len(y)
        n_features = len(X[0])
        for epoch in range(epochs):
            predictions = self.predict(X)
            errors = [pred - actual for pred, actual in zip(predictions, y)]
            mse = sum(e ** 2 for e in errors) / n
            reg_term = self.alpha * sum(w ** 2 for w in self.weights)
            cost = mse + reg_term
            for j in range(n_features):
                grad = (2 / n) * sum(errors[i] * X[i][j] for i in range(n))
                grad += 2 * self.alpha * self.weights[j]
                self.weights[j] -= self.lr * grad
            grad_b = (2 / n) * sum(errors)
            self.bias -= self.lr * grad_b
            if epoch % print_every == 0:
                print(f"  Epoch {epoch:4d} | Cost: {cost:.4f} | L2 penalty: {reg_term:.4f}")
        return self


print("\n=== Ridge Regression (L2 Regularization) ===")
print("Same data as multiple regression, with alpha=0.1")
ridge = RidgeRegression(n_features=3, learning_rate=0.01, alpha=0.1)
ridge.fit(X_scaled, y_scaled, epochs=1000, print_every=200)
print(f"\nRidge weights: {[round(w, 4) for w in ridge.weights]}")
print(f"Plain weights: {[round(w, 4) for w in multi_model.weights]}")
print("Ridge weights are smaller (shrunk toward zero) due to the L2 penalty.")
```

## 使用它

现在用 scikit-learn 做同样的事情，这是你在生产中实际会使用的。

```python
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

np.random.seed(42)
X_sk = np.random.uniform(0, 10, (100, 1))
y_sk = 3.0 * X_sk.squeeze() + 7.0 + np.random.normal(0, 2.0, 100)

X_train, X_test, y_train, y_test = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

lr = SklearnLR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("=== Scikit-learn Linear Regression ===")
print(f"Coefficient (w): {lr.coef_[0]:.4f}")
print(f"Intercept (b): {lr.intercept_:.4f}")
print(f"R-squared (test): {r2_score(y_test, y_pred):.4f}")
print(f"MSE (test): {mean_squared_error(y_test, y_pred):.4f}")

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_sk = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

lr_poly = SklearnLR()
lr_poly.fit(X_poly_sk, y_train)
print(f"\nPolynomial degree 2 R-squared: {r2_score(y_test, lr_poly.predict(X_poly_test)):.4f}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
print(f"Ridge R-squared: {r2_score(y_test, ridge.predict(X_test_scaled)):.4f}")
print(f"Ridge coefficient: {ridge.coef_[0]:.4f}")
```

你的从零实现和 scikit-learn 产生相同的结果。区别在于：scikit-learn 处理边界情况、数值稳定性和性能优化。生产中使用库。使用从零实现来理解发生了什么。

## 交付物

本课产出：
- `outputs/skill-regression.md` - 一个根据问题选择正确回归方法的技能

## 练习

1. 实现 batch gradient descent (批量梯度下降)、stochastic gradient descent (SGD) (随机梯度下降) 和 mini-batch gradient descent (小批量梯度下降)。在同一数据集上比较收敛速度。哪个收敛最快？哪个的 cost curve 最平滑？
2. 从三次函数生成数据 (y = ax^3 + bx^2 + cx + d + noise)。拟合 1 次、3 次和 10 次多项式。比较训练 R^2 和测试 R^2。在哪个次数时 overfitting (过拟合) 变得明显？
3. 实现 Lasso regression (L1 regularization (正则化): penalty = alpha * sum(|w_i|))。在多特征房屋数据上训练。比较哪些权重变为零 vs Ridge。为什么 L1 产生稀疏解而 L2 不会？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Linear regression (线性回归) | "画一条穿过数据的线" | 找到权重 w 和偏置 b，使得 wx+b 与实际 y 值之间平方差之和最小 |
| Cost function (损失函数) | "模型有多差" | 将模型参数映射到单一数字的函数，衡量预测误差，优化过程使其最小化 |
| Mean squared error (均方误差) | "平方误差的平均值" | (1/n) * (predicted - actual)^2 之和，对大错误的惩罚不成比例 |
| Gradient descent (梯度下降) | "向山下走" | 使用偏导数，沿降低 cost function (损失函数) 的方向迭代调整参数 |
| Learning rate (学习率) | "步长" | 控制每次 gradient descent (梯度下降) 步中参数变化多少的标量 |
| Normal equation (正规方程) | "直接求解" | 闭式解 w = (X^T X)^-1 X^T y，无需迭代即可给出最优权重 |
| R-squared (R平方) | "拟合有多好" | 模型解释的 y 方差比例，范围从负无穷到 1.0 |
| Feature scaling (特征缩放) | "让特征可比" | 将特征转换到相似范围（如零均值、单位方差），使 gradient descent (梯度下降) 收敛更快 |
| Regularization (正则化) | "惩罚复杂度" | 在 cost function (损失函数) 中添加一项来收缩权重，防止 overfitting (过拟合) |
| Ridge regression (岭回归) | "L2 regularization (正则化)" | 在 MSE 上添加 lambda * sum(w_i^2) 惩罚的 linear regression (线性回归) |
| Polynomial regression (多项式回归) | "用线性数学拟合曲线" | 对 polynomial features (多项式特征) (x, x^2, x^3, ...) 进行 linear regression (线性回归)，在权重上仍是线性的 |
| Overfitting (过拟合) | "记忆训练数据" | 使用过于复杂的模型拟合训练数据中的噪声，在新数据上失败 |

## 延伸阅读

- [An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/) -- 免费 PDF，第 3 和第 6 章涵盖 linear regression (线性回归) 和 regularization (正则化)，有实用的 R 示例
- [The Elements of Statistical Learning (ESL)](https://hastie.su.domains/ElemStatLearn/) -- 免费 PDF，ISLR 的更数学化的伴侣，对 ridge 和 lasso 有更深入的讲解
- [Stanford CS229 Lecture Notes on Linear Regression](https://cs229.stanford.edu/main_notes.pdf) -- Andrew Ng 的笔记，从头推导 normal equation (正规方程) 和 gradient descent (梯度下降)
- [scikit-learn LinearRegression documentation](https://scikit-learn.org/stable/modules/linear_model.html) -- LinearRegression、Ridge、Lasso 和 ElasticNet 的实用参考，含代码示例
