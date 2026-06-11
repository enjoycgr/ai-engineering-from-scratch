# JAX 入门

> PyTorch 会就地修改 tensor (张量)。TensorFlow 构建计算图。JAX 编译纯函数。最后一种方式会改变你对深度学习的思考方式。

**类型：** Build (构建)
**语言：** Python
**前置知识：** Phase 03 Lessons 01-10，基础 NumPy (NumPy)
**时间：** ~90 分钟

## 学习目标

- 使用 JAX 的函数式 API（jax.numpy、jax.grad、jax.jit、jax.vmap）编写纯函数形式的神经网络代码
- 解释 PyTorch 的 eager mutation (即时修改) 与 JAX 的函数式编译模型之间的关键设计差异
- 应用 JIT compilation (即时编译) 和 vmap (向量化映射) 来加速训练循环，相比朴素 Python 实现
- 在 JAX 中训练一个简单的网络，并将其显式状态管理与 PyTorch 的面向对象方法进行对比

## 问题背景

你已经知道如何在 PyTorch 中构建神经网络。定义一个 `nn.Module`，调用 `.backward()`，执行优化器步进。这很有效，数百万人在使用它。

但 PyTorch 有一个根植于其 DNA 的约束：它逐行追踪操作，每次一个，在 Python 中执行。每一个 `tensor + tensor` 都是一次独立的内核启动。每一步训练都会重新解释相同的 Python 代码。这在训练小规模模型时没问题，但当你需要在 2,048 个 TPU (张量处理器) 上训练一个 5400 亿参数的模型时，这种开销会致命。

Google DeepMind 使用 JAX 训练 Gemini。Anthropic 使用 JAX 训练 Claude。这些不是小规模操作——它们是地球上最大的神经网络训练任务。他们选择 JAX，是因为 JAX 将你的训练循环视为一个可编译的程序，而不是一系列 Python 调用。

JAX 是 NumPy (NumPy) 加上三大超能力：automatic differentiation (自动微分)、JIT compilation (即时编译) 到 XLA (加速线性代数)，以及 automatic vectorization (自动向量化)。你编写一个处理单个样本的函数。JAX 给你一个能处理整个 batch (批量)、计算 gradient (梯度)、编译为机器码、并在多个设备上运行的函数。而且完全不需要修改原始函数。

## 核心概念

### JAX 的设计哲学

JAX 是一个函数式框架。没有类，没有可变状态，没有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| `nn.Module` 类，内部包含状态 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution (即时执行) | 通过 XLA (加速线性代数) 进行 JIT compilation (即时编译) |
| `for x in batch:` 手动循环 | `jax.vmap(f)` 自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自动并行化 |
| 可变的 `model.parameters()` | 不可变的 pytree (嵌套树结构) of array (数组) |

这不是风格偏好。这是编译器的约束。JIT compilation (即时编译) 要求 pure function (纯函数)——相同的输入总是产生相同的输出，没有副作用。正是这个限制使得 100 倍加速成为可能。

### jax.numpy：熟悉的接口

JAX 在加速器上重新实现了 NumPy (NumPy) API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名。相同的广播规则。相同的切片语义。但 array (数组) 位于 GPU (图形处理器)/TPU (张量处理器) 上，并且每个操作都可以被编译器追踪。

一个关键区别：JAX 的 array (数组) 是不可变的。不能写 `a[0] = 5`。取而代之的是：`a = a.at[0].set(5)`。一开始会觉得别扭，但很快就会理解——不可变性正是 `grad`、`jit` 和 `vmap` 能够组合的原因。

### jax.grad：函数式自动微分

PyTorch 将 gradient (梯度) 附加到 tensor (张量) 上（`.grad`）。JAX 将 gradient (梯度) 附加到函数上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接受一个函数，返回一个计算其 gradient (梯度) 的新函数。没有 `.backward()` 调用。没有存储在 tensor (张量) 上的计算图。gradient (梯度) 只是另一个你可以调用、组合或 JIT-compile (JIT编译) 的函数。

这种组合可以任意嵌套：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。Jacobian (雅可比矩阵)。Hessian (黑塞矩阵)。全部通过组合 `grad` 实现。PyTorch 也能做到这些（`torch.autograd.functional.hessian`），但它是后加上去的。在 JAX 中，这是基础。

约束条件：`grad` 只对 pure function (纯函数) 有效。内部不能有 print 语句（它们只在追踪时运行，不在执行时运行）。不能修改外部状态。没有显式的 random key (随机密钥) 管理就不能生成随机数。

### jit：编译到 XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

在第一次调用时，JAX 会 trace (追踪) 该函数——它记录发生了哪些操作，但不实际执行。然后将这个 trace 交给 XLA (Accelerated Linear Algebra，加速线性代数)，Google 专为 TPU (张量处理器) 和 GPU (图形处理器) 设计的编译器。XLA 会融合操作、消除冗余的内存拷贝，并生成优化的机器码。

后续调用完全跳过 Python。编译后的代码以 C++ 速度在加速器上运行。

JIT compilation (即时编译) 适用的场景：
- 训练步骤（相同的计算重复数千次）
- 推理（相同的模型，不同的输入）
- 任何被多次调用且输入形状相似的函数

JIT compilation (即时编译) 不适用的场景：
- 包含依赖于数值的 Python 控制流的函数（`if x > 0`，其中 x 是被 trace 的 array (数组)）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪隐藏了实际执行过程）

控制流限制是真实存在的。`jax.lax.cond` 替代 `if/else`。`jax.lax.scan` 替代 `for` 循环。这些不是可选的——它们是编译的代价。

### vmap：自动向量化

你编写一个处理单个样本的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 将其提升为处理整个 batch (批量)：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的含义是：不要对 `params` 进行 batch (广播)，对 `x` 的第 0 轴进行 batch。没有手动的 `for` 循环。没有 reshape。没有 batch dimension (批量维度) 的传递。JAX 会自动找出 batch dimension 并对整个计算进行向量化。

这不是语法糖。`vmap` 会生成融合的向量化代码，比 Python 循环快 10-100 倍。而且它可以与 `jit` 和 `grad` 组合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐样本 gradient (梯度)。一行代码。这在 PyTorch 中几乎不可能实现，除非使用 hack。

### pmap：跨设备的数据并行

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 将函数复制到所有可用设备（GPU (图形处理器)/TPU (张量处理器)）上，并分割 batch。在函数内部，`jax.lax.pmean` 和 `jax.lax.psum` 会在设备间同步 gradient (梯度)。

Google 使用 `pmap`（及其继任者 `shard_map`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型是：先写单设备版本，然后包上 `pmap`，完成。

### Pytree：通用数据结构

JAX 操作的对象是 "pytree"——list、tuple、dict 和 array (数组) 的嵌套组合。你的模型 parameter (参数) 就是一个 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每个 JAX 变换——`grad`、`jit`、`vmap`——都知道如何遍历 pytree。`jax.tree.map(f, tree)` 将 `f` 应用到每个叶子节点。这就是 optimizer (优化器) 一次性更新所有 parameter (参数) 的方式：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有 `.parameters()` 方法。没有 parameter 注册。树结构本身就是模型。

### 函数式 vs 面向对象

PyTorch 将状态存储在对象内部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用带有显式状态的 pure function (纯函数)：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params 作为参数传入。不存储任何状态。不修改任何状态。这使得每个函数都可测试、可组合、可编译。这也意味着你需要自己管理 params——或者使用 Flax 或 Equinox 这样的库。

### JAX 生态系统

JAX 提供原语。库提供易用性：

| 库 | 角色 | 风格 |
|---------|------|-------|
| **Flax** (Google) | Neural network (神经网络) layer (层) | `nn.Module`，显式状态管理 |
| **Equinox** (Patrick Kidger) | Neural network (神经网络) layer (层) | 基于 pytree，Pythonic |
| **Optax** (DeepMind) | Optimizer (优化器) + LR schedule | 可组合的 gradient (梯度) 变换 |
| **Orbax** (Google) | Checkpointing (检查点) | 保存/恢复 pytree |
| **CLU** (Google) | Metrics + logging | 训练循环工具 |

Optax 是标准的 optimizer (优化器) 库。它将 gradient (梯度) 变换（Adam、SGD、clipping）与 parameter 更新分离，使得组合变得异常简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何时使用 JAX vs PyTorch

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU (张量处理器) 支持 | 原生支持（Google 构建了两者） | 社区维护（torch_xla） |
| GPU (图形处理器) 支持 | 良好（通过 XLA (加速线性代数) 的 CUDA） | 最佳（原生 CUDA） |
| 调试 | 困难（追踪 + 编译） | 简单（eager，可逐行调试） |
| 生态系统 | 研究导向（Flax、Equinox） | 庞大（HuggingFace、torchvision 等） |
| 招聘 | 小众（Google/DeepMind/Anthropic） | 主流（随处可见） |
| 大规模训练 | 更优（XLA、pmap、mesh） | 良好（FSDP、DeepSpeed） |
| 原型开发速度 | 较慢（函数式开销） | 较快（修改即可运行） |
| 生产推理 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 使用者 | DeepMind（Gemini）、Anthropic（Claude） | Meta（Llama）、OpenAI（GPT）、Stability AI |

诚实的答案：除非你有特定理由使用 JAX，否则使用 PyTorch。这些理由包括——TPU (张量处理器) 访问权限、需要 per-example gradient (逐样本梯度)、超大规模多设备训练，或在 Google/DeepMind/Anthropic 工作。

### JAX 中的随机数

JAX 没有全局随机状态。每个随机操作都需要一个显式的 PRNG (Pseudo-Random Number Generator，伪随机数生成器) key (密钥)：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一开始这很烦人。但它保证了跨设备和跨编译的可复现性——这是 PyTorch 的 `torch.manual_seed` 在多 GPU (图形处理器) 设置中无法保证的特性。

## 动手构建

### 步骤 1：环境设置与数据

我们将使用 JAX 和 Optax 在 MNIST 上训练一个 3 层 MLP (多层感知机)。784 个输入，两个隐藏层分别为 256 和 128 个神经元，10 个输出类别。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤 2：初始化参数

没有类。只有一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动实现 He-initialization。三个 PRNG (伪随机数生成器) key (密钥) 从一个种子分割而来。每个权重都是嵌套 dict 中的不可变 array (数组)。

### 步骤 3：前向传播

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Pure function (纯函数)。Params 进，prediction 出。没有 `self`，没有存储状态。`loss_fn` 从头计算 cross-entropy (交叉熵)——softmax、log、负均值。

### 步骤 4：JIT 编译的训练步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 一次性返回 loss 值和 gradient (梯度)。`@jax.jit` 装饰器将两个函数编译到 XLA (加速线性代数)。第一次调用后，每个训练步骤都不再接触 Python。

### 步骤 5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 个 epoch (轮次)。~97% 测试准确率。第一个 epoch 较慢（JIT compilation (即时编译)）。第 2-10 个 epoch 很快。

注意缺少了什么：没有 `.zero_grad()`，没有 `.backward()`，没有 `.step()`。整个更新是一个组合函数调用。gradient (梯度) 被计算、被 Adam 变换、应用到 parameter (参数) 上——全部在 `train_step` 内部完成。

## 使用它

### Flax：Google 的标准方案

Flax 是最常用的 JAX neural network (神经网络) 库。它重新引入了 `nn.Module`，但采用显式状态管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与 PyTorch 相同，但 `params` 与模型分离。`model.init()` 创建 params。`model.apply(params, x)` 运行前向传播。模型对象本身没有状态。

### Equinox：更 Pythonic 的替代方案

Equinox（由 Patrick Kidger 开发）将模型表示为 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是 pytree。不需要 `.apply()`。parameter (参数) 就是模型的叶子节点。这更接近 JAX 的思维方式。

### Optax：可组合的优化器

Optax 将 gradient (梯度) 变换与更新操作解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping (梯度裁剪)、learning rate warmup (学习率预热)、weight decay (权重衰减)——全部作为变换链组合在一起。每个变换看到 gradient (梯度)，修改它，然后传给下一个。没有单一的 optimizer 类。

## 部署上线

**安装：**

```bash
pip install jax jaxlib optax flax
```

GPU (图形处理器) 支持：

```bash
pip install jax[cuda12]
```

TPU (张量处理器)（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次 JIT 调用较慢（compilation (编译)）。在 benchmark 之前先 warm up (预热)。
- 避免在 JIT 内部对 JAX array (数组) 使用 Python 循环。使用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 在 JIT 内部有效。普通 `print()` 无效。
- 使用 `jax.profiler` 或 TensorBoard 进行性能分析。XLA compilation 可能隐藏瓶颈。
- JAX 默认预分配 75% 的 GPU (图形处理器) 内存。设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可禁用。

**检查点保存：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**本课产出：**
- `outputs/prompt-jax-optimizer.md` —— 一个用于选择正确 JAX optimizer (优化器) 配置的 prompt
- `outputs/skill-jax-patterns.md` —— 一个涵盖 JAX 函数式模式的 skill

## 练习题

1. 为 MLP 添加 dropout (随机失活)。在 JAX 中，dropout 需要一个 PRNG key (随机密钥)——将 key 贯穿前向传播，并为每个 dropout layer (层) 分割 key。对比添加前后的测试准确率。

2. 使用 `jax.vmap` 计算 32 张 MNIST 图像的 per-example gradient (逐样本梯度)。计算每个样本的 gradient norm (梯度范数)。哪些样本的 gradient 最大，为什么？

3. 将手动的前向函数替换为通用的 `mlp_forward(params, x)`，使其适用于任意层数。使用 `jax.tree.leaves` 自动推断深度。

4. 对训练步骤进行 benchmark，对比有和没有 `@jax.jit` 的情况。各计时 100 步。在你的硬件上加速比是多少？第一次调用的 compilation (编译) 开销是多少？

5. 通过组合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 实现 gradient clipping (梯度裁剪)。对比有和没有裁剪的训练效果。绘制训练过程中的 gradient norm (梯度范数) 变化图。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|----------------|----------------------|
| XLA (加速线性代数) | "让 JAX 变快的东西" | Accelerated Linear Algebra (加速线性代数) —— 一个编译器，融合操作并从计算图生成优化的 GPU (图形处理器)/TPU (张量处理器) 内核 |
| JIT (即时编译) | "Just-in-time compilation (即时编译)" | JAX 在第一次调用时 trace (追踪) 函数，编译到 XLA (加速线性代数)，后续调用运行编译后的版本 |
| Pure function (纯函数) | "无副作用" | 输出仅依赖于输入的函数——没有全局状态，没有修改，没有显式 key 就不生成随机数 |
| vmap (向量化映射) | "自动批处理" | 将处理单个样本的函数转换为处理整个 batch (批量) 的函数，无需重写 |
| pmap (并行映射) | "自动并行化" | 将函数复制到多个设备并分割输入 batch (批量) |
| Pytree | "嵌套 dict of array (数组)" | JAX 可以遍历和变换的 list、tuple、dict 和 array (数组) 的任意嵌套结构 |
| Tracing (追踪) | "记录计算过程" | JAX 用抽象值执行函数以构建计算图，不计算真实结果 |
| Functional autodiff (函数式自动微分) | "函数的 grad" | 通过变换函数来计算导数，而不是将 gradient (梯度) 存储附加到 tensor (张量) 上 |
| Optax | "JAX 的 optimizer (优化器) 库" | 可组合的 gradient (梯度) 变换库——Adam、SGD、clipping、scheduling——可以链式组合 |
| Flax | "JAX 的 nn.Module" | Google 的 JAX neural network (神经网络) 库，添加 layer (层) 抽象同时保持状态显式 |

## 延伸阅读

- JAX 文档：https://jax.readthedocs.io/ —— 官方文档，包含关于 grad、jit 和 vmap 的优秀教程
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) —— 解释设计哲学的原始论文
- Flax 文档：https://flax.readthedocs.io/ —— Google 的 JAX neural network (神经网络) 库
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) —— Flax 的更 Pythonic 的替代方案
- DeepMind, "Optax: composable gradient transformation and optimisation" —— 标准 optimizer (优化器) 库
- "You Don't Know JAX" (Colin Raffel, 2020) —— 来自 T5 作者之一的 JAX 陷阱和模式实用指南
