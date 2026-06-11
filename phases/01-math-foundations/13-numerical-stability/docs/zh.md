# Numerical Stability (数值稳定性)

> Floating point (浮点数) 是一个漏水的抽象。它会在训练期间咬你一口，而你不会看到它的到来。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- 使用 max-subtraction trick (最大值减法技巧) 实现数值稳定的 softmax 和 log-sum-exp
- 识别 floating-point computation (浮点计算) 中的 overflow (上溢)、underflow (下溢) 和 catastrophic cancellation (灾难性抵消)
- 使用 centered finite differences (中心有限差分) 验证 analytical gradients (解析梯度) 与 numerical gradients (数值梯度)
- 解释为什么 bfloat16 比 float16 更受训练青睐，以及 loss scaling (损失缩放) 如何防止 gradient underflow (梯度下溢)

## The Problem

你的模型训练了三小时，然后 loss 变成 NaN。你添加 print。Step 9,000 的 logits 正常。Step 9,001 它们是 `inf`。到 step 9,002 每个 gradient 都是 `nan`，训练死亡。

或者：你的模型训练完成，但准确率比论文声称的低 2%。你检查了一切。架构匹配。Hyperparameters (超参数) 匹配。数据匹配。问题是论文使用 float32，而你在没有正确 scaling 的情况下使用了 float16。三十二位累积的 rounding error (舍入误差) 悄悄地吞噬了你的准确率。

或者：你从头实现 cross-entropy loss (交叉熵损失)。它在小的 logits 上工作。当 logits 超过 100 时，它返回 `inf`。Softmax 溢出了，因为 `exp(100)` 大于 float32 能表示的范围。每个 ML 框架都通过一个两行 trick 处理这个。你不知道这个 trick 存在。

Numerical stability (数值稳定性) 不是理论关切。它是训练成功与静默失败之间的区别。你最终要调试的每个严重的 ML bug 都归结为 floating point。

## The Concept

### IEEE 754: 计算机如何存储实数

计算机按照 IEEE 754 标准将实数存储为 floating point (浮点数)。一个 float 有三部分：sign bit (符号位)、exponent (指数) 和 mantissa (尾数 / 有效数)。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Mantissa 决定 precision (精度)（有多少有效数字）。Exponent 决定 range (范围)（数字可以有多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 提供约 7 位十进制精度。这意味着它能区分 1.0000001 和 1.0000002，但不能区分 1.00000001 和 1.00000002。7 位之后，一切都是舍入噪声。

float16 提供约 3 位。它能表示的最大数是 65,504。这对 ML 来说小得惊人，其中 logits、gradients 和 activations 经常超过这个值。

bfloat16 是 Google 对 float16 range 问题的答案。它有与 float32 相同的 8 位 exponent（相同范围，最大到 3.4e38），但只有 7 位 mantissa（比 float16 精度低）。对训练神经网络来说，range 比 precision 更重要，所以 bfloat16 通常胜出。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点中精确表示。在二进制中，它是一个循环小数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 将其截断为 23 位 mantissa。存储值约为 0.100000001490116。类似地，0.2 存储为约 0.200000002980232。它们的和是 0.300000004470348，不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对 ML 很重要，因为：

1. 像 `if loss < threshold` 这样的 loss 比较可能给出错误答案
2. 累积许多小值（数千步的 gradient update）会从真实和偏离
3. Checksum 和 reproducibility test 如果你用 `==` 比较 float 会失败

修复：永远不要用 `==` 比较 float。使用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### Catastrophic Cancellation (灾难性抵消)

当你减去两个几乎相等的浮点数时，有效数字抵消，剩下的舍入噪声被提升为前导数字。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

这就是单次减法产生 19% 相对误差的原因。在 ML 中，这发生在：

- 计算具有大 mean 的数据的 variance：`E[x^2] - E[x]^2` 当 E[x] 很大时
- 减去几乎相等的 log-probabilities
- 用太小的 epsilon 计算 finite-difference gradients

修复：重新排列公式以避免减去大的、几乎相等的数。对 variance，使用 Welford algorithm (Welford 算法) 或先中心化数据。对 log-probabilities，全程在对数空间工作。

### Overflow (上溢) 和 Underflow (下溢)

Overflow 发生在结果太大而无法表示时。Underflow 发生在它太小（比最小的可表示正数更接近零）。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函数是 ML 中 overflow 的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函数朝另一个方向：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在 cross-entropy、log-likelihoods 和 KL divergence 中。没有正确 trick 的 `log(exp(x))` 组合是一个雷区。

### Log-Sum-Exp Trick (对数求和指数技巧)

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 会 overflow。如果所有 `x_i` 都很负，每个 `exp(x_i)` 都会 underflow 到零，然后 `log(0)` 是 `-inf`。

Trick：在取指数前减去最大值。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么有效：减去 `max(x)` 后，最大的 exponent 是 `exp(0) = 1`。不可能 overflow。至少有一项是 1，所以和至少为 1，`log(1) = 0`。不可能 underflow 到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

设 `c = max(x)`，overflow 被消除。

这个 trick 在 ML 中无处不在：
- Softmax normalization
- Cross-entropy loss 计算
- Sequence model 中的 log-probability 求和
- Mixture of Gaussians
- Variational inference

### 为什么 Softmax 需要 Max-Subtraction Trick (最大值减法技巧)

Softmax 将 logits 转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

没有 trick，logits [100, 101, 102] 会导致 overflow：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用 trick，减去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

概率完全相同。计算是安全的。这不是优化。它是正确性的要求。

### NaN 和 Inf：检测与预防

`nan` (Not a Number) 和 `inf` (infinity) 像病毒一样在计算中传播。一个 gradient update 中的 `nan` 会使 weight 变成 `nan`，从而使每个后续输出变成 `nan`。训练在一步内死亡。

`inf` 如何出现：
- 大正数的 `exp()`
- 除以零：`1.0 / 0.0`
- `float32` accumulation 中的 overflow

`nan` 如何出现：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 任何涉及现有 `nan` 的算术

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 将 `exp()` 的输入限制：`exp(clamp(x, -80, 80))`
2. 分母加 epsilon：`x / (y + 1e-8)`
3. `log()` 内部加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、stable softmax）
5. Gradient clipping (梯度裁剪) 防止 weight explosion
6. 调试期间每次 forward pass 后检查 `nan`/`inf`

### Numerical Gradient Checking (数值梯度检查)

Analytical gradients (来自 backpropagation / 反向传播) 可能有 bug。Numerical gradient checking 通过 finite differences (有限差分) 计算 gradient 来验证它们。

Centered difference formula (中心差分公式)：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

这是 O(h^2) 精度，远好于 forward difference `(f(x+h) - f(x)) / h`，后者只有 O(h)。

选择 h：太大近似错误。太小则 catastrophic cancellation 破坏答案。`h = 1e-5` 到 `1e-7` 是典型值。

检查：计算 analytical 与 numerical gradient 之间的 relative difference (相对差异)。

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- relative_error < 1e-7：完美，gradient 正确
- relative_error < 1e-5：可接受，可能正确
- relative_error > 1e-3：有问题
- relative_error > 1：gradient 完全错误

实现新 layer 或 loss function 时总是检查 gradients。PyTorch 为此提供 `torch.autograd.gradcheck()`。

### Mixed Precision Training (混合精度训练)

现代 GPU 有专门的硬件（Tensor Cores），计算 float16 matrix multiplication 比 float32 快 2-8 倍。Mixed precision training (混合精度训练) 利用这一点：

```
1. 维护 float32 master copy of weights
2. Forward pass 使用 float16（快）
3. Loss 使用 float32 计算（防止 overflow）
4. Backward pass 使用 float16（快）
5. Gradients 缩放到 float32
6. 更新 float32 master weights
```

纯 float16 训练的问题：gradients 通常非常小（1e-8 或更小）。Float16 将低于 ~6e-8 的一切 underflow 为零。你的模型停止学习，因为所有 gradient update 都是零。

修复是 loss scaling (损失缩放)：

```
1. Loss 乘以一个大的 scale factor（如 1024）
2. Backward pass 计算 (loss * 1024) 的 gradients
3. 所有 gradients 都大了 1024 倍（被推入 float16 underflow 之上）
4. 更新 weight 前将 gradients 除以 1024
5. 净效果：相同的更新，但没有 underflow
```

Dynamic loss scaling (动态损失缩放) 自动调整 scale factor。从一个大的值（65536）开始。如果 gradients overflow 到 `inf`，将其减半。如果 N 步没有 overflow，将其加倍。

### bfloat16 vs float16: 为什么 bfloat16 在训练中胜出

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 有更多 precision（10 位 mantissa vs 7 位），但 range 有限（最大 ~65,504）。bfloat16 precision 较低，但 range 与 float32 相同（最大 ~3.4e38）。

对训练神经网络：

- Activations 和 logits 在训练 spike 期间经常超过 65,504。float16 overflow；bfloat16 处理它。
- float16 需要 loss scaling，而 bfloat16 通常不需要，因为它的 range 覆盖了 gradient magnitude 谱。
- bfloat16 是对 float32 的简单截断：丢弃 mantissa 的底部 16 位。转换是简单的，在 exponent 上无损。

float16 更适合 inference，其中值有界且 precision 更重要。bfloat16 更适合训练，其中 range 更重要。这就是 TPU 和现代 NVIDIA GPU（A100、H100）具有原生 bfloat16 支持的原因。

### Gradient Clipping (梯度裁剪)

Exploding gradients (梯度爆炸) 发生在 gradients 通过多层指数增长时（在 RNN、深层网络和 transformer 中常见）。一个大的 gradient 可以在一步内破坏所有 weight。

两种 clipping 类型：

**Clip by value (按值裁剪)：** 独立限制每个 gradient 元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单，但会改变 gradient vector 的方向。

**Clip by norm (按范数裁剪)：** 缩放整个 gradient vector，使其 norm 不超过 threshold。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留 gradient 的方向。这就是 `torch.nn.utils.clip_grad_norm_()` 所做的。它是标准选择。

典型值：transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，简单网络用 `max_norm=5.0`。

Gradient clipping 不是 hack。它是一种安全机制。没有它，单个异常 batch 可以产生足以毁掉数周训练的 gradient。

### Normalization Layers as Numerical Stabilizers (归一化层作为数值稳定器)

Batch normalization (批归一化)、layer normalization (层归一化) 和 RMS normalization 通常被介绍为帮助训练收敛的 regularizers (正则化器)。它们也是 numerical stabilizers (数值稳定器)。

没有 normalization，activations 可以通过层指数增长：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization 在每一层重新居中并 rescale activations：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常 1e-5）防止所有 activations 相同时除以零。学习到的参数 `gamma` 和 `beta` 让网络恢复它需要的任何 scale。

这使值在整个网络中保持在数值安全的范围内，防止 forward pass 中的 overflow 和 backward pass 中的 gradient explosion。

### 常见的 ML 数值 Bug

**Bug: Loss 在几个 epoch 后变为 NaN。**
原因：logits 增长太大，softmax overflow。或 learning rate (学习率) 太高，weight 发散。
修复：使用 stable softmax（max subtraction）、降低 learning rate、添加 gradient clipping。

**Bug: Loss 卡在 log(num_classes)。**
原因：模型输出接近均匀概率。通常意味着 gradients 正在消失或模型根本没有学习。
修复：检查数据 label 是否正确，验证 loss function，检查 dead ReLUs。

**Bug: Validation accuracy 比预期低 1-3%。**
原因：mixed precision 没有 proper loss scaling。Gradient underflow 静默地将小的 update 归零。
修复：启用 dynamic loss scaling，或切换到 bfloat16。

**Bug: 某些层的 gradient norm 为 0.0。**
原因：dead ReLU neurons（所有输入为负），或 float16 underflow。
修复：使用 LeakyReLU 或 GELU，使用 gradient scaling，检查 weight initialization。

**Bug: 模型在一个 GPU 上工作，但在另一个上给出不同结果。**
原因：非确定性的 floating point accumulation order。GPU parallel reductions 在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
修复：接受微小差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度损失。

**Bug: `exp()` 在 loss 计算中返回 `inf`。**
原因：raw logits 未经 max-subtraction trick 直接传入 `exp()`。
修复：使用 `torch.nn.functional.log_softmax()`，它在内部实现了 log-sum-exp。

**Bug: 从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示低于 6e-8 的 gradient magnitudes 或高于 65,504 的 activations。
修复：使用 mixed precision 配合 loss scaling (AMP)，或改用 bfloat16。

## Build It

### Step 1: 演示浮点精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: 实现 naive vs stable softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3: 实现稳定的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4: 实现稳定的 cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: Gradient checking (梯度检查)

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### Mixed precision simulation (混合精度模拟)

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping (梯度裁剪)

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection (检测)

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整实现及所有 edge cases (边界情况) 的演示请参见 `code/numerical.py`。

## Ship It

本节课产出：
- `code/numerical.py`，包含 stable softmax、log-sum-exp、cross-entropy、gradient checking 和 mixed precision simulation
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定实现在 Phase 3 构建训练循环和 Phase 4 实现 attention mechanism 时再次出现。

## Exercises

1. **Catastrophic cancellation (灾难性抵消)。** 在 float32 中使用 naive 公式 `E[x^2] - E[x]^2` 计算 [1000000.0, 1000001.0, 1000002.0] 的 variance (方差)。然后使用 Welford algorithm (Welford 算法)。对比与真实 variance (0.6667) 的误差。

2. **Precision hunt (精度搜寻)。** 找到最小的正 float32 值 `x` 使得 `1.0 + x == 1.0` 在 Python 中成立。这就是 machine epsilon (机器 epsilon)。验证它匹配 `numpy.finfo(numpy.float32).eps`。

3. **Log-sum-exp edge cases (边界情况)。** 测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其余，(c) 所有值非常负 (-1000)。验证它在 naive 版本失败的地方给出正确结果。

4. **Neural network layer 的 gradient checking。** 实现单层 linear layer `y = Wx + b` 及其 analytical backward pass。使用 `numerical_gradient` 验证 3x2 weight matrix 的正确性。

5. **Loss scaling experiment (损失缩放实验)。** 模拟 float16 训练：创建范围 [1e-9, 1e-3] 的随机 gradients，转换为 float16，测量有多少变为零。然后应用 loss scaling (乘以 1024)，转换为 float16，缩回，再次测量零的比例。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| IEEE 754 | "The float standard" | International standard defining binary floating point formats, rounding rules, and special values (inf, nan). Every modern CPU and GPU implements it. |
| Machine epsilon | "The precision limit" | The smallest value e such that 1.0 + e != 1.0 in a given float format. For float32, it is about 1.19e-7. |
| Catastrophic cancellation | "Precision loss from subtraction" | When subtracting nearly equal floating point numbers, significant digits cancel and rounding noise dominates the result. |
| Overflow | "Number too big" | A result exceeds the maximum representable value and becomes inf. exp(89) overflows float32. |
| Underflow | "Number too small" | A result is closer to zero than the smallest representable positive number and becomes 0.0. exp(-104) underflows float32. |
| Log-sum-exp trick | "Subtract the max first" | Computing log(sum(exp(x))) by factoring out exp(max(x)) to prevent overflow and underflow. Used in softmax, cross-entropy, and log-probability math. |
| Stable softmax | "Softmax that does not explode" | Subtracting max(logits) before exponentiating. Numerically identical result, no overflow possible. |
| Gradient checking | "Verify your backprop" | Comparing analytical gradients from backpropagation against numerical gradients from finite differences to catch implementation bugs. |
| Mixed precision | "Float16 forward, float32 backward" | Using lower-precision floats for speed-critical operations and higher-precision floats for numerically sensitive operations. Typical speedup is 2-3x. |
| Loss scaling | "Prevent gradient underflow" | Multiplying the loss by a large constant before backprop so gradients stay in float16's representable range, then dividing by the same constant before weight updates. |
| bfloat16 | "Brain floating point" | Google's 16-bit format with 8 exponent bits (same range as float32) and 7 mantissa bits (less precision than float16). Preferred for training. |
| Gradient clipping | "Cap the gradient norm" | Scaling the gradient vector so its norm does not exceed a threshold. Prevents exploding gradients from ruining weights. |
| NaN | "Not a Number" | Special float value from undefined operations (0/0, inf-inf, sqrt(-1)). Propagates through all subsequent arithmetic. |
| Inf | "Infinity" | Special float value from overflow or division by zero. Can combine to produce NaN (inf - inf, inf * 0). |
| Numerical gradient | "Brute force derivative" | Approximating a derivative by evaluating f(x+h) and f(x-h) and dividing by 2h. Slow but reliable for verification. |

## Further Reading

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- the definitive reference, dense but complete
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- the NVIDIA paper that introduced loss scaling for float16 training
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- practical guide to mixed precision in PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- why Google chose this format for TPUs
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- algorithm for reducing rounding error in floating point sums
