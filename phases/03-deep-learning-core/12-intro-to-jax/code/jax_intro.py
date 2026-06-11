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


def init_params(key):
    # 将单个 PRNG key (随机密钥) 分割为 3 个，每层一个
    k1, k2, k3 = random.split(key, 3)
    # He-initialization：根据输入维度计算缩放因子
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


def forward(params, x):
    # 3 层 MLP (多层感知机)，使用 ReLU (修正线性单元) 激活函数
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x


def loss_fn(params, x, y):
    # 计算 cross-entropy (交叉熵) loss：log_softmax + 负均值
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))


# 使用 Adam (自适应矩估计) optimizer (优化器)
optimizer = optax.adam(learning_rate=1e-3)


@jax.jit
def train_step(params, opt_state, x, y):
    # JIT-compiled (JIT编译) 训练步骤：计算 gradient (梯度) 并更新 parameter (参数)
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jax.jit
def accuracy(params, x, y):
    # JIT-compiled (JIT编译) 准确率计算
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


def train():
    X_train, y_train, X_test, y_test = get_mnist_data()
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)

    # 初始化 PRNG key (随机密钥) 和 parameter (参数)
    key = random.PRNGKey(0)
    params = init_params(key)
    opt_state = optimizer.init(params)

    batch_size = 128
    n_epochs = 10

    for epoch in range(n_epochs):
        # 每个 epoch (轮次) 重新分割 key 并生成新的随机排列
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

    return params


def demo_grad():
    print("=== jax.grad demo ===")

    def f(x):
        return x ** 3

    # grad (梯度函数) 返回一阶导数；嵌套 grad 返回高阶导数
    df = jax.grad(f)
    d2f = jax.grad(df)
    print(f"f(2.0)   = {f(2.0)}")
    print(f"f'(2.0)  = {df(2.0)}")
    print(f"f''(2.0) = {d2f(2.0)}")
    print()


def demo_vmap():
    print("=== jax.vmap demo ===")
    key = random.PRNGKey(42)
    k1, k2 = random.split(key)

    params = {'w': random.normal(k1, (3,)), 'b': 0.0}

    def predict_single(params, x):
        # 单个样本的预测函数
        return jnp.dot(params['w'], x) + params['b']

    # vmap (向量化映射) 自动将单样本函数提升为 batch (批量) 函数
    batch_x = random.normal(k2, (5, 3))
    batch_predict = jax.vmap(predict_single, in_axes=(None, 0))
    results = batch_predict(params, batch_x)
    print(f"Input shape:  {batch_x.shape}")
    print(f"Output shape: {results.shape}")
    print(f"Predictions:  {results}")
    print()


def demo_jit():
    print("=== jax.jit demo ===")
    import time

    key = random.PRNGKey(0)
    x = random.normal(key, (1000, 1000))

    def slow_fn(x):
        # 包含多次矩阵乘法的计算密集型函数
        for _ in range(10):
            x = jnp.dot(x, x)
            x = x / jnp.linalg.norm(x)
        return x

    # jitting (JIT编译) 函数：第一次调用慢（compilation (编译)），后续调用快
    fast_fn = jax.jit(slow_fn)
    _ = fast_fn(x)

    start = time.perf_counter()
    for _ in range(10):
        _ = slow_fn(x)
    eager_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(10):
        _ = fast_fn(x).block_until_ready()
    jit_time = time.perf_counter() - start

    print(f"Eager: {eager_time:.4f}s")
    print(f"JIT:   {jit_time:.4f}s")
    print(f"Speedup: {eager_time / jit_time:.1f}x")
    print()


if __name__ == '__main__':
    demo_grad()
    demo_vmap()
    demo_jit()
    print("=== MNIST Training ===")
    train()
