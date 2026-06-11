import torch
import torch.nn as nn
import struct
import gzip
import urllib.request
import os
import time


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_mnist(path="./mnist_data"):
    """下载 MNIST 原始 gzip 文件到指定目录（如果不存在）。"""
    os.makedirs(path, exist_ok=True)
    for f in MNIST_FILES:
        filepath = os.path.join(path, f)
        if not os.path.exists(filepath):
            print(f"  正在下载 {f}...")
            urllib.request.urlretrieve(MNIST_BASE_URL + f, filepath)


def load_images(filepath):
    """从 gzip 文件解析 MNIST 图片并返回归一化后的 tensor (张量)。"""
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        images = images.reshape(num, rows * cols).float() / 255.0
    return images


def load_labels(filepath):
    """从 gzip 文件解析 MNIST 标签并返回 long 类型的 tensor (张量)。"""
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = f.read()
        labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).long()
    return labels


class MNISTModel(nn.Module):
    """3 层 MLP，使用 Dropout (随机失活)进行正则化。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class MNISTModelWithBatchNorm(nn.Module):
    """3 层 MLP，使用 BatchNorm1d (批归一化)替代 Dropout (随机失活)。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch。返回平均 loss (损失)和准确率。"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """在验证/测试集上评估模型。使用 torch.no_grad() 禁用 gradient (梯度)追踪。"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def load_data(data_path="./mnist_data"):
    """下载（如需要）并加载 MNIST 训练集和测试集。"""
    download_mnist(data_path)
    train_images = load_images(os.path.join(data_path, "train-images-idx3-ubyte.gz"))
    train_labels = load_labels(os.path.join(data_path, "train-labels-idx1-ubyte.gz"))
    test_images = load_images(os.path.join(data_path, "t10k-images-idx3-ubyte.gz"))
    test_labels = load_labels(os.path.join(data_path, "t10k-labels-idx1-ubyte.gz"))
    return train_images, train_labels, test_images, test_labels


def create_loaders(train_images, train_labels, test_images, test_labels, batch_size=64):
    """从 tensor (张量)创建 DataLoader (数据加载器)。"""
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False
    )
    return train_loader, test_loader


def run_experiment(name, model, train_loader, test_loader, optimizer, device, epochs=10):
    """运行一个完整实验：训练、评估、计时。"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")
    print(f"  优化器:  {optimizer.__class__.__name__}")
    print(f"  设备:     {device}")
    print()

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        print(
            f"  Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    elapsed = time.time() - start_time
    print(f"\n  耗时: {elapsed:.1f}s ({elapsed/epochs:.1f}s/epoch)")
    print(f"  最终测试准确率: {test_acc:.4f}")
    return test_acc


def experiment_adam(train_loader, test_loader, device):
    """实验 1：Adam optimizer (优化器) + Dropout (随机失活)。"""
    model = MNISTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return run_experiment(
        "实验 1: Adam + Dropout",
        model, train_loader, test_loader, optimizer, device
    )


def experiment_sgd(train_loader, test_loader, device):
    """实验 2：SGD (随机梯度下降) + Momentum + Dropout (随机失活)。"""
    model = MNISTModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return run_experiment(
        "实验 2: SGD + Momentum + Dropout",
        model, train_loader, test_loader, optimizer, device
    )


def experiment_batchnorm(train_loader, test_loader, device):
    """实验 3：Adam + BatchNorm (批归一化，无 Dropout)。"""
    model = MNISTModelWithBatchNorm().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return run_experiment(
        "实验 3: Adam + BatchNorm (无 dropout)",
        model, train_loader, test_loader, optimizer, device
    )


def experiment_sgd_cosine(train_loader, test_loader, device, epochs=10):
    """实验 4：SGD + CosineAnnealingLR learning rate (学习率)调度。"""
    model = MNISTModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  实验 4: SGD + Cosine LR Schedule")
    print(f"{'='*60}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {num_params:,}")
    print(f"  优化器:  SGD (lr=0.05, momentum=0.9) + CosineAnnealing")
    print(f"  设备:     {device}")
    print()

    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    test_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch+1:2d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        scheduler.step()

    elapsed = time.time() - start_time
    print(f"\n  耗时: {elapsed:.1f}s ({elapsed/epochs:.1f}s/epoch)")
    print(f"  最终测试准确率: {test_acc:.4f}")
    return test_acc


def show_model_info(model, name="Model"):
    """打印模型架构和各层的 parameter (参数)数量。"""
    print(f"\n  {name} 架构:")
    print(f"  {'-'*40}")
    total = 0
    for pname, param in model.named_parameters():
        print(f"    {pname:30s} {str(list(param.shape)):15s} ({param.numel():,} params)")
        total += param.numel()
    print(f"  {'-'*40}")
    print(f"    总计: {total:,} 个参数")


def demo_tensor_basics():
    """演示 tensor (张量)的创建、dtype 转换和 reshape 操作。"""
    print(f"\n{'='*60}")
    print(f"  Tensor (张量)基础")
    print(f"{'='*60}")

    x = torch.randn(3, 4)
    print(f"\n  torch.randn(3, 4):")
    print(f"    shape={x.shape}, dtype={x.dtype}, device={x.device}")

    x_int = x.to(torch.int8)
    print(f"\n  .to(torch.int8):")
    print(f"    dtype={x_int.dtype}")

    y = x.view(2, 6)
    print(f"\n  .view(2, 6):")
    print(f"    shape={y.shape}")

    z = x.unsqueeze(0)
    print(f"\n  .unsqueeze(0):")
    print(f"    shape={z.shape}")


def demo_autograd():
    """演示 autograd (自动微分)：计算 gradient (梯度)并执行手动参数更新。"""
    print(f"\n{'='*60}")
    print(f"  Autograd (自动微分)演示")
    print(f"{'='*60}")

    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2 + 3 * x
    z = y.sum()
    z.backward()

    print(f"\n  x = [1.0, 2.0, 3.0]")
    print(f"  y = x^2 + 3x")
    print(f"  z = sum(y) = {z.item():.1f}")
    print(f"  dz/dx = 2x + 3 = {x.grad.tolist()}")

    w = torch.randn(3, requires_grad=True)
    for step in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f"\n  第 {step} 步: loss={loss.item():.4f}, grad={w.grad.tolist()}")
        with torch.no_grad():
            w -= 0.1 * w.grad
        w.grad.zero_()


if __name__ == "__main__":
    print("=" * 60)
    print("  PyTorch 简介 -- 第 3 阶段, 第 11 课")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  PyTorch 版本: {torch.__version__}")
    print(f"  设备: {device}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    demo_tensor_basics()
    demo_autograd()

    print(f"\n{'='*60}")
    print(f"  正在加载 MNIST...")
    print(f"{'='*60}")

    train_images, train_labels, test_images, test_labels = load_data()
    print(f"  训练集: {train_images.shape[0]:,} 张图片")
    print(f"  测试集:  {test_images.shape[0]:,} 张图片")
    print(f"  图片维度: {train_images.shape[1]} 个特征 (28x28 展平)")
    print(f"  类别: {train_labels.unique().tolist()}")

    train_loader, test_loader = create_loaders(
        train_images, train_labels, test_images, test_labels
    )

    model_preview = MNISTModel()
    show_model_info(model_preview, "MNISTModel (Dropout)")

    model_preview_bn = MNISTModelWithBatchNorm()
    show_model_info(model_preview_bn, "MNISTModel (BatchNorm)")

    acc_adam = experiment_adam(train_loader, test_loader, device)
    acc_sgd = experiment_sgd(train_loader, test_loader, device)
    acc_bn = experiment_batchnorm(train_loader, test_loader, device)
    acc_cosine = experiment_sgd_cosine(train_loader, test_loader, device)

    print(f"\n{'='*60}")
    print(f"  总结")
    print(f"{'='*60}")
    print(f"  Adam + Dropout:           {acc_adam:.4f}")
    print(f"  SGD + Momentum + Dropout: {acc_sgd:.4f}")
    print(f"  Adam + BatchNorm:         {acc_bn:.4f}")
    print(f"  SGD + Cosine Schedule:    {acc_cosine:.4f}")
    print()

    best_model = MNISTModel().to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        train_one_epoch(best_model, train_loader, criterion, optimizer, device)

    torch.save(best_model.state_dict(), "mnist_mlp.pt")
    print(f"  模型已保存到 mnist_mlp.pt")

    loaded_model = MNISTModel().to(device)
    loaded_model.load_state_dict(
        torch.load("mnist_mlp.pt", map_location=device, weights_only=True)
    )
    _, loaded_acc = evaluate(loaded_model, test_loader, criterion, device)
    print(f"  加载后模型测试准确率: {loaded_acc:.4f}")
