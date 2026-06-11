---
name: skill-pytorch-patterns
description: PyTorch 训练、评估和部署的参考模式
version: 1.0.0
phase: 03
lesson: 11
tags: [pytorch, training, deep-learning, gpu, patterns]
---

## 标准训练循环

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
```

## 混合精度训练

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

使用场景：在支持 float16 的 GPU 硬件（V100、A100、H100、RTX 3090+）上训练。预期约 1.5-2 倍加速和约 50% 内存减少。

## 梯度累积

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

使用场景：当 effective batch size (有效批量大小)需要大于 GPU 内存允许的范围时。将 loss 除以 accumulation_steps 可以保持 gradient (梯度)尺度一致。

## 保存和加载

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}, "checkpoint.pt")

checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

恢复训练时始终保存 optimizer (优化器)状态。仅用于 inference (推理)时，只需保存 `model.state_dict()`。

## 自定义 Dataset (数据集)

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = self._load_samples(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def _load_samples(self, data_dir):
        ...
```

## DataLoader (数据加载器)配置

```python
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)
```

| 参数 | 作用 | 何时使用 |
|-----------|-------------|-------------|
| num_workers=4 | 并行数据加载 | 多核机器上始终使用 |
| pin_memory=True | 页锁定 CPU 内存 | 在 GPU 上训练时 |
| drop_last=True | 丢弃不完整的最后一个 batch | 使用 BatchNorm (批归一化)时 |
| persistent_workers=True | 跨 epoch 保持 worker 存活 | 当 num_workers > 0 时 |

## Learning Rate (学习率) 调度

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.1,
)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        ...
        optimizer.step()
        scheduler.step()
```

OneCycleLR：大多数任务的最佳默认选择。先 warmup (预热)到 max_lr，然后 cosine decay (余弦衰减)。每次 batch 后调用 `scheduler.step()`，不是每个 epoch。

## 权重初始化

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

model.apply(init_weights)
```

## Inference (推理) 模式

```python
model.eval()

with torch.inference_mode():
    outputs = model(inputs)
```

`torch.inference_mode()` 比 `torch.no_grad()` 更快，因为它完全禁用 autograd (自动微分)，而不只是抑制 gradient (梯度)计算。

## 常见错误检查清单

1. 在 CrossEntropyLoss 之前应用 softmax（它内部已包含 log_softmax）
2. 验证时忘记调用 model.eval()
3. 忘记将 tensor (张量)移到与模型相同的 device (设备)
4. 没有调用 optimizer.zero_grad()（gradient (梯度)默认会累积）
5. 训练时使用 torch.no_grad()（会禁用 gradient (梯度)计算）
6. num_workers 设置过高（产生过多进程，内存抖动）
7. 在 GPU 上训练时没有使用 pin_memory=True
8. 保存整个模型对象而不是 state_dict（重构代码后会失效）
