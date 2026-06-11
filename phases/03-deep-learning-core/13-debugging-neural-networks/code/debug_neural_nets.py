import torch
import torch.nn as nn
import math
import copy


class NetworkDebugger:
    """Hook 到 PyTorch 模型以记录每层的 activation (激活) 和 gradient (梯度) 统计。"""

    def __init__(self, model):
        self.model = model
        self.activation_stats = {}
        self.gradient_stats = {}
        self.loss_history = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """为 Linear、Conv2d、ReLU 和 LeakyReLU 层注册 forward 和 backward hooks。"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.LeakyReLU)):
                hook = module.register_forward_hook(self._make_activation_hook(name))
                self.hooks.append(hook)
                hook = module.register_full_backward_hook(self._make_gradient_hook(name))
                self.hooks.append(hook)

    def _make_activation_hook(self, name):
        """创建一个 hook 以捕获 activation (激活) 统计（mean、std、zero fraction）。"""
        def hook(module, input, output):
            with torch.no_grad():
                out = output.detach().float()
                self.activation_stats[name] = {
                    "mean": out.mean().item(),
                    "std": out.std().item(),
                    "fraction_zero": (out == 0).float().mean().item(),
                    "min": out.min().item(),
                    "max": out.max().item(),
                }
        return hook

    def _make_gradient_hook(self, name):
        """创建一个 hook 以捕获 gradient (梯度) 统计（mean、std、abs_mean、max）。"""
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                with torch.no_grad():
                    grad = grad_output[0].detach().float()
                    self.gradient_stats[name] = {
                        "mean": grad.mean().item(),
                        "std": grad.std().item(),
                        "abs_mean": grad.abs().mean().item(),
                        "max": grad.abs().max().item(),
                    }
        return hook

    def record_loss(self, loss_value):
        """记录一个 loss 值供后续分析。"""
        self.loss_history.append(loss_value)

    def check_loss_health(self):
        """检查 loss history 中是否有 NaN/Inf、平台期或振荡。

        返回:
            "HEALTHY" | "NOT_ENOUGH_DATA" | "NAN_OR_INF" | "NOT_DECREASING" | "OSCILLATING"
        """
        if len(self.loss_history) < 2:
            return "NOT_ENOUGH_DATA"
        recent = self.loss_history[-10:]
        if any(math.isnan(v) or math.isinf(v) for v in recent):
            return "NAN_OR_INF"
        if len(self.loss_history) >= 20:
            first_half = sum(self.loss_history[:10]) / 10
            second_half = sum(self.loss_history[-10:]) / 10
            if second_half >= first_half * 0.99:
                return "NOT_DECREASING"
        if len(recent) >= 5:
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            if max(diffs) - min(diffs) > 2 * abs(sum(diffs) / len(diffs) + 1e-10):
                return "OSCILLATING"
        return "HEALTHY"

    def check_activations(self):
        """检查 dead neurons (死亡神经元)、exploding (爆炸) 或 collapsed (坍塌) activations (激活)。

        返回:
            诊断字符串列表，如果没有问题则为 ["HEALTHY"]。
        """
        issues = []
        for name, stats in self.activation_stats.items():
            if stats["fraction_zero"] > 0.5:
                issues.append(
                    f"DEAD_NEURONS: {name} has {stats['fraction_zero']:.0%} zero activations"
                )
            if abs(stats["mean"]) > 10:
                issues.append(
                    f"EXPLODING_ACTIVATIONS: {name} mean={stats['mean']:.2f}"
                )
            if stats["std"] < 1e-6:
                issues.append(
                    f"COLLAPSED_ACTIVATIONS: {name} std={stats['std']:.2e}"
                )
        return issues if issues else ["HEALTHY"]

    def check_gradients(self):
        """检查 vanishing (消失) 或 exploding (爆炸) gradients (梯度) 以及层间比率。

        返回:
            诊断字符串列表，如果没有问题则为 ["HEALTHY"]。
        """
        issues = []
        grad_magnitudes = []
        for name, stats in self.gradient_stats.items():
            grad_magnitudes.append((name, stats["abs_mean"]))
            if stats["abs_mean"] < 1e-7:
                issues.append(
                    f"VANISHING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}"
                )
            if stats["abs_mean"] > 100:
                issues.append(
                    f"EXPLODING_GRADIENT: {name} abs_mean={stats['abs_mean']:.2e}"
                )
        if len(grad_magnitudes) >= 2:
            first_mag = grad_magnitudes[0][1]
            last_mag = grad_magnitudes[-1][1]
            if last_mag > 0 and first_mag / (last_mag + 1e-15) > 100:
                issues.append(
                    f"GRADIENT_RATIO: first/last = {first_mag / (last_mag + 1e-15):.0f}x (vanishing)"
                )
        return issues if issues else ["HEALTHY"]

    def print_report(self):
        """打印包含 loss health、activation (激活) 和 gradient (梯度) 诊断的完整报告。"""
        print("\n=== NETWORK DEBUGGER REPORT ===")
        print(f"\nLoss health: {self.check_loss_health()}")
        if self.loss_history:
            print(
                f"  Last 5 losses: {[f'{v:.4f}' for v in self.loss_history[-5:]]}"
            )
        print("\nActivation diagnostics:")
        for item in self.check_activations():
            print(f"  {item}")
        print("\nGradient diagnostics:")
        for item in self.check_gradients():
            print(f"  {item}")
        print("\nPer-layer activation stats:")
        for name, stats in self.activation_stats.items():
            print(
                f"  {name}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
                f"zero={stats['fraction_zero']:.1%}"
            )
        print("\nPer-layer gradient stats:")
        for name, stats in self.gradient_stats.items():
            print(
                f"  {name}: abs_mean={stats['abs_mean']:.2e} max={stats['max']:.2e}"
            )

    def remove_hooks(self):
        """移除所有已注册的 hooks 以防止内存泄漏。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def overfit_one_batch(model, x_batch, y_batch, criterion, lr=0.01, steps=200):
    """在单个小 batch 上训练以验证模型 CAN 学习。

    如果 loss 在 {steps} 步后未收敛到 <0.1，则模型或训练循环有 bug。

    参数:
        model: 要测试的 PyTorch 模型。
        x_batch: 输入张量 (batch_size, ...)。
        y_batch: 目标张量。
        criterion: Loss function (损失函数)（例如 nn.CrossEntropyLoss()）。
        lr: 用于 Adam optimizer (优化器) 的 learning rate (学习率)。
        steps: 训练步数。

    返回:
        如果 loss 收敛则返回 True，否则返回 False。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("\n=== OVERFIT ONE BATCH TEST ===")
    print(f"Batch size: {x_batch.shape[0]}, Steps: {steps}")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                if output.shape[-1] == 1:
                    preds = (output > 0).float().squeeze()
                else:
                    preds = output.argmax(dim=1)
                targets = y_batch if y_batch.dim() == 1 else y_batch.squeeze()
                acc = (preds == targets).float().mean().item()
            print(f"  Step {step:3d} | Loss: {loss.item():.6f} | Accuracy: {acc:.1%}")

    final_loss = loss.item()
    if final_loss > 0.1:
        print(
            f"\n  FAIL: Loss did not converge ({final_loss:.4f}). "
            f"Model or training loop is broken."
        )
        return False
    print(f"\n  PASS: Loss converged to {final_loss:.6f}")
    return True


def find_learning_rate(
    model, x_data, y_data, criterion, start_lr=1e-7, end_lr=10, steps=100
):
    """执行 learning rate (学习率) 范围测试（Leslie Smith 方法）。

    从 {start_lr} 到 {end_lr} 指数级增加 LR，记录 loss，并建议一个 LR。
    模型状态在扫描后恢复。

    参数:
        model: 要测试的 PyTorch 模型。
        x_data: 用于测试的输入数据。
        y_data: 用于测试的目标数据。
        criterion: Loss function (损失函数)。
        start_lr: 扫描的起始 learning rate (学习率)（非常小）。
        end_lr: 扫描的结束 learning rate (学习率)（非常大）。
        steps: 扫描的步数。

    返回:
        (lr, loss) 元组列表。
    """
    original_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1 / steps)

    model.train()
    results = []
    best_loss = float("inf")
    current_lr = start_lr

    print("\n=== LEARNING RATE FINDER ===")

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = criterion(output, y_data)

        if math.isnan(loss.item()) or loss.item() > best_loss * 10:
            break

        best_loss = min(best_loss, loss.item())
        results.append((current_lr, loss.item()))

        loss.backward()
        optimizer.step()

        current_lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

    model.load_state_dict(original_state)

    if len(results) < 10:
        print("  Could not complete LR sweep -- loss diverged too quickly")
        return results

    min_loss_idx = min(range(len(results)), key=lambda i: results[i][1])
    suggested_lr = results[max(0, min_loss_idx - 10)][0]

    print(
        f"  Swept {len(results)} steps from {start_lr:.0e} to {results[-1][0]:.0e}"
    )
    print(
        f"  Minimum loss {results[min_loss_idx][1]:.4f} at lr={results[min_loss_idx][0]:.2e}"
    )
    print(f"  Suggested learning rate: {suggested_lr:.2e}")

    return results


def _flat_to_multi_index(flat_idx, shape):
    """将扁平索引转换为多维索引（用于 gradient checking (梯度检查)）。"""
    multi_idx = []
    remaining = flat_idx
    for dim in reversed(shape):
        multi_idx.insert(0, remaining % dim)
        remaining //= dim
    return tuple(multi_idx)


def gradient_check(model, x, y, criterion, eps=1e-4):
    """通过有限差分验证反向传播梯度。

    将 analytical gradients (解析梯度)（来自 .backward()）与 numerical gradients (数值梯度)
    （来自 (loss(w+eps) - loss(w-eps)) / 2eps）进行比较。
    如果 relative difference (相对差异) < 1e-5，则梯度正确。

    参数:
        model: 要检查的 PyTorch 模型。
        x: 输入张量。
        y: 目标张量。
        criterion: Loss function (损失函数)。
        eps: 有限差分的 epsilon（步长）。

    返回:
        所有已检查参数中的最大 relative difference (相对差异)。
    """
    model.train()
    x_double = x.double()
    y_double = y.double()
    model_double = model.double()

    print("\n=== GRADIENT CHECK ===")
    overall_max_diff = 0
    checked = 0

    for name, param in model_double.named_parameters():
        if not param.requires_grad:
            continue

        layer_max_diff = 0

        model_double.zero_grad()
        output = model_double(x_double)
        loss = criterion(output, y_double)
        loss.backward()
        analytical_grad = param.grad.clone()

        num_checks = min(5, param.numel())
        for i in range(num_checks):
            idx = _flat_to_multi_index(i, param.shape)
            original = param.data[idx].item()

            param.data[idx] = original + eps
            with torch.no_grad():
                loss_plus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original - eps
            with torch.no_grad():
                loss_minus = criterion(model_double(x_double), y_double).item()

            param.data[idx] = original

            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytical = analytical_grad[idx].item()

            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_diff = abs(numerical - analytical) / denom

            layer_max_diff = max(layer_max_diff, rel_diff)
            checked += 1

        overall_max_diff = max(overall_max_diff, layer_max_diff)
        status = "OK" if layer_max_diff < 1e-5 else "MISMATCH"
        print(f"  {name}: max_rel_diff={layer_max_diff:.2e} [{status}]")

    model.float()

    print(f"\n  Checked {checked} parameters")
    if overall_max_diff < 1e-5:
        print("  PASS: Gradients match (rel_diff < 1e-5)")
    elif overall_max_diff < 1e-3:
        print("  WARN: Small differences (1e-5 < rel_diff < 1e-3)")
    else:
        print("  FAIL: Gradient mismatch detected (rel_diff > 1e-3)")
    return overall_max_diff


def demo_broken_networks():
    """演示常见 bug 的诊断：learning rate (学习率) 过高、dead ReLU (死亡ReLU)、
    缺少 zero_grad，以及 healthy network (健康网络) 基线。"""
    torch.manual_seed(42)
    x = torch.randn(64, 10)
    y = (x[:, 0] > 0).long()
    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print("BUG 1: Learning rate too high (lr=10)")
    print("=" * 60)
    model1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger1 = NetworkDebugger(model1)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=10.0)
    for step in range(20):
        optimizer1.zero_grad()
        out = model1(x)
        loss = criterion(out, y)
        debugger1.record_loss(loss.item())
        loss.backward()
        optimizer1.step()
    debugger1.print_report()
    debugger1.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 2: Dead ReLUs from bad initialization")
    print("=" * 60)
    model2 = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )
    with torch.no_grad():
        for m in model2.modules():
            if isinstance(m, nn.Linear):
                m.weight.fill_(-1.0)
                m.bias.fill_(-5.0)
    debugger2 = NetworkDebugger(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for step in range(50):
        optimizer2.zero_grad()
        out = model2(x)
        loss = criterion(out, y)
        debugger2.record_loss(loss.item())
        loss.backward()
        optimizer2.step()
    debugger2.print_report()
    debugger2.remove_hooks()

    print("\n" + "=" * 60)
    print("BUG 3: Missing zero_grad (gradients accumulate)")
    print("=" * 60)
    model3 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger3 = NetworkDebugger(model3)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.01)
    for step in range(50):
        out = model3(x)
        loss = criterion(out, y)
        debugger3.record_loss(loss.item())
        loss.backward()
        optimizer3.step()
    debugger3.print_report()
    debugger3.remove_hooks()

    print("\n" + "=" * 60)
    print("HEALTHY NETWORK: Correct setup for comparison")
    print("=" * 60)
    model_good = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    debugger_good = NetworkDebugger(model_good)
    optimizer_good = torch.optim.Adam(model_good.parameters(), lr=1e-3)
    for step in range(50):
        optimizer_good.zero_grad()
        out = model_good(x)
        loss = criterion(out, y)
        debugger_good.record_loss(loss.item())
        loss.backward()
        optimizer_good.step()
    debugger_good.print_report()
    debugger_good.remove_hooks()

    print("\n" + "=" * 60)
    print("OVERFIT-ONE-BATCH TEST")
    print("=" * 60)
    model_test = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    overfit_one_batch(model_test, x[:8], y[:8], criterion)

    print("\n" + "=" * 60)
    print("LEARNING RATE FINDER")
    print("=" * 60)
    model_lr = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    find_learning_rate(model_lr, x, y, criterion)

    print("\n" + "=" * 60)
    print("GRADIENT CHECK (smooth model + MSE loss for clean finite differences)")
    print("=" * 60)
    torch.manual_seed(123)
    x_check = torch.randn(4, 3)
    y_check = torch.randn(4, 1)
    model_grad = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    gradient_check(model_grad, x_check, y_check, nn.MSELoss())


if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING NEURAL NETWORKS -- Phase 3, Lesson 13")
    print("=" * 60)
    demo_broken_networks()
