# Flow Matching (流匹配) & Rectified Flows (整流流)

> Diffusion models (扩散模型) 需要 20–50 个采样步骤，因为它们沿着一条弯曲的路径从噪声走到数据。Flow matching (流匹配, Lipman et al., 2023) 和 rectified flow (整流流, Liu et al., 2022) 训练出直线路径。更直的路径意味着更少的步骤，意味着更快的 inference (推理)。Stable Diffusion 3、Flux.1 和 AudioCraft 2 都在 2024 年切换到了 flow matching。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 分钟

## The Problem

DDPM 的反向过程是从 $N(0, I)$ 回到数据分布的 1000 步随机 (stochastic) 行走。DDIM 将其压缩为 20–50 步确定性 (deterministic) 步骤。你想进一步减少步骤—— ideally 一步。阻碍在于求解反向过程的 ODE (常微分方程) 是 stiff 的；路径是弯曲的。

如果你能训练模型使得从噪声到数据的路径是一条*直线*，那么从 $t=1$ 到 $t=0$ 的单步 Euler 就能奏效。Flow matching 直接构建这一点：定义从 $x_1 \sim N(0, I)$ 到 $x_0 \sim \text{data}$ 的直线路径插值，训练 vector field (速度场) $v_\theta(x, t)$ 以匹配其时间导数，在 inference 时进行积分。

Rectified flow (Liu 2022) 更进一步：通过 reflow (重流) 过程迭代地拉直路径，产生 progressively closer-to-linear 的 ODE。经过两次 reflow 迭代后，2-step sampler 即可匹配 50-step DDPM 的质量。

## The Concept

![Flow matching：噪声与数据之间的直线路径插值](../assets/flow-matching.svg)

### Straight-line flow

定义：

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

其中 $x_0 \sim \text{data}$ 且 $x_1 \sim N(0, I)$。沿这条直线的时间导数是恒定的：

```
dx_t / dt = x_1 - x_0
```

定义神经 vector field (速度场) $v_\theta(x_t, t)$ 并训练它匹配该导数：

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

这就是 **conditional flow matching (条件流匹配)** loss (Lipman 2023)。训练是 simulation-free 的：你从不展开 ODE。只需采样 $(x_0, x_1, t)$ 并进行回归。

### Sampling (采样)

在 inference (推理) 时，沿着学习到的 velocity field (速度场) 反向积分：

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

从 $x_1 \sim N(0, I)$ 开始，Euler-step 下降到 $t=0$。

### Rectified flow (Liu 2022)

Straight-line flow 有效，但学习到的路径*实际上并不直*——因为许多 $x_0$ 可能映射到同一个 $x_1$。Rectified flow 的 reflow (重流) 步骤：

1. 用随机配对训练 flow model $v_1$。
2. 通过从 $x_1$ 积分 $v_1$ 到其着陆点 $x_0$，采样 $N$ 对 $(x_1, x_0)$。
3. 在这些配对样本上训练 $v_2$。因为这些配对现在 "ODE-matched"，它们之间的直线路径插值 genuinely flatter。
4. 重复。

在实践中，2 次 reflow 迭代即可达到 near-linear，从而实现 2–4 步 inference。SDXL-Turbo、SD3-Turbo、LCM 都是从 flow-matching 模型蒸馏而来的。

### Why this won for images in 2024

三个原因：

1. **Simulation-free training** —— 训练期间无需展开 ODE，实现起来非常简单。
2. **Better loss geometry** —— 直线路径具有 consistent signal-to-noise，而 DDPM 的 ε-loss 在调度边缘处 SNR (信噪比) 很差。
3. **Faster inference** —— 4–8 步即可达到 SDXL-Turbo 质量；通过 consistency distillation (一致性蒸馏) 可进一步到 1 步。

## Flow matching vs DDPM — the exact connection

Flow matching with a Gaussian-conditional path is diffusion *with a specific noise schedule*。选取 $x_t = \alpha(t) x_0 + \sigma(t) x_1$ 的调度，flow matching 即可恢复 Stratonovich-reformulated diffusion，其中 $v = \alpha' \cdot x_0 - \sigma' \cdot x_1$。对于高斯路径，两者在代数上是等价的。

Flow matching 新增的是：目标的*清晰度*（一个 plain velocity）、更干净的 loss，以及实验非高斯插值的自由。

## Build It

`code/main.py` 在一维双模态高斯混合上实现了 flow matching。Vector field (速度场) $v_\theta(x, t)$ 是一个 tiny MLP，使用 straight-line target 进行训练。在 inference 时，分别用 1、2、4、20 步 Euler 积分并比较 sample (采样) 质量。

### Step 1: training loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Step 2: multi-step inference

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Step 3: compare step counts

预期 4-step sampler 已经匹配 20-step 质量——这对延迟来说意义重大。

## Pitfalls

- **Time parameterization.** Flow matching 使用 $t \in [0, 1]$，$t=0$ 在数据端，$t=1$ 在噪声端。DDPM 使用 $t \in [0, T]$，$t=0$ 在数据端，$t=T$ 在噪声端。方向相同，尺度不同。论文 constantly 搞错这一点。
- **Schedule choice.** Rectified flow 的直线是 flow matching 的 "the" 调度，但你可以使用 cosine (余弦) 或 logit-normal t (logit-正态 t 采样, SD3 使用这个) 以获得更好的 scale coverage。
- **Reflow cost.** 生成 reflow 的配对数据集需要对每个样本进行一次完整的 inference pass。只有当你真正需要 1–2 步 inference 时才进行 reflow。
- **Classifier-free guidance (无分类器引导, CFG) 仍然适用。** 只需将 ε 替换为 v 进行线性组合：$v_{cfg} = (1+w) v_{cond} - w v_{uncond}$。

## Use It

| Use case | 2026 stack |
|----------|-----------|
| Text-to-image, best quality | Flow matching: SD3, Flux.1-dev |
| Text-to-image, 1-4 steps | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Real-time inference | Consistency distillation (一致性蒸馏) from a flow-matched base (LCM, PCM) |
| Audio generation | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Video generation | Flow matching mixed with diffusion (Sora, Veo, Stable Video) |
| Science / physics (particle trajectories, molecules) | Flow matching + equivariant vector field |

2025–2026 年任何论文说 "faster than diffusion"，几乎总是 flow matching + distillation。

## Ship It

保存 `outputs/skill-fm-tuner.md`。该 skill 接收一份 diffusion-style model spec 并将其转换为 flow-matching 训练配置：schedule choice、time sampling distribution（uniform / logit-normal）、optimizer、reflow plan、target step count、eval protocol。

## Exercises

1. **Easy.** 运行 `code/main.py` 并比较 1-step 与 20-step MSE (均方误差) 与真实数据分布的差距。
2. **Medium.** 将 uniform $t$ sampling 切换到 logit-normal（将采样集中在 mid-t）。模型质量是否提升？
3. **Hard.** 实现一次 reflow (重流) 迭代：通过积分第一个模型生成配对 $(x_0, x_1)$，在这些配对上训练第二个模型，并比较 1-step sample (采样) 质量。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Flow matching | "Straight-line diffusion" | 训练 $v_\theta(x, t)$ 以匹配插值上的 $x_1 - x_0$。 |
| Rectified flow | "Reflow" | 拉直学习流的迭代过程。 |
| Velocity field | "$v_\theta$" | 模型输出——移动 $x_t$ 的方向。 |
| Straight-line interpolant | "The path" | $x_t = (1-t)\cdot x_0 + t\cdot x_1$；平凡的 target derivative。 |
| Euler sampler | "1st order ODE solver" | 最简单的积分器；当路径较直时效果很好。 |
| Logit-normal t | "SD3 sampling" | 将 $t$ 采样集中在 mid-values，此处梯度最强。 |
| Consistency distillation | "1-step sampler" | 训练一个学生模型，将任何 $x_t$ 直接映射到 $x_0$。 |
| CFG with velocity | "v-CFG" | $v_{cfg} = (1+w) v_{cond} - w v_{uncond}$；同样的 trick，新的变量。 |

## Production note: Flux.1-schnell is flow matching at its fastest

Flow matching 的生产级胜利是 Flux.1-schnell —— 一个 flow-matched DiT，蒸馏到 1–4 步 inference，同时保持 Flux-dev 级别的质量。Niels 的 "Run Flux on an 8GB machine" notebook 是部署参考方案：T5 + CLIP encode，quantized MMDiT denoise（schnell 用 4 步 vs dev 用 50 步），VAE decode。成本核算：

| Variant | Steps | Latency at 1024² on L4 | Total FLOPs (relative) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

生产规则：**flow-matched base + distillation = 2026 年快速文本到图像的默认方案。** 每个主要厂商都提供这种组合：SD3-Turbo（SD3 + flow + distillation）、Flux-schnell（Flux-dev + rectified-flow straightening）、CogView-4-Flash。纯 diffusion base 仅存在于 legacy checkpoints。

## Further Reading

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flow at scale。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 覆盖 FM + diffusion 的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — diffusion / flow 的 1-step 蒸馏。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo variant。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 生产中的 flow matching。
