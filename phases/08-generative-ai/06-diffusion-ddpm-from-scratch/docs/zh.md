# 扩散模型 —— 从零实现 DDPM

> Ho、Jain、Abbeel (2020) 给整个领域提供了一份让人无法拒绝的配方。在数千个小步骤中用噪声摧毁数据，训练一个神经网络来预测噪声，然后在推理时逆转这个过程。如今，所有主流的图像、视频、3D 和音乐模型都运行在这个循环之上，可能还叠加了 flow matching（流匹配）或一致性模型等技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 分钟

## The Problem

你需要一个用于 `p_data(x)` 的采样器（sampler）。GAN（生成对抗网络）玩的是 minimax 博弈，经常发散。VAE（变分自编码器）从高斯解码器（Gaussian decoder）中生成模糊的样本。你真正想要的是一个满足以下条件的训练目标：(a) 单一的稳定损失（没有鞍点，没有 minimax），(b) 是 `log p(x)` 的下界（因此你能得到似然值），以及 (c) 样本质量达到 SOTA。

Sohl-Dickstein 等人 (2015) 从理论上给出了答案：定义一个马尔可夫链（Markov chain）`q(x_t | x_{t-1})`，逐步添加高斯噪声（Gaussian noise），然后训练一个反向链 `p_θ(x_{t-1} | x_t)` 来进行去噪（denoising）。Ho、Jain、Abbeel (2020) 证明了损失可以被简化到一行代码 —— 预测噪声 —— 并清理了数学推导。2020 年这还是一项好奇之作，2021 年它产出了 SOTA 样本，2022 年它变成了 Stable Diffusion，到 2026 年它已是底层基础设施。

## The Concept

![DDPM: 前向加噪，反向去噪](../assets/ddpm.svg)

**前向过程（Forward process）`q`。** 在 `T` 个小步骤中添加高斯噪声。数学上可处理的关键在于，其累积步骤也具有高斯闭式解：

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

其中 `α̅_t = ∏_{s=1..t} (1 - β_s)`，`β_t` 是一组噪声调度（noise schedule）。让 `β_t` 从 1e-4 到 0.02 在 T=1000 个时间步（timestep）上线性增长，`x_T` 就近似于 `N(0, I)`。

**反向过程（Reverse process）`p_θ`。** 学习一个神经网络 `ε_θ(x_t, t)` 来预测被添加的噪声。给定 `x_t`，通过以下方式去噪（denoising）：

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

其中 `σ_t` 可以是 `sqrt(β_t)` 或一个学习得到的方差（variance）。这个表达式看起来复杂，但只是代数运算 —— 根据后验分布 `q(x_{t-1} | x_t, x_0)` 求解 `x_{t-1}`，并用噪声预测估计替换 `x_0`。

**训练损失（Training loss）。**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

从数据中采样 `x_0`，随机选取 `t`，采样 `ε ~ N(0, I)`，通过闭式形式一步计算出有噪声的 `x_t`，然后对噪声做回归。一个损失函数，没有 minimax，没有 KL，没有重参数化技巧。

**采样（Sampling）。** 从 `x_T ~ N(0, I)` 开始，从 `t = T` 到 `1` 迭代反向步骤。完成。

## Why it works

三个直觉：

1. **去噪（Denoising）容易，生成难。** 在 `t=T` 时，数据是纯噪声 —— 网络只需要解决一个简单问题。在 `t=0` 时，网络只需要清理几个像素。在中间 `t`，问题很难，但网络通过相同的权重从每个噪声级别获得大量梯度。

2. **伪装成噪声预测的分数匹配（Score matching）。** Vincent (2011) 证明了预测噪声等价于估计 `∇_x log q(x_t | x_0)`，即 *score*。反向 SDE 利用这个 score 沿着密度梯度向上走 —— 一个朝向高密度区域的引导随机游走。

3. **ELBO（证据下界）简化为简单 MSE（均方误差）。** 完整的变分下界在每个时间步都有一个 KL 项。在 DDPM 的参数化下，这些 KL 项简化为对噪声预测的 MSE 并带有特定系数；Ho 去掉了这些系数（称之为 "simple" 损失），结果质量反而*提升了*。

## Build It

`code/main.py` 实现了一个一维 DDPM。数据是一个双峰混合分布。"网络"是一个微型 MLP，接收 `(x_t, t)` 并输出预测的噪声。训练就是那一行损失。采样迭代反向链。

### Step 1: 前向调度（闭式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Step 2: 一步采样 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Step 3: 一步训练

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Step 4: 反向采样

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

对于一个 40 个时间步、24 单元 MLP 的一维问题，这大约在 200 个轮次（epoch）内学会了双峰混合分布。

## Time conditioning

网络需要知道它正在对哪个时间步（timestep）进行去噪。两个标准选项：

- **正弦嵌入（Sinusoidal embedding）。** 类似于 Transformer 的位置编码（positional encoding）。`embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`。通过一个 MLP，广播到网络中。
- **FiLM / 组归一化条件化（group-norm conditioning）。** 将嵌入投影到每个通道的缩放/偏置（scale/bias）上。

我们的玩具代码使用正弦 → 拼接。生产级 U-Net 使用 FiLM。

## Pitfalls

- **调度（Schedule）非常重要。** 线性 `β` 是 DDPM 的默认值，但余弦调度（cosine schedule，Nichol & Dhariwal, 2021）在相同计算量下给出更好的 FID。如果质量遇到瓶颈，就换调度。
- **时间步嵌入（Timestep embedding）很脆弱。** 把原始 `t` 作为浮点数传入对一维玩具有效，但对图像会失败；始终使用恰当的嵌入。
- **V-prediction 与 ε-prediction。** 在极端范围（`t` 非常小或非常大），`ε` 的信噪比很差。V-prediction（`v = α·ε - σ·x`）更稳定；SDXL、SD3 和 Flux 都使用它。
- **Classifier-free guidance（无分类器引导）。** 在推理时，同时计算条件和非条件的 `ε`，然后 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`，其中 `w ≈ 3-7`。在第 08 课中介绍。
- **1000 步太多了。** 生产环境使用 DDIM（20-50 步）、DPM-Solver（10-20 步）或蒸馏（1-4 步）。参见第 12 课。

## Use It

| 用途 | 2026 年的典型技术栈 |
|------|---------------------|
| 图像像素空间扩散（小规模、玩具） | DDPM + U-Net |
| 图像隐空间扩散（latent diffusion） | VAE 编码器 + U-Net 或 DiT（第 07 课） |
| 视频隐空间扩散 | 时空 DiT（Sora、Veo、WAN） |
| 音频隐空间扩散 | EnCodec + diffusion transformer |
| 科学（分子、蛋白质、物理） | 等变扩散（EDM、RFdiffusion、AlphaFold3） |

扩散（Diffusion）是通用的生成式骨干网络。Flow matching（第 13 课）是 2024-2026 年的竞争者，通常在相同质量下推理速度更快。

## Ship It

保存 `outputs/skill-diffusion-trainer.md`。该 skill 接收一个数据集 + 计算预算，输出：调度（线性/余弦/ sigmoid）、预测目标（ε/v/x）、步数、引导尺度、采样器家族，以及一个评估协议。

## Exercises

1. **简单。** 在 `code/main.py` 中将 T 从 40 改为 10。样本质量（输出的可视化直方图）如何下降？在哪个 T 下双峰结构会崩溃？
2. **中等。** 从 ε-prediction 切换到 v-prediction。重新推导反向步骤。比较最终样本质量。
3. **困难。** 添加 classifier-free guidance。在类别标签 `c ∈ {0, 1}` 上条件化，训练时 10% 的时间丢弃它，采样时使用 `ε = (1+w)·ε_cond - w·ε_uncond`。测量 `w = 0, 1, 3, 7` 下的条件模式命中率。

## Key Terms

| 术语 | 人们的说法 | 实际含义 |
|------|----------|---------|
| Forward process | "加噪" | 固定的马尔可夫链（Markov chain）`q(x_t \| x_{t-1})`，逐步摧毁数据。 |
| Reverse process | "去噪" | 学习得到的链 `p_θ(x_{t-1} \| x_t)`，重建数据。 |
| β schedule | "噪声阶梯" | 每步方差（variance）；线性、余弦或 sigmoid。 |
| α̅ | "Alpha bar" | 累积乘积 `∏(1 - β)`；给出从 `x_0` 到 `x_t` 的闭式解。 |
| Simple loss | "噪声上的 MSE" | `\|\|ε - ε_θ(x_t, t)\|\|²`；所有变分推导都坍缩为这个。 |
| ε-prediction | "预测噪声" | 输出是添加的噪声；标准 DDPM。 |
| V-prediction | "预测速度" | 输出是 `α·ε - σ·x`；在 `t` 上提供更好的条件化。 |
| DDPM | "那篇论文" | Ho 等人 2020；线性 β，1000 步，U-Net。 |
| DDIM | "确定性采样器" | 非马尔可夫采样器，20-50 步，相同的训练目标。 |
| Classifier-free guidance | "CFG" | 混合条件和非条件的噪声预测来放大条件控制。 |

## Production note: diffusion inference is a step-count problem

DDPM 论文运行 T=1000 个反向步骤。没有人在生产环境中部署这个。每个真实推理栈选择以下三种策略之一 —— 每种都清晰地映射到生产文献中 "延迟来自哪里" 的框架：

1. **更快的采样器，相同的模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。反向循环的即插即用替换；训练好的 `ε_θ` 权重完全不动。将延迟降低 20-50 倍。
2. **蒸馏。** 训练一个学生模型在更少的步骤中匹配教师模型：渐进蒸馏（2 → 1）、一致性模型（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。再将延迟降低 5-10 倍，但需要重新训练。
3. **缓存和编译。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的扩散后端、`xformers`/SDPA attention、bf16 权重。将每步延迟降低约 2 倍。可以与 (1) 和 (2) 叠加。

对于一个生产级扩散服务器，预算讨论与生产文献中对 LLM 的描述相同：延迟 = `num_steps × step_cost + VAE_decode`，吞吐量 = `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；TPOT 等价物是完整响应时间，因为从用户角度看图像生成是"一次性"完成的。

## Further Reading

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) —— 扩散论文，超前于时代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) —— DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) —— DDIM，更少的步骤。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) —— 余弦调度，学习方差。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) —— 分类器引导。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) —— 统一符号，最干净的配方。
