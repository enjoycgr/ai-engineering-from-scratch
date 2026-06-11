# 自编码器与变分自编码器 (Autoencoders & Variational Autoencoders, VAE)

> 普通自编码器 (autoencoder) 先压缩再重建。它只是记忆数据，并不生成。加入一个技巧——强迫编码看起来像高斯分布 (Gaussian)——你就得到了一个采样器 (sampler)。这唯一的技巧，`z = μ + σ·ε` 的重参数化 (reparameterization)，正是 2026 年你使用的每一个 latent-diffusion 和 flow-matching 图像模型在输入端都配备 VAE 的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 分钟

## 问题

将 784 像素的 MNIST 数字压缩为 16 维编码，然后重建。普通自编码器 (autoencoder) 在重建 MSE 上表现完美，但编码空间是一团乱麻。在编码空间中随机取一个点进行解码，得到的是噪声。它没有采样能力，只是一个伪装成生成模型的压缩模型。

你真正想要的是：(a) 编码空间是一个干净、平滑、可以从中采样的分布——比如各向同性高斯分布 `N(0, I)`；(b) 解码任意样本都能产生合理的数字；(c) 编码器和解码器仍然能很好地压缩。三个目标，一种架构，一个损失函数。

Kingma 在 2013 年提出的 VAE 通过训练编码器输出一个分布 `q(z|x) = N(μ(x), σ(x)²)` 来解决这个问题，并通过 KL 惩罚项将该分布拉向先验 (prior) `N(0, I)`，然后在解码前从 `q(z|x)` 中采样 `z`。在推理 (inference) 阶段，丢弃编码器，直接采样 `z ~ N(0, I)` 进行解码。KL 惩罚项正是强制编码空间具有结构的关键。

在 2026 年，VAE 很少单独发布——它们在原始图像质量上已被扩散模型 (diffusion) 超越——但它们是每一个 latent-diffusion 模型（SD 1/2/XL/3、Flux、AudioCraft）首选的编码器。学好 VAE，你就学懂了每一个图像 pipeline 中看不见的第一层。

## 概念

![Autoencoder vs VAE: 重参数化技巧 (reparameterization trick)](../assets/vae.svg)

**自编码器 (Autoencoder)。** `z = encoder(x)`，`x̂ = decoder(z)`，loss = `||x - x̂||²`。编码空间无结构。

**VAE 编码器。** 输出两个向量：`μ(x)` 和 `log σ²(x)`。它们定义了 `q(z|x) = N(μ, diag(σ²))`。

**重参数化技巧 (Reparameterization trick)。** 从 `q(z|x)` 中采样不可微。将采样重写为 `z = μ + σ·ε`，其中 `ε ~ N(0, I)`。现在 `z` 是 `(μ, σ)` 的确定性函数加上纯噪声——梯度可以流过 `μ` 和 `σ`。

**损失函数 (Loss)。** 证据下界 (ELBO, Evidence Lower BOund)，两项：

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

重建 (reconstruction) 项推动 `x̂` 靠近 `x`。KL 项推动 `q(z|x)` 靠近先验。二者此消彼长。小的 β (<1) = 更锐利的样本，编码空间更少高斯性。大的 β (>1) = 更干净的编码空间，但更模糊的样本。β-VAE（Higgins 2017）让这个旋钮出了名，并开启了解耦 (disentanglement) 研究。

**采样 (Sampling)。** 在推理阶段：采样 `z ~ N(0, I)`，前向通过解码器。一次前向传播——不像扩散模型那样需要迭代采样。

## 动手构建

`code/main.py` 实现了一个不使用 numpy 或 torch 的微型 VAE。输入是从 8 维高斯混合分布中抽取的合成数据。编码器和解码器都是单隐藏层 MLP。我们手动实现了 tanh 激活函数、前向传播、损失函数和手写的反向传播 (backpropagation)。这不是生产代码——是教学代码。

### 步骤 1：编码器前向传播

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

使用 `log σ²` 而不是 `σ`，这样网络输出无约束（σ 的 softplus 是个陷阱——在 σ ≈ 0 时梯度会消失）。

### 步骤 2：重参数化与解码

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步骤 3：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因为两个分布都是高斯分布，KL 有闭式解。不要数值积分。2026 年仍然有人发布用蒙特卡洛估计 KL 的代码——没理由地慢 3 倍。

### 步骤 4：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

这就是生成模型。五行代码。

## 常见陷阱

- **后验坍塌 (Posterior collapse)。** KL 项过于激进地将 `q(z|x)` 推向 `N(0, I)`，导致 `z` 不再携带关于 `x` 的任何信息。修复方法：β-退火（初始 β=0，逐渐升至 1）、free bits，或在 inactive 维度上跳过 KL。
- **模糊样本。** 高斯解码器似然隐含了 MSE 重建，而 MSE 在 L2 意义下是贝叶斯最优的（即均值）——一堆合理数字的均值是一个模糊的数字。修复方法：离散解码器（VQ-VAE、NVAE），或将 VAE 仅用作编码器并在隐变量 (latent) 上堆叠扩散模型（这就是 Stable Diffusion 的做法）。
- **β 过大、过早。** 参见后验坍塌。从 β≈0.01 开始并逐渐提升。
- **隐变量维度 (latent dim) 过小。** MNIST 用 16 维，ImageNet 256² 用 256 维，ImageNet 1024² 用 2048 维。Stable Diffusion 的 VAE 将 512×512×3 压缩为 64×64×4（空间面积下采样 32 倍，通道 32 倍）。

## 实际应用

2026 年的 VAE 技术栈：

| 场景 | 选择 |
|-----------|------|
| 用于扩散模型的图像隐变量编码器 | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| 音频隐变量编码器 | Encodec (Meta)、SoundStream 或 DAC (Descript) |
| 视频隐变量 | Sora 的时空 patch、Latte VAE、WAN VAE |
| 解耦表示学习 | β-VAE、FactorVAE、TCVAE |
| 离散隐变量（用于 transformer 建模）| VQ-VAE、RVQ (ResidualVQ) |
| 用于生成的连续隐变量 | 普通 VAE，然后在其隐空间中条件化一个 flow/diffusion 模型 |

Latent-diffusion 模型就是 VAE + 住在编码器和解码器之间的扩散模型。VAE 做粗略压缩，扩散模型做重活。视频（VAE + 视频扩散 DiT）和音频（Encodec + MusicGen transformer）遵循同样的模式。

## 交付它

保存 `outputs/skill-vae-trainer.md`。

该 skill 接收：数据集画像 + 目标隐变量维度 + 下游用途（重建、采样，或 latent-diffusion 输入），并输出：架构选择（plain/β/VQ/RVQ）、β 调度、隐变量维度、解码器似然（高斯 vs 分类），以及评估计划（重建 MSE、每维 KL、Fréchet 距离）。

## 练习

1. **简单。** 在 `code/main.py` 中将 `β` 改为 `0.01`、`0.1`、`1.0`、`5.0`。记录最终重建 MSE 和 KL。对于你的合成数据，哪个 β 是帕累托最优的？
2. **中等。** 将高斯解码器似然替换为伯努利似然（交叉熵损失）。在相同合成数据的二值化版本上比较样本质量。
3. **困难。** 将 `code/main.py` 扩展为迷你 VQ-VAE：用 K=32 条目的码本中的最近邻查找替换连续 `z`。比较重建 MSE 并报告有多少码本条目被使用（码本坍塌 (codebook collapse) 是真实存在的）。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Autoencoder | 编码-解码网络 | `x → z → x̂`，学习 MSE。不是生成模型。 |
| VAE | 带采样器的自编码器 | 编码器输出分布，KL 惩罚塑造编码空间。 |
| ELBO | 证据下界 | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`；当 `q = p(z\|x)` 时紧致。 |
| Reparameterization | `z = μ + σ·ε` | 将随机节点重写为确定性 + 纯噪声。使反向传播 (backpropagation) 能穿过采样。 |
| Prior | `p(z)` | 隐变量的目标分布，通常为 `N(0, I)`。 |
| Posterior collapse | "KL 项赢了" | 编码器忽略 `x`，输出先验；解码器只能 hallucinate。 |
| β-VAE | 可调 KL 权重 | `loss = recon + β·KL`。更高的 β = 更解耦但更模糊。 |
| VQ-VAE | 离散隐变量 | 用码本中最邻近的向量替换连续 `z`；使 transformer 建模成为可能。 |

## 生产备忘：VAE 是扩散服务器中最热的路径

在 Stable Diffusion / Flux / SD3 的 pipeline 中，VAE 每个请求被调用两次——一次编码（如果做 img2img / inpainting）和一次解码。在 1024² 分辨率下，解码器前向传播往往是整个 pipeline 中最大的激活内存峰值，因为它将 `128×128×16` 的隐变量上采样回 `1024×1024×3`。两个实际后果：

- **切片或分块解码。** `diffusers` 提供了 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。分块用微小的接缝伪影换取 `O(tile²)` 内存而不是 `O(H·W)`。在消费级 GPU 上处理 1024²+ 是必需的。
- **bf16 解码器，fp32 数值用于最终 resize。** SD 1.x 的 VAE 以 fp32 发布，*在 1024²+ 转为 fp16 时会静默产生 NaN*。SDXL 提供了 `madebyollin/sdxl-vae-fp16-fix`——始终优先使用 fp16-fix 变体，或使用 bf16。

## 延伸阅读

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) —— VAE 论文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) —— 解耦 β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) —— VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) —— 最先进的图像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) —— Stable Diffusion；VAE 作为编码器。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) —— Encodec，音频 VAE 标准。
