# GAN —— 生成器与判别器 (Generator vs Discriminator)

> Goodfellow 在 2014 年的技巧是完全跳过密度估计。两个网络。一个造假。一个抓假。它们互相对抗，直到假样本与真实样本无法区分。这不应该能工作。它经常不能。但当它工作时，样本在特定领域仍然是最锐利的。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 分钟

## 问题

VAE 产生模糊样本，因为它们的 MSE 解码器损失在贝叶斯意义上对*均值*图像是最优的——而许多合理数字的均值是一个模糊的数字。你想要一种奖励*合理性*而非像素级接近某个单一目标的损失。合理性没有闭式解，你必须学习它。

Goodfellow 的想法：训练一个分类器 `D(x)` 来区分真实图像和伪造图像。训练一个生成器 `G(z)` 来欺骗 `D`。`G` 的损失信号就是 `D` 当前认为让某物看起来真实的东西。这个信号随着 `G` 的改进而更新，追逐一个移动的目标。如果两个网络都收敛，`G` 就学会了数据分布，而从未写下 `log p(x)`。

这就是对抗训练 (adversarial training)。数学上是一个极小极大博弈 (minimax game)：

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

在 2026 年，GAN 不再是 SOTA 生成器（扩散 (diffusion) 和流匹配 (flow matching) 夺走了这顶皇冠）。但 StyleGAN 2/3 仍然是发布过的最锐利的人脸模型，GAN 判别器 (discriminator) 被用作扩散训练中的*感知损失 (perceptual loss)*，而对抗训练驱动了快速的 1 步蒸馏（SDXL-Turbo、SD3-Turbo、LCM），让你能发布实时扩散。

## 概念

![GAN 训练：生成器与判别器的极小极大博弈](../assets/gan.svg)

**生成器 `G(z)`。** 将噪声向量 `z ~ N(0, I)` 映射到样本 `x̂`。一个解码器形状的网络（全连接或转置卷积）。

**判别器 `D(x)`。** 将样本映射到标量概率（或分数）。真实 → 1，伪造 → 0。

**损失函数 (Loss)。** 两个交替更新：

- **训练 `D`：** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`。真实=1、伪造=0 的二元交叉熵 (binary cross-entropy)。
- **训练 `G`：** `loss_G = -log D(G(z))`。这是 Goodfellow 使用的*非饱和 (non-saturating)*形式（原始 `log(1 - D(G(z)))` 在 `D` 有信心时会饱和并杀死梯度）。

**训练循环。** 一步 `D`，一步 `G`。重复。

**为什么有效。** 如果 `G` 完美匹配 `p_data`，那么 `D` 无法做得比随机猜测更好，到处输出 0.5；`G` 不再获得梯度。均衡。

**为什么失效。** 模式崩溃 (mode collapse)（`G` 找到一个 `D` 无法分类的模式并永远铸造它）、梯度消失 (vanishing gradient)（`D` 学得太快，`log D` 饱和）、训练不稳定（学习率、批量大小，任何东西）。

## 让 GAN 真正工作的变体

| 年份 | 创新 | 修复 |
|------|------------|-----|
| 2015 | DCGAN | 卷积/反卷积、批量归一化 (batch norm)、LeakyReLU —— 第一个稳定架构。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein 距离 + 梯度惩罚替换 BCE。修复梯度消失。 |
| 2017 | Spectral normalization | 对判别器 (discriminator) 做 Lipschitz 约束。2026 年仍在判别器中使用。 |
| 2018 | Progressive GAN | 先训练低分辨率，再逐步增加层。第一个百万像素结果。 |
| 2019 | StyleGAN / StyleGAN2 | 映射网络 + 自适应实例归一化。固定域照片级真实感的最高水平。 |
| 2021 | StyleGAN3 | 无混叠、平移等变——2026 年人脸金标准。 |
| 2022 | StyleGAN-XL | 条件化、类别感知、更大规模。 |
| 2024 | R3GAN | 用更强的正则化 (regularization) 重新品牌；无需技巧即可在 1024² 上工作。 |

## 动手构建

`code/main.py` 在 1 维数据上训练一个微型 GAN：两个高斯分布的混合。生成器 (generator) 和判别器 (discriminator) 都是单隐藏层 MLP。我们手动实现前向传播、反向传播 (backpropagation) 和极小极大循环。目标是亲眼看到两种关键失效模式（模式崩溃 + 梯度消失）的发生。

### 步骤 1：非饱和损失

原始 Goodfellow 损失 `log(1 - D(G(z)))` 当 D 将 G 的伪造 confidently 分类为伪造时趋于 0。此时 G 的梯度基本上为零——G 无法改进。非饱和形式 `-log D(G(z))` 有相反的渐近线：当 D 有信心时它会爆炸，给 G 一个强烈的信号。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 步骤 2：每步生成器对应一步判别器

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

G 使用新鲜的伪造样本，否则梯度是陈旧的。

### 步骤 3：观察模式崩溃

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症状：两个真实模式之一停止被生成。判别器停止纠正它，因为它从未被当作伪造样本看到。

## 常见陷阱

- **判别器 (discriminator) 太强。** 将 D 的学习率降低 2-5 倍，或添加实例/层噪声。如果 D 达到 >95% 准确率，G 就死了。
- **生成器 (generator) 记忆了一个模式。** 向 D 输入添加噪声，使用 minibatch-discrimination 层，或切换到 WGAN-GP。
- **批量归一化 (batch norm) 泄露统计量。** 真实批量 + 伪造批量流经同一个 BN 层会混合它们的统计量。改用实例归一化 (instance norm) 或谱归一化 (spectral norm)。
- **Inception score 游戏化。** FID 和 IS 在低样本数下噪声很大。评估时使用 ≥10k 样本。
- **单步采样对条件任务是谎言。** 你仍然需要 CFG scales、截断技巧和重采样来获得可用输出。

## 实际应用

2026 年的 GAN 技术栈：

| 场景 | 选择 |
|-----------|------|
| 照片级真实人脸，固定姿势 | StyleGAN3（最锐利，最小） |
| 动漫/风格化人脸 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 图像到图像翻译 | Pix2Pix / CycleGAN (Phase 8 · 04) 或 ControlNet (Phase 8 · 08) |
| 快速 1 步文本到图像 | 扩散的对抗蒸馏 (SDXL-Turbo, SD3-Turbo) |
| 扩散训练器内部的感知损失 | 图像块上的小 GAN 判别器 |
| 任何多模态、开放式生成 | 不要——改用 diffusion 或 flow matching |

GAN 锐利但狭窄。一旦你的领域打开——照片、任意文本提示、视频——就切换到扩散。对抗技巧作为一种组件存活下来（感知损失、蒸馏），而不是独立的生成器。

## 交付它

保存 `outputs/skill-gan-debugger.md`。该 skill 接收一个失败的 GAN 运行（损失曲线、样本网格、数据集大小）并输出可能原因的有序列表、一行修复方案和重跑协议。

## 练习

1. **简单。** 用默认设置运行 `code/main.py`。然后设置 `D_LR = 5 * G_LR` 并重跑。G 的损失多快就崩溃为一个常数？
2. **中等。** 将 Goodfellow BCE 损失替换为 WGAN 损失：`loss_D = E[D(fake)] - E[D(real)]`，`loss_G = -E[D(fake)]`，并将 D 的权重裁剪到 `[-0.01, 0.01]`。训练更稳定吗？比较 wall-clock 收敛。
3. **困难。** 将 1-D 示例扩展到 2-D 数据（环上的 8 个高斯混合）。跟踪生成器在 1k、5k、10k 步时捕获了多少个 8 个模式。实现 minibatch discrimination 并重新测量。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| Generator | "G" | 噪声到样本网络，`G: z → x̂`。 |
| Discriminator | "D" | 分类器 `D: x → [0, 1]`，真实 vs 伪造。 |
| Minimax | "博弈" | 联合目标的 `min_G max_D`。 |
| Non-saturating loss | "修复" | 对 G 使用 `-log D(G(z))` 而不是 `log(1 - D(G(z)))`。 |
| Mode collapse | "G 只记住了一样东西" | 生成器尽管数据多样，却只产生少数不同输出。 |
| WGAN | "Wasserstein" | 用 Earth-Mover 距离 + 梯度惩罚替换 BCE；更平滑的梯度。 |
| Spectral norm | "Lipschitz 技巧" | 约束 D 的权重范数以限制其斜率；稳定训练。 |
| StyleGAN | "那个能工作的" | 映射网络 + AdaIN；人脸最佳水平，2026 年依然如此。 |

## 生产备忘：单步推理是 GAN 的持久优势

GAN 在开放域生成上不再在样本质量上获胜，但它们在推理成本上仍然获胜。在生产推理文献词汇中，GAN 拥有：

- **没有 prefill，没有 decode 阶段。** 单次 `G(z)` 前向传播。TTFT ≈ 总延迟。
- **没有 KV-cache 压力。** 唯一的状态是权重。批量大小受激活内存限制，而非缓存。
- **平凡的连续批处理。** 由于每个请求消耗相同的固定 FLOPs，服务器目标占用率的静态批次通常是最优的。不需要在途调度器。

这就是 GAN 蒸馏（SDXL-Turbo、SD3-Turbo、ADD、LCM）成为 2026 年快速文本到图像主导技术的原因：它将 20-50 步的扩散 pipeline 折叠为 1-4 次 GAN 风格的前向传播，同时保持扩散基础模型的分布。对抗损失作为一种训练时旋钮存活下来，用于将慢速生成器转化为快速生成器。

## 延伸阅读

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) —— 原始 GAN 论文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) —— 第一个稳定架构。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) —— WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) —— SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) —— SDXL-Turbo。
