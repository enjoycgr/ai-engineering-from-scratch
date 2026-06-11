# 条件 GAN 与 Pix2Pix

> 2014-2017 年的第一个重大突破是控制 GAN 生成什么。附加一个标签、一张图片或一句话。Pix2Pix 完成了图像版本，在狭窄的图像到图像任务上仍然优于所有通用文本到图像模型。

**类型:** Build
**语言:** Python
**前置条件:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**时间:** ~75 分钟

## 问题

无条件 GAN（unconditional GAN）随机采样任意人脸。适合演示，但在生产环境中毫无用处。你想要的是：*将草图映射为照片*、*将地图映射为航拍照片*、*将白天场景映射为夜晚*、*为灰度图像上色*。在所有这些场景中，你都会获得输入图像 `x`，并必须输出与之有语义对应关系的 `y`。每个 `x` 对应多个合理的 `y`。均方误差（mean-squared error）会把它们压成 mush。对抗损失（adversarial loss）不会，因为"看起来真实"是锐利的。

条件 GAN（conditional GAN, Mirza & Osindero, 2014）将条件 `c` 作为输入同时添加到 `G` 和 `D` 中。Pix2Pix（Isola et al., 2017）将其特化：条件是一整张输入图像，生成器（generator）是 U-Net，判别器（discriminator）是*基于 patch* 的分类器（PatchGAN），损失函数（loss）由对抗损失 + L1 组成。这个配方即使到了 2026 年，在狭窄的图像到图像领域仍然优于从零开始的文本到图像模型，因为它是在*配对数据（paired data）*上训练的——你正好拥有所需的信号。

## 概念

![Pix2Pix：U-Net 生成器，PatchGAN 判别器](../assets/pix2pix.svg)

**条件 G（Conditional G）。** `G(x, z) → y`。在 Pix2Pix 中，`z` 是 G 内部的 dropout（没有输入噪声——Isola 发现显式噪声被忽略了）。

**条件 D（Conditional D）。** `D(x, y) → [0, 1]`。输入是*配对*（条件，输出）。这是关键区别：D 必须判断 `y` 是否与 `x` 一致，而不仅仅是 `y` 看起来是否真实。

**U-Net 生成器（U-Net generator）。** 带 bottleneck 跨层跳跃连接（skip connections）的编码器-解码器（encoder-decoder）。对于输入和输出共享低级结构（边缘、轮廓）的任务至关重要。没有跳跃连接，高频细节会消失。

**PatchGAN 判别器（PatchGAN discriminator）。** 不是输出单一的 real/fake 分数，而是输出 `N×N` 网格，其中每个单元格判断约 70×70 像素的感受野（receptive field）。取平均。这是一个马尔可夫随机场（Markov random field）假设：真实感是局部的。训练更快，参数更少，输出更锐利。

**损失函数（Loss）。**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 项稳定训练并将 G 推向已知目标。L1 比 L2 给出更锐利的边缘（中位数，而非均值）。Pix2Pix 默认 `λ = 100`。

## CycleGAN —— 当你没有配对数据时

Pix2Pix 需要配对的 `(x, y)` 数据。CycleGAN（Zhu et al., 2017）放弃了这一要求，代价是额外的损失：*循环一致性（cycle consistency）*损失。两个生成器 `G: X → Y` 和 `F: Y → X`。训练它们使得 `F(G(x)) ≈ x` 和 `G(F(y)) ≈ y`。这让你可以在没有配对示例的情况下将马翻译为斑马，夏天翻译为冬天。

到了 2026 年，未配对的图像到图像任务大多通过扩散（diffusion）完成（ControlNet、IP-Adapter），而不是 CycleGAN，但循环一致性的思想在几乎每一篇未配对域自适应（unpaired domain adaptation）论文中得以延续。

## Build It

`code/main.py` 实现了一个基于 1-D 数据的微型条件 GAN（conditional GAN）。条件 `c` 是类别标签（0 或 1）。任务：为给定类别从条件分布（conditional distribution）中生成样本。

### 第 1 步：将条件附加到 G 和 D 的输入

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot 编码是最简单的方式。更大的模型使用 learned embedding、FiLM modulation 或 cross-attention。

### 第 2 步：条件训练

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

生成器（generator）必须匹配给定条件下的真实分布（real distribution），而不是边缘分布（marginal）。

### 第 3 步：验证每类输出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Pitfalls

- **条件被忽略（Condition ignored）。** G 学会了边缘化（marginalize），D 从不惩罚，因为条件信号太弱。修复：更激进地将条件输入 D（早期层，而非仅后期层），使用 projection discriminator（Miyato & Koyama 2018）。
- **L1 权重太低。** G 漂移到任意看起来真实的输出，而不是忠实于输入。对于 Pix2Pix 风格的任务，从 λ≈100 开始。
- **L1 权重太高。** G 产生模糊输出，因为 L1 仍然是 L_p 范数。训练稳定后向下退火（anneal down）。
- **D 中的 ground-truth 泄漏。** 将 `(x, y)` 拼接为 D 的输入，而不仅仅是 `y`。没有这一点，D 无法检查一致性。
- **每类模式崩溃（Mode collapse per class）。** 每个类别可以独立崩溃。运行按类别条件化的多样性检查（class-conditional diversity checks）。

## Use It

2026 年图像到图像任务的状态：

| 任务 | 最佳方法 |
|------|----------|
| 草图 → 照片，同域，配对数据 | Pix2Pix / Pix2PixHD（仍然快，仍然锐利） |
| 草图 → 照片，未配对 | 带 Scribble conditioning model 的 ControlNet |
| 语义分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 风格迁移 | 带 IP-Adapter 或 LoRA 的扩散；GAN 方法已是 legacy |
| 深度 → 照片 | 基于 Stable Diffusion 的 ControlNet-Depth |
| 超分辨率 | Real-ESRGAN（GAN）、ESRGAN-Plus 或 SD-Upscale（扩散） |
| 上色 | ColTran、基于扩散的上色器或 Pix2Pix-color |
| 白天 → 夜晚、季节、天气 | CycleGAN 或基于 ControlNet 的方法 |

当你满足以下条件时，Pix2Pix 仍然是正确的工具：（a）你有数千对配对示例，（b）任务狭窄且可重复，（c）你需要快速推理。在通用开放域任务上，扩散（diffusion）获胜。

## Ship It

保存 `outputs/skill-img2img-chooser.md`。该 skill 接受任务描述、数据可用性（paired vs unpaired、N 个样本）以及延迟/质量预算，然后输出：方法（Pix2Pix、CycleGAN、ControlNet 变体、SDXL + IP-Adapter）、训练数据要求、推理成本以及评估协议（LPIPS、FID、任务特定指标）。

## 练习

1. **简单。** 修改 `code/main.py` 添加第三个类别。确认 G 仍然将每个类别的噪声映射到正确的模式。
2. **中等。** 在 1-D 设置中将 L1 替换为感知风格损失（perceptual-style loss）（例如一个小型冻结 D 作为特征提取器）。它会改变条件分布（conditional distribution）的锐利度吗？
3. **困难。** 在 1-D 设置中勾勒一个 CycleGAN：两个分布、两个生成器、循环损失。展示它在没有配对数据的情况下学会了在它们之间映射。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Conditional GAN | "带标签的 GAN" | G(z, c), D(x, c)。两个网络都看到条件。 |
| Pix2Pix | "图像到图像 GAN" | 配对 cGAN，U-Net G 和 PatchGAN D + L1 损失。 |
| U-Net | "带跳跃连接的编码器-解码器" | 对称卷积网络；跳跃连接保留高频。 |
| PatchGAN | "局部真实感分类器" | D 输出每 patch 分数而非全局分数。 |
| CycleGAN | "未配对图像翻译" | 两个 G + 循环一致性损失；无需配对数据。 |
| SPADE | "GauGAN" | 用语义图归一化中间激活；分割到图像。 |
| FiLM | "特征级线性调制" | 来自条件的逐特征仿射变换；廉价条件化。 |

## 生产备注：Pix2Pix 作为延迟受限基线

当你有配对数据且任务狭窄（草图 → 渲染、语义图 → 照片、白天 → 夜晚）时，Pix2Pix 的单次推理在延迟上比扩散快一个数量级。生产对比通常是：

| 路径 | 步数 | 512² 单张 L4 典型延迟 |
|------|------|------------------------|
| Pix2Pix（U-Net 前向） | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix 在静态批次中赢得吞吐量（每个请求的 FLOPs 相同）。扩散在质量和泛化上获胜。现代策略通常是：为狭窄任务部署 Pix2Pix 风格的蒸馏模型，并为尾部输入提供扩散回退。

## 延伸阅读

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) —— cGAN 论文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) —— Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) —— CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) —— Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) —— SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) —— projection D。
