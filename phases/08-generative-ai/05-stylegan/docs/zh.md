# StyleGAN

> 大多数生成器（generator）同时在每一层搅拌 `z`。StyleGAN 将其拆分：首先将 `z` 映射到中间空间 `w`，然后通过 AdaIN 在每个分辨率层级*注入* `w`。这一单一改变解耦了潜在空间（latent space），并使逼真人脸在七年内成为已解决的问题。

**类型:** Build
**语言:** Python
**前置条件:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**时间:** ~45 分钟

## 问题

DCGAN 通过转置卷积（transposed convolutions）堆栈将 `z` 映射为图像。问题：`z` 控制一切——姿态、光照、身份、背景——全部纠缠在一起。沿 `z` 的一个轴移动，四个都会改变。你无法要求模型"同一个人，不同姿态"，因为表示没有按这种方式分解。

Karras et al. (2019, NVIDIA) 提出：停止将 `z` 直接输入卷积层。用一个恒定的 `4×4×512` 张量作为网络输入。学习一个 8 层 MLP，将 `z ∈ Z → w ∈ W`。通过*自适应实例归一化（adaptive instance normalization, AdaIN）*在每个分辨率注入 `w`：归一化每个卷积特征图（conv feature map），然后通过 `w` 的仿射投影进行缩放和平移。添加逐层噪声以产生随机细节（皮肤毛孔、发丝）。

结果：`W` 拥有大致正交的轴，分别对应"高级风格"（姿态、身份）和"精细风格"（光照、颜色）。你可以通过在低分辨率层级使用图像 A 的 `w`、在高分辨率层级使用图像 B 的 `w` 来在两幅图像之间交换风格。这解锁了编辑、跨域风格化以及整个"StyleGAN 反演（StyleGAN-inversion）"研究路线。

## 概念

![StyleGAN：映射网络 + AdaIN + 逐层噪声](../assets/stylegan.svg)

**映射网络（Mapping network）。** `f: Z → W`，一个 8 层 MLP。`Z = N(0, I)^512`。`W` 不被强制为高斯分布——它学习数据自适应的形状。

**合成网络（Synthesis network）。** 从一个学习的常数 `4×4×512` 开始。每个分辨率块：`上采样 → 卷积 → AdaIN(w_i) → 噪声 → 卷积 → AdaIN(w_i) → 噪声`。分辨率翻倍：4、8、16、32、64、128、256、512、1024。

**AdaIN。**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

其中 `y_scale` 和 `y_bias` 来自 `w` 的仿射投影。按特征图归一化，然后重设风格。这里的"风格"是特征图的一阶和二阶统计量。

**逐层噪声（Per-layer noise）。** 单通道高斯噪声（Gaussian noise）添加到每个特征图，由学习的逐通道因子缩放。控制随机细节而不影响全局结构。

**截断技巧（Truncation trick）。** 在推理时，采样 `z`，计算 `w = mapping(z)`，然后 `w' = ŵ + ψ·(w - ŵ)`，其中 `ŵ` 是许多样本上的平均 `w`。`ψ < 1` 以多样性换取质量。几乎每个 StyleGAN 演示都使用 `ψ ≈ 0.7`。

## StyleGAN 1 → 2 → 3

| 版本 | 年份 | 创新 |
|---------|------|-----------|
| StyleGAN | 2019 | 映射网络 + AdaIN + 噪声 + 渐进增长。 |
| StyleGAN2 | 2020 | 权重解调（weight demodulation）替代 AdaIN（修复液滴伪影）；跳跃/残差架构；路径长度正则化（path-length regularization）。 |
| StyleGAN3 | 2021 | 无别名卷积（alias-free convolution）+ 等变核；消除纹理粘附到像素网格。 |
| StyleGAN-XL | 2022 | 类别条件化，1024²，ImageNet。 |
| R3GAN | 2024 | 以更强正则化重新品牌；在 FFHQ-1024 上缩小与扩散的差距，参数少 20 倍。 |

到了 2026 年，StyleGAN3 仍然是以下场景默认选择：（a）高 FPS 的狭窄域照片级真实感，（b）小样本域自适应（用 100 张图像训练新数据集，冻结映射网络），（c）基于反演的编辑（找到重建真实照片的 `w`，然后编辑该 `w`）。对于开放域文本到图像，它不是合适的工具——扩散（diffusion）才是。

## Build It

`code/main.py` 实现了一个 1-D 的玩具"StyleGAN lite"：一个映射 MLP，一个合成函数，它接受一个学习到的常数向量并用 `w` 衍生的缩放/偏置对其进行调制，以及逐层噪声。它展示了通过仿射调制注入 `w` 匹配或优于将 `z` 拼接到生成器输入。

### 第 1 步：映射网络

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 第 2 步：自适应实例归一化

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

逐特征图的缩放和平移来自 `w` 的线性投影。

### 第 3 步：逐层噪声

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma 逐通道是可学习的。

## Pitfalls

- **液滴伪影（Droplet artifacts）。** StyleGAN 1 在特征图中产生块状液滴，因为 AdaIN 将均值归零。StyleGAN 2 的权重解调通过缩放卷积权重而非激活来修复它。
- **纹理粘附（Texture sticking）。** StyleGAN 1 和 2 的纹理跟随像素坐标而非物体坐标（插值时可见）。StyleGAN 3 的无别名卷积通过加窗 sinc 滤波器修复了这一点。
- **模式覆盖（Mode coverage）。** 截断 `ψ < 0.7` 看起来干净，但从狭窄锥体采样；如果需要多样性，使用 `ψ = 1.0`。
- **反演是有损的（Inversion is lossy）。** 将真实照片反演到 `W` 通常通过优化或编码器（e4e、ReStyle、HyperStyle）完成。结果在多次迭代中会漂移。

## Use It

| 用例 | 方法 |
|----------|----------|
| 照片级真实人脸（动漫、产品、狭窄域） | StyleGAN3 FFHQ / 自定义微调 |
| 从照片进行人脸编辑 | e4e 反演 + StyleSpace / InterFaceGAN 方向 |
| 换脸 / 重演 | StyleGAN + 编码器 + 混合 |
| 头像管线 | 带 ADA 的 StyleGAN3 用于小数据微调 |
| 从少量图像进行域自适应 | 冻结映射网络，微调合成网络 |
| 多模态或文本条件化生成 | 不要——用扩散 |

对于产品级 demo，答案是"人物面部的照片"时，StyleGAN 在推理成本（单次前向传播，4090 上 <10ms）和相同质量门槛的锐利度上击败扩散。

## Ship It

保存 `outputs/skill-stylegan-inversion.md`。该 skill 接受一张真实照片并输出：反演方法（e4e / ReStyle / HyperStyle）、期望的潜在损失（latent loss）、编辑预算（在 `W` 中可以移动多远才会出现伪影），以及已知有效的编辑方向列表（年龄、表情、姿态）。

## 练习

1. **简单。** 以 `adain_on=True` 和 `adain_on=False` 运行 `code/main.py`。比较固定 latent 与扰动 latent 下输出的分布。
2. **中等。** 实现混合正则化（mixing regularization）：对于一个训练批次，计算 `w_a`、`w_b`，并对前半段合成应用 `w_a`，后半段应用 `w_b`。解码器会学习解耦的风格吗？
3. **困难。** 取一个预训练的 StyleGAN3 FFHQ 模型（ffhq-1024.pkl）。通过在标注样本上训练 SVM 找到控制"微笑"的 `w` 方向；报告在身份漂移前可以推多远。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Mapping network | "那个 MLP" | `f: Z → W`，8 层，将潜在几何与数据统计解耦。 |
| W space | "风格空间" | 映射网络的输出；大致解耦。 |
| AdaIN | "自适应实例归一化" | 归一化特征图，然后按 `w` 投影进行缩放 + 平移。 |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`，ψ<1 以多样性换取质量。 |
| Path-length regularization | "PL reg" | 惩罚 `w` 单位变化引起的图像巨大变化；使 `W` 更平滑。 |
| Weight demodulation | "StyleGAN2 的修复" | 归一化卷积权重而非激活；消除液滴伪影。 |
| Alias-free | "StyleGAN3 的技巧" | 加窗 sinc 滤波器；消除纹理粘附到像素网格。 |
| Inversion | "为真实图像找到 w" | 优化或编码 `x → w` 使得 `G(w) ≈ x`。 |

## 生产备注：为什么 StyleGAN 在 2026 年仍然在产线中

StyleGAN3 在 4090 上生成 1024² 的 FFHQ 人脸只需不到 10 ms——`num_steps = 1`，无 VAE 解码，无交叉注意力（cross-attention）传递。按生产术语，这是任何图像生成器的地板延迟。50 步 SDXL + VAE-解码管线在相同分辨率下约需 3 秒。这是 **300 倍差距**，对于狭窄域产品（头像服务、证件照管线、库存人脸生成），它在总拥有成本（TCO）上获胜。

两个运营后果：

- **没有调度器，没有批处理器。** 在目标占用率下的静态批次是最优的。连续批处理（对 LLM 和扩散至关重要）提供零好处，因为每个请求消耗相同的 FLOPs。
- **截断 `ψ` 是安全旋钮。** `ψ < 0.7` 从映射网络范围的狭窄锥体采样。这是服务层对样本方差的唯一控制杆。峰值负载时降低 `ψ`，为高级用户提高 `ψ`。

## 延伸阅读

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) —— StyleGAN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) —— StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) —— StyleGAN3。
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) —— e4e 反演。
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) —— StyleGAN-XL。
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) —— 现代极简 GAN 配方。
