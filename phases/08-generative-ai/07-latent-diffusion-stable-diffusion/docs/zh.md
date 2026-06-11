# 隐空间扩散（Latent Diffusion）与 Stable Diffusion

> 在 512×512 的像素空间上做扩散是一种计算上的暴行。Rombach 等人 (2022) 注意到，你不需要全部 78.6 万个维度来生成图像 —— 你只需要足够的维度来捕捉语义结构，剩下的交给一个独立的解码器。在 VAE 的隐空间（latent space）里运行扩散。这一个想法就是 Stable Diffusion。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 分钟

## The Problem

像素空间（pixel-space）扩散在 512² 上意味着 U-Net 运行在形状为 `[B, 3, 512, 512]` 的张量上。对于一个 5 亿参数的 U-Net，每步采样约 100 GFLOPS。50 步就是每张图像 5 TFLOPS。在十亿张图像上训练，计算账单是荒谬的。

这些 FLOPS 大部分花在让感知上并不重要的细节穿过网络 —— 那些有损 VAE 就能压缩掉的高频纹理。Rombach 的想法：一次性训练一个 VAE（*第一阶段*），冻结它，然后在 4 通道 64×64 的隐空间（latent space）里完全运行扩散（*第二阶段*）。同一个 U-Net。1/16 的像素。相同质量下约 64 倍更少的 FLOPS。

这就是 Stable Diffusion 的配方。SD 1.x / 2.x 使用一个 8.6 亿参数的 U-Net 在 `64×64×4` 的隐变量上，SDXL 使用一个 26 亿参数的 U-Net 在 `128×128×4` 上，SD3 将 U-Net 换成了带有 flow matching 的 Diffusion Transformer（DiT）。Flux.1-dev（Black Forest Labs, 2024）搭载了一个 120 亿参数的 MMDiT。它们都运行在相同的两阶段（two-stage）底板上。

## The Concept

![隐空间扩散：VAE 压缩 + 在隐空间里做扩散](../assets/latent-diffusion.svg)

**两个阶段，分别训练。**

1. **第一阶段 —— VAE。** 编码器 `E(x) → z`，解码器 `D(z) → x`。目标压缩率：每个空间轴下采样 8 倍 + 调整通道数，使隐空间总大小约为像素数的 1/16。损失 = 重建（L1 + LPIPS 感知损失）+ KL（权重很小，这样 `z` 不会被强迫过于高斯，因为我们不需要从 `z` 中精确采样）。通常用对抗损失训练，使解码图像锐利。

2. **第二阶段 —— 在 `z` 上做扩散。** 将 `z = E(x_real)` 视为数据。训练一个 U-Net（或 DiT）对 `z_t` 去噪。推理时：通过扩散采样 `z_0`，然后 `x = D(z_0)`。

**文本条件化（Text conditioning）。** 两个额外组件。一个冻结的文本编码器（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。一个交叉注意力（cross-attention）注入：每个 U-Net 块接收 `[Q = 图像特征, K = V = 文本 token]` 并在其中混合。token 是文本影响图像的唯一途径。

**损失函数与第 06 课完全相同。** 同样是 DDPM / flow matching 的 MSE 噪声损失。你只是换了数据域。

## Architecture variants

| 模型 | 年份 | 骨干网络 | 隐空间形状 | 文本编码器 | 参数量 |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 个 token) | 8.6 亿 |
| SD 2.1 | 2022022 | U-Net | 64×64×4 | OpenCLIP-H | 8.65 亿 |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 26 亿 + 66 亿 |
| SDXL-Turbo | 2023 | 蒸馏版 | 128×128×4 | 相同 | 1-4 步采样 |
| SD3 | 2024 | MMDiT（多模态 DiT） | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 20 亿 / 80 亿 |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 120 亿 |
| Flux.1-schnell | 2024 | MMDiT 蒸馏版 | 128×128×16 | T5-XXL + CLIP-L | 120 亿，1-4 步 |

趋势：用 DiT（在隐变量 patch 上的 transformer）替换 U-Net，扩大文本编码器（T5 在提示遵循度上击败 CLIP），增加隐空间通道数（4 → 16 提供更多细节余量）。

## Build It

`code/main.py` 在一个玩具一维 "VAE" 之上堆叠了第 06 课的 DDPM（恒等编码器 + 解码器，仅用于演示；真正的 VAE 会是卷积网络），并添加了类别条件化和 classifier-free guidance（无分类器引导）。它展示了同样的扩散损失无论运行在原始一维值还是编码后的值上都有效 —— 这就是关键洞察。

### Step 1: 编码器/解码器

```python
def encode(x):    return x * 0.5          # 玩具"压缩"到更小的尺度
def decode(z):    return z * 2.0
```

真正的 VAE 有训练好的权重。为了教学，这个线性映射足以展示扩散在 `z` 上运行，而无需关心原始数据空间。

### Step 2: 在 `z`-空间做扩散

与第 06 课相同的 DDPM。网络看到的数据是 `z = E(x)`。采样 `z_0` 后，用 `D(z_0)` 解码。

### Step 3: classifier-free guidance

训练时，10% 的时间丢弃类别标签（替换为空 token）。推理时，同时计算 `ε_cond` 和 `ε_uncond`：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = 无引导（完整多样性），`w = 3` = 默认值，`w = 7+` = 饱和/过度锐利。

### Step 4: 文本条件化（概念，非代码）

用类别标签替换为冻结文本编码器的输出。通过交叉注意力（cross-attention）将文本嵌入送入 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

这就是类条件扩散模型和 Stable Diffusion 之间唯一的实质性区别。

## Pitfalls

- **VAE 尺度不匹配。** SD 1.x 的 VAE 有一个缩放常数（`scaling_factor ≈ 0.18215`），在编码后应用。忘记这个会让 U-Net 在方差完全错误的隐变量上训练。每个 checkpoint 都带一个。
- **文本编码器静默出错。** SD3 需要 T5-XXL 且 >=128 个 token，回退到纯 CLIP 是有损的。始终检查 `use_t5=True`，否则提示遵循度会崩塌。
- **混用隐空间（latent spaces）。** SDXL、SD3、Flux 使用不同的 VAE。在 SDXL 隐变量上训练的 LoRA 无法在 SD3 上工作。Hugging Face diffusers 0.30+ 拒绝加载不匹配的 checkpoint。
- **CFG 过高。** `w > 10` 会产生饱和、油腻的图像，以多样性为代价过度拟合提示。最佳区间是 `w = 3-7`。
- **负面提示泄漏。** 空负面提示变成空 token；填写的负面提示变成 `ε_uncond`。它们不一样；有些 pipeline 静默默认使用空 token。

## Use It

2026 年的生产级技术栈：

| 目标 | 推荐骨干网络 |
|--------|----------------------|
| 狭窄领域，配对数据，从零训练模型 | SDXL 微调（LoRA / 全参数）—— 最快交付 |
| 开放域文本到图像，开放权重 | Flux.1-dev（120 亿，Apache / 非商业）或 SD3.5-Large |
| 最快推理，开放权重 | Flux.1-schnell（1-4 步，Apache）或 SDXL-Lightning |
| 最佳提示遵循度，托管服务 | GPT-Image / DALL-E 3（仍领先）、Midjourney v7、Imagen 4 |
| 编辑工作流 | Flux.1-Kontext（2024 年 12 月）—— 原生接受图像 + 文本 |
| 研究，基线 | SD 1.5 —— 古老但被充分研究 |

## Ship It

保存 `outputs/skill-sd-prompter.md`。该 skill 接收一个文本提示 + 目标风格，输出：模型 + checkpoint、CFG 尺度、采样器、负面提示、分辨率、可选的 ControlNet/IP-Adapter 组合，以及每步 QA 检查清单。

## Exercises

1. **简单。** 用 guidance `w ∈ {0, 1, 3, 7, 15}` 运行 `code/main.py`。记录每类的平均样本。在哪个 `w` 下类均值会偏离真实数据均值？
2. **中等。** 将玩具线性编码器替换为带有重建损失的 tanh-MLP 编码器/解码器对。在新的隐变量上重新训练扩散。样本质量会改变吗？
3. **困难。** 用 diffusers 搭建真正的 Stable Diffusion 推理：加载 `sdxl-base`，用 CFG=7 运行 30 步 Euler，计时。然后切换到 `sdxl-turbo`，4 步，CFG=0。同一主题，不同质量 —— 描述发生了什么以及为什么。

## Key Terms

| 术语 | 人们的说法 | 实际含义 |
|------|----------|---------|
| First stage | "VAE" | 训练好的编码器/解码器对；将 512² 压缩到 64²。 |
| Second stage | "U-Net" | 在隐空间（latent space）上的扩散模型。 |
| CFG | "引导尺度" | `(1+w)·ε_cond - w·ε_uncond`；调节条件强度。 |
| Null token | "空提示嵌入" | 用于 `ε_uncond` 的无条件嵌入。 |
| Cross-attention | "文本怎么进去" | 每个 U-Net 块将文本 token 作为 K 和 V 进行注意力计算。 |
| DiT | "扩散 Transformer" | 用 latent patch 上的 transformer 替换 U-Net；扩展性更好。 |
| MMDiT | "多模态 DiT" | SD3 的架构：文本和图像流联合注意力。 |
| VAE scaling factor | "魔法数字" | 将隐变量除以 ~5.4，使扩散在单位方差空间运行。 |

## Production note: 在 8GB 消费级 GPU 上运行 Flux-12B

Flux 的参考集成是经典的"我有一块消费级 GPU，能部署吗？"配方。技巧与生产推理文献中列出的三旋钮配方相同，应用于扩散 DiT：

1. **交错加载。** Flux 有三个网络，永远不需要同时存在于 VRAM 中：T5-XXL 文本编码器（fp32 下约 10 GB）、CLIP-L（小）、120 亿参数的 MMDiT，以及 VAE。先编码提示，*删除*编码器，加载 DiT，去噪，*删除* DiT，加载 VAE，解码。消费级 8GB GPU 一次只能装下一个阶段。
2. **通过 bitsandbytes 做 4-bit 量化。** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`，同时应用于 T5 编码器和 DiT。内存降低 8 倍，按 Aritra 的基准测试（链接在 notebook 中），文本到图像的质量下降不可感知。
3. **CPU 卸载。** `pipe.enable_model_cpu_offload()` 在每个前向传播推进时自动在 CPU 和 GPU 之间交换模块。增加 10-20% 延迟，但让 pipeline 能跑起来。

内存账本是：`10 GB T5 / 8 = 1.25 GB` 量化后，`12 B 参数 × 0.5 字节 = ~6 GB` 量化 DiT，加上激活值。用 stas00 的术语，这是 TP=1 推理的极端端 —— 没有模型并行，最大量化。生产环境你会在 H100 上跑 TP=2 或 TP=4；对于单个开发笔记本，这就是配方。

## Further Reading

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) —— Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) —— SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) —— DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) —— SD3, MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) —— CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) —— Flux.1 家族。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) —— 上述每个 checkpoint 的参考实现。
