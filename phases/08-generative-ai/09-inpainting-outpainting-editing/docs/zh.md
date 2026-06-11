# Inpainting、Outpainting 与图像编辑

> Text-to-image 创造新事物。Inpainting 修复旧事物。在生产中，70% 的可计费图像工作是编辑 —— 更换背景、移除 logo、扩展画布、重新生成一只手。Inpainting 是 diffusion 赚钱的地方。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 分钟

## The Problem

客户发送了一张完美的产品照片，但背景中有一个分散注意力的标志。你想抹除标志并让其他一切像素级保持不变。你不能从零开始运行 text-to-image —— 结果会有不同的颜色、不同的光照、不同的产品角度。你想*只*重新生成 masked（被掩码的）区域，并且希望重新生成尊重周围的上下文。

这就是 inpainting（图像修复）。变体：

- **Inpainting。** 在 mask 内部重新生成，保持外部像素。
- **Outpainting。** 在 mask 外部（或画布之外）重新生成，保持内部。
- **Image editing。** 重新生成整张图像，但保持对原始的语义或结构保真度（SDEdit、InstructPix2Pix）。

2026 年的每个 diffusion pipeline 都附带 inpainting 模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它们基于相同的原理工作。

## The Concept

![Inpainting: 感知 mask 的去噪，带有上下文保留的重注入](../assets/inpainting.svg)

### The naive approach (and why it's wrong)

用 mask 运行标准 text-to-image。在每个 sampling step（采样步），用 clean image 的 forward-diffused（前向扩散）版本替换 noisy latent 中未掩码的区域。它能工作……但很糟糕。边界 artifacts 会渗透，因为模型对掩码区域内有什么没有信息。

### The proper inpainting model

训练一个修改后的 U-Net，接受 9 个 input channels（输入通道）而不是 4 个：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

额外的通道是 VAE-encoded source image 的副本加上一个单通道 mask。在训练时，你随机 mask 图像的区域并训练模型只去噪 masked region，而未掩码区域作为 clean conditioning signal 给出。在 inference 时，模型可以 "看到" 掩码区域周围的内容并产生连贯的补全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用这种 9-channel（或类似）输入。Diffusers `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit (Meng et al., 2022) —— 免费编辑

将 noise 添加到 source image 直到某个中间 `t`，然后从 `t` 向下运行 reverse chain（反向链）到 0，使用新的 prompt。无需重新训练。起始 `t` 的选择在 fidelity（保真度）和 creative freedom（创作自由度）之间权衡：

- `t/T = 0.3` → 与 source 几乎相同，小的风格变化
- `t/T = 0.6` → 中等编辑，保留粗略结构
- `t/T = 0.9` → 从近 noise 生成，最小 source preservation

### InstructPix2Pix (Brooks et al., 2023)

在 `(input_image, instruction, output_image)` 三元组上微调 diffusion model。在 inference 时，对 input image 和 text instruction 进行 conditioning（"make it sunset"、"add a dragon"）。两个 CFG scales：image scale 和 text scale。

### RePaint (Lugmayr et al., 2022)

保留一个标准 unconditional diffusion model。在每个 reverse step，重新采样 —— 偶尔跳回更 noisy 的状态并重新生成。避免边界 artifacts。在你没有训练好的 inpainting model 时使用。

## Build It

`code/main.py` 在 5 维数据上实现了一个 toy 1-D inpainting 方案。我们在两个 cluster 的 5 维混合数据上训练 DDPM，其中每个样本是来自两个 cluster 之一的 5 个 float。在 inference 时，我们 "mask" 5 个维度中的 2 个，在每个 step 注入未掩码三个维度的 noisy-forward 版本，并只重新生成被 mask 的维度。

### Step 1: 5-D DDPM data

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: train denoiser over all 5 dims

标准 DDPM。Net 为 5 维 noisy input 输出 5 维 noise prediction。

### Step 3: at inference, mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

这是 naive approach，在 toy 1-D 数据上有效。真正的图像 inpainting 使用 9-channel input，因为 texture coherence（纹理连贯性）更重要。

### Step 4: outpainting

Outpainting 就是 mask 反转的 inpainting：mask 新的（之前不存在的）画布，用原始图像填充其余部分。相同的训练目标。

## Pitfalls

- **Seams（接缝）。** Naive approach 留下可见的边界，因为梯度信息不跨 mask 流动。修复：将 mask 膨胀 8-16 像素，或使用 proper inpainting model。
- **Mask leakage。** 如果 conditioning image 的未掩码区域质量低或 noisy，它会污染 mask 内部的生成。稍微去噪或模糊。
- **CFG 与 mask size 交互。** 小 mask 上的高 CFG = 饱和的 patch。对于小编辑降低 CFG。
- **SDEdit fidelity cliff。** 从 `t/T = 0.5` 到 `t/T = 0.6` 可能会丢失主体的 identity。扫描并 checkpoint。
- **Prompt mismatch。** Prompt 应该描述*整张*图像，不只是新内容。"A cat sitting on a chair" 而不是 "a cat"。

## Use It

| Task | Pipeline |
|------|----------|
| Remove object, small mask | SD-Inpaint 或 Flux-Fill，标准 prompt |
| Replace sky | SD-Inpaint + "blue sky at sunset" |
| Extend canvas | SDXL outpaint mode (8px feather) 或 Flux-Fill with outpaint mask |
| Regenerate hand / face | SD-Inpaint with prompt re-describing the subject + ControlNet-Openpose |
| Change style of one region | SDEdit at `t/T=0.5` on masked region |
| "Make it sunset" | InstructPix2Pix 或 Flux-Kontext |
| Background replacement | SAM mask → SD-Inpaint |
| Ultra-high-fidelity | Flux-Fill 或 GPT-Image (hosted) for hardest cases |

SAM (Meta's Segment Anything, 2023) + diffusion inpaint 是 2026 年的 background-removal pipeline。SAM 2 (2024) 在视频上工作。

## Ship It

保存 `outputs/skill-editing-pipeline.md`。Skill 接收原始图像 + 编辑描述 + 可选 mask（或 SAM prompt）并输出：mask-generation approach、base model、CFG scales（image + text）、SDEdit-t 或 inpainting mode、以及 QA checklist。

## Exercises

1. **Easy。** 在 `code/main.py` 中，将被 mask 的维度比例从 0.2 变到 0.8。在什么比例下，inpaint quality（被掩码维度的残差）等于 unconditional generation？
2. **Medium。** 实现 RePaint：每 10 个 reverse step，跳回 5 个 step（添加 noise）并重新去噪。测量它是否减少了 mask edge 的边界残差。
3. **Hard。** 使用 Hugging Face diffusers 来比较：SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill 在 20 个 face-regeneration 任务上。分别打分 pose adherence 和 identity preservation。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Inpainting | "Fill the hole" | 在 mask 内部重新生成；保持外部像素。 |
| Outpainting | "Extend the canvas" | 在画布外部重新生成；保持内部。 |
| 9-channel U-Net | "Proper inpainting model" | 以 `noisy | encoded-source | mask` 为输入的 U-Net。 |
| SDEdit | "Img2img with noise level" | Noise 到时间 `t`，用新 prompt 去噪。 |
| InstructPix2Pix | "Text-only edits" | 在 (image, instruction, output) 三元组上微调的 diffusion。 |
| RePaint | "No retraining" | 在 reverse 期间定期重新加 noise 以减少接缝。 |
| SAM | "Segment Anything" | 通过点击或框生成 mask；与 inpaint 配对。 |
| Flux-Kontext | "Edit with context" | 接受 reference image + instruction 进行编辑的 Flux 变体。 |

## Production note: edit pipelines are latency-sensitive

编辑图像的用户期望亚 5 秒的往返。30 步 SDXL-Inpaint 在 1024² 上 L4 是 3-4 秒，加上 SAM mask generation (~200 ms) 和 VAE encode/decode (~500 ms 合计)。在生产框架中，这是 TTFT-bound 而不是 throughput-bound —— batch 1、低并发、最小化每个阶段：

- **SAM-H 是慢的那个。** SAM-H 在 1024² 上约 200 ms；SAM-ViT-B 约 40 ms 且质量损失轻微。SAM 2 (video) 增加 temporal overhead；不要将其用于单图像编辑。
- **尽可能跳过 encode。** `pipe.image_processor.preprocess(img)` 编码为 latents。如果你有来自前一次生成的 latents（迭代编辑 UI 中的典型情况），通过 `latents=...` 直接传递它们以跳过一个 VAE encode。
- **Mask dilation 对 throughput 也很重要。** 小 mask 意味着 U-Net forward pass 的大部分被浪费（未掩码像素无论如何都被钳制）。`diffusers` 的 `StableDiffusionInpaintPipeline` 无论如何运行完整 U-Net；只有 9-channel proper-inpaint 变体利用 masked compute。
- **Flux-Kontext 是 2025 年的答案。** 对 `(source_image, instruction)` 的 single forward pass —— 无需单独 mask、无需 SDEdit noise sweep。在 H100 上约 1.5 秒完成编辑。架构教训：坍缩阶段。

## Further Reading

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 无需训练的 inpainting。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文本指令编辑。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，mask 来源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 视频 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — attention-level 编辑。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 工具。
