# ControlNet、LoRA 与 Conditioning

> 仅凭文本是一种笨拙的控制信号。ControlNet 让你克隆一个预训练的 diffusion（扩散）模型，并用深度图、姿态骨架、涂鸦或边缘图像来引导它。LoRA 让你通过训练一千万参数来微调一个二十亿参数的模型。二者共同将 Stable Diffusion 从一个玩具变成了 2026 年每个 agency 都在部署的图像管线。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**Time:** ~75 分钟

## The Problem

像 "a woman in a red dress walking a dog on a busy street" 这样的提示词，没有给模型提供关于狗*在哪*、女人*是什么姿态*、街道*是什么透视*的信息。文本只能锁定大约 10% 指定一张图像所需的信息。其余部分是视觉的，无法用文字高效描述。

为每一种信号（姿态、深度、边缘、分割）从头训练一个新的 conditional（条件化）模型是 prohibitive（不可行的）。你希望保持 26 亿参数的 SDXL backbone（主干网络）冻结，附加一个读取 conditioning（条件控制）的小型 side-network（旁路网络），并让它轻轻推动 backbone 的中间特征。这就是 ControlNet。

你还想教会模型新概念（你的脸、你的产品、你的风格），而不重新训练整个模型。你想要一个缩小 100 倍的 delta。这就是 LoRA —— low-rank adapters（低秩适配器），插入现有的 attention（注意力）权重中。

ControlNet + LoRA + 文本 = 2026 年从业者的工具箱。大多数生产图像管线会在 SDXL / SD3 / Flux base 之上叠加 2-5 个 LoRA、1-3 个 ControlNet 和一个 IP-Adapter。

## The Concept

![ControlNet 克隆编码器；LoRA 添加低秩增量](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

取一个预训练的 SD。*克隆* U-Net 的编码器一半。冻结原始网络。训练克隆体接受额外的 conditioning input（条件输入）（边缘、深度、姿态）。通过 *zero-convolution（零卷积）* skip connections（跳跃连接）（初始化为零的 1×1 卷积 —— 开始时无操作，学习一个增量）将克隆体连接回原始网络的解码器一半。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv init（零卷积初始化）意味着 ControlNet 从 identity（恒等映射）开始 —— 甚至在训练之前也不会造成损害。使用标准的 diffusion loss（扩散损失），在 100 万个 (prompt, condition, image) 三元组上训练。

每种模态的 ControlNet 作为小型 side model（旁路模型）发布（SDXL 约 360M，SD 1.5 约 70M）。你可以在 inference（推理）时将它们组合：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

对于模型中的任何线性层 `W ∈ R^{d×d}`，冻结 `W` 并添加一个低秩增量：

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

其中 `r << d`。attention 的 rank 4-16 是标准值，重型微调的 rank 64-128。新参数数量：`2 · d · r` 而不是 `d²`。对于 SDXL attention 中 `d=640`、`r=16`：每个 adapter 20k 参数而不是 410k —— 减少了 20 倍。在整个模型中：一个 LoRA 通常是 20-200MB，而 base 是 5GB。

在 inference 时你可以缩放 LoRA：`W' = W + α · B @ A`。`α = 0.5-1.5` 是正常的。多个 LoRA 以 additive（加性）方式堆叠（带有它们以非线性方式交互的通常警告）。

### IP-Adapter (Ye et al., 2023)

一个微型 adapter（适配器），接受*图像*作为 conditioning（与文本一起）。使用 CLIP image encoder 生成 image tokens，将它们注入 cross-attention（交叉注意力）中与文本 token 并排。每个 base model 约 20MB。让你可以执行 "generate an image in the style of this reference"，而无需 LoRA。

## Composability matrix

| Tool | What it controls | Size | When to use |
|------|------------------|------|-------------|
| ControlNet | Spatial structure (pose, depth, edges) | 70-360MB | 精确布局、构图 |
| LoRA | Style, subject, concept | 20-200MB | 个性化、风格 |
| IP-Adapter | Style or subject from reference image | 20MB | 文本无法描述外观 |
| Textual Inversion | Single concept as a new token | 10KB |  legacy，大多被 LoRA 取代 |
| DreamBooth | Full fine-tune on a subject | 2-5GB | 强身份、高算力 |
| T2I-Adapter | Lighter ControlNet alternative | 70MB | 边缘设备、推理预算 |

ControlNet ≈ 空间。LoRA ≈ 语义。两者一起使用。

## Build It

`code/main.py` 在一维上模拟两种机制：

1. **LoRA。** 一个预训练的线性层 `W`。冻结它。训练一个低秩 `B @ A`，使得 `W + BA` 匹配目标线性层。展示 `r = 1` 足以完美学习一个 rank-1 correction。

2. **ControlNet-lite。** 一个 "frozen base（冻结基座）" predictor 和一个读取额外信号的 "side network（旁路网络）"。side network 的输出由一个初始化为零的可学习标量 gated（门控）（我们版本的 zero-conv）。训练并观察 gate 逐渐上升。

### Step 1: LoRA math

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Step 2: zero-init side network

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在第 0 步时，输出与 base 完全相同。早期训练缓慢更新 `gate` —— 没有灾难性的漂移。

## Pitfalls

- **Over-scaling LoRAs。** `α = 2` 或 `α = 3` 是一种常见的 "make it stronger" hack，会产生过度风格化 / 损坏的输出。保持 `α ≤ 1.5`。
- **ControlNet weight conflict。** 以权重 1.0 使用 Pose ControlNet 并以权重 1.0 使用 Depth ControlNet 通常会 overshoot（过冲）。权重总和 ≈ 1.0 是一个安全的默认值。
- **LoRA on the wrong base。** SDXL LoRA 在 SD 1.5 上会静默 no-op（无操作），因为 attention 维度不匹配。Diffusers 在 0.30+ 会发出警告。
- **Textual Inversion drift。** 在一个 checkpoint 上训练的 token 在另一个 checkpoint 上会严重漂移。LoRA 更具可移植性。
- **LoRA weight-merging and storage。** 你可以将 LoRA 烘焙进 base model 权重中以实现更快的 inference（无需运行时加法），但你会失去在运行时缩放 `α` 的能力。保留两个版本。

## Use It

| Goal | 2026 pipeline |
|------|---------------|
| Reproduce a brand's art style | LoRA 在 rank 32 上训练约 30 张精选图像 |
| Put my face in a generated image | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| Specific pose + prompt | ControlNet-Openpose + SDXL + text |
| Depth-aware composition | ControlNet-Depth + SD3 |
| Reference + prompt | IP-Adapter + text |
| Exact layout | ControlNet-Scribble 或 ControlNet-Canny |
| Background replace | ControlNet-Seg + Inpainting (Lesson 09) |
| Fast 1-step style | LCM-LoRA on SDXL-Turbo |

## Ship It

保存 `outputs/skill-sd-toolkit-composer.md`。Skill 接收一个任务（输入资产：prompt、可选参考图像、可选姿态、可选深度、可选涂鸦）并输出工具栈、权重和可复现的 seed protocol。

## Exercises

1. **Easy。** 在 `code/main.py` 中，将 LoRA rank `r` 从 1 变到 4。在什么 rank 时 LoRA 能精确匹配一个 rank-2 目标增量？
2. **Medium。** 在两种目标变换上训练两个独立的 LoRA。将它们一起加载并展示它们的 additive 交互。交互在什么情况下会打破线性？
3. **Hard。** 使用 diffusers 来堆叠：SDXL-base + Canny-ControlNet (weight 0.8) + 一个 style LoRA (α 0.8) + IP-Adapter (weight 0.6)。测量 FID-vs-prompt-adherence trade-off，随着堆叠权重的变化。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ControlNet | "Spatial control" | 克隆的编码器 + zero-conv skips；读取一个 conditioning image。 |
| Zero convolution | "Starts as identity" | 初始化为零的 1×1 conv；ControlNet 从 no-op 开始。 |
| LoRA | "Low-rank adapter" | `W + B @ A`，`r << d`；比 full fine-tune 少 100 倍参数。 |
| rank r | "The knob" | LoRA 压缩；4-16 典型值，64+ 用于重型个性化。 |
| α | "LoRA strength" | 运行时对 LoRA delta 的缩放。 |
| IP-Adapter | "Reference image" | 通过 CLIP-image tokens 的小型图像 conditioning adapter。 |
| DreamBooth | "Full subject fine-tune" | 在约 30 张主体图像上训练整个模型。 |
| Textual Inversion | "New token" | 只学习一个新词 embedding；legacy，大多被取代。 |

## Production note: LoRA swaps, ControlNet lanes, multi-tenant serving

一个真正的 text-to-image SaaS 在同一 base checkpoint 之上服务数百个 LoRA 和十几个 ControlNet。serving 问题看起来很像 LLM multi-tenancy（生产文献在 continuous batching 和 LoRAX / S-LoRA 下覆盖了 LLM 案例）：

- **Hot-swap LoRAs, do not merge。** 将 `W' = W + α·B·A` 合并进 base 会给每步 inference 带来约 3-5% 的加速，但会冻结 `α` 和 base。将 LoRA 作为 rank-r deltas 热保存在 VRAM 中；diffusers 暴露 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` 用于 per-request activation。Swap 成本是 `2 · d · r · num_layers` 权重 —— MB 级别，亚秒级。
- **ControlNet as a second attention lane。** 克隆的编码器与 base 并行运行。两个 ControlNet 各权重 1.0 = 每步两次额外的 forward pass，而不是一次合并 pass。Batch-size headroom 二次下降。为每个活跃的 ControlNet 预算约 1.5× 步成本。
- **Quantized LoRAs too。** 如果你量化了 base（见 Lesson 07，Flux on 8GB），LoRA delta 也可以干净地量化到 8-bit 或 4-bit。QLoRA-style loading 让你在 4-bit Flux base 之上堆叠 5-10 个 LoRA 而不会爆内存。

Flux-specific: Niels 的 Flux-on-8GB notebook 将 base 量化到 4-bit；在该量化 base 之上堆叠一个 style LoRA (`pipe.load_lora_weights("user/style-lora")`)，以 `weight_name="pytorch_lora_weights.safetensors"` 仍然有效。这是 2026 年大多数 SaaS agency 部署的配方。

## Further Reading

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初用于 LLM；移植到 diffusion）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet 的更轻替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 参考管线。
