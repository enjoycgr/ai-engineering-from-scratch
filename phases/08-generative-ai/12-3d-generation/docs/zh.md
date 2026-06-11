# 3D 生成 (3D Generation)

> 3D 是 2D-to-3D 杠杆效应最强的模态。2023 年的突破是 3D Gaussian Splatting (3D 高斯泼溅)。2024–2026 年的生成式推进则是在其之上叠加多视图扩散 (multi-view diffusion) + 3D 重建，从而从单个提示词或照片生成物体与场景。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 分钟

## The Problem

3D 内容令人头疼：

- **Representation.** mesh (网格)、point cloud (点云)、voxel grid (体素网格)、signed distance fields / SDF (有符号距离场)、neural radiance fields / NeRF (神经辐射场)、3D Gaussians。每种都有各自的权衡。
- **Data scarcity.** ImageNet 有 1400 万张图像。最大的干净 3D 数据集 (Objaverse-XL, 2023) 约有 1000 万个物体，且大多数质量低下。
- **Memory.** 一个 $512^3$ 的体素网格有 1.28 亿个体素；一个可用场景的 NeRF 需要每条光线采样 100 万次。生成比重建更难。
- **Supervision.** 对于 2D 图像，你拥有像素本身。对于 3D，你通常只有少量 2D 视图，必须将其提升到 3D。

2026 年的技术栈将两个问题拆分开。首先，用 diffusion model (扩散模型) 生成 *2D 多视图图像*。其次，将 *3D 表示*（通常是 Gaussian splatting）拟合到这些图像上。

## The Concept

![3D 生成：多视图扩散 + 3D 重建](../assets/3d-generation.svg)

### Representation: 3D Gaussian Splatting (Kerbl et al., 2023)

将场景表示为约 100 万个 3D 高斯 (3D Gaussians) 的集合。每个高斯有 59 个参数：位置 (3)、协方差 (6，或四元数 4 + 尺度 3)、不透明度 (1)、球谐函数颜色 (degree 3 时 48 个，degree 0 时 3 个)。

渲染 = 投影 + alpha 合成。速度快（在 4090 上 1080p 约 100 fps）。可微分。通过梯度下降拟合真实照片。一个场景在消费级 GPU 上只需 5–30 分钟即可完成拟合。

2023–2024 年的两项创新：
- **Generative Gaussian splats.** LGM、LRM、InstantMesh 等模型直接从一个或少数图像预测高斯集合。
- **4D Gaussian Splatting.** 具有每帧偏移量的高斯集合，用于动态场景。

### Multi-view diffusion

微调预训练的图像扩散模型，使其从文本提示词或单张图像生成同一物体的多个一致视图。Zero123 (Liu et al., 2023)、MVDream (Shi et al., 2023)、SV3D (Stability, 2024)、CAT3D (Google, 2024)。通常输出围绕物体的 4–16 个视图，然后通过 Gaussian splatting 或 NeRF 提升到 3D。

### Text-to-3D pipelines

| Model | Input | Output | Time |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 hour per asset |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane (三平面) | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

2025–2026 年方向：直接生成带 PBR (基于物理的渲染) 材质的 text-to-mesh 模型，可直接用于游戏引擎。对于通用物体，多视图扩散中间步骤仍然是表现最佳的方案。

### NeRF (for context)

Neural Radiance Field (Mildenhall et al., 2020)。一个微型 MLP 接收 `(x, y, z, view direction)` 并输出 `(color, density)`。通过沿光线积分进行渲染。在新视角合成质量上优于基于 mesh 的方法，但渲染速度慢 100–1000 倍。对于大多数实时用途已被 Gaussian splatting 取代，但在研究中仍占主导地位。

## Build It

`code/main.py` 实现了一个玩具级的 2D "Gaussian splatting" 拟合：将合成目标图像（一个平滑渐变）表示为 2D Gaussian splats 的总和。通过梯度下降优化位置、颜色和协方差以匹配目标。你会看到两个核心操作：前向渲染（splat + alpha 合成）和通过梯度下降进行拟合。

### Step 1: 2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Step 2: 渲染时对所有 splats 求和

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

真实的 3D Gaussian Splatting 会按深度对高斯排序并进行 alpha 合成。我们的 2D 玩具只是简单求和。

### Step 3: 通过梯度下降进行拟合

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Pitfalls

- **View inconsistency.** 如果你独立生成 4 个视图，而它们在物体结构上存在分歧，那么 3D 拟合结果会很模糊。解决方法：使用共享注意力机制的多视图扩散 (multi-view diffusion)。
- **Back-side hallucination.** 单张图像 → 3D 必须凭空想象看不见的那一面。质量差异巨大。
- **Gaussian splat explosion.** 无约束的训练会增长到 1000 万个 splats 并产生过拟合 (overfitting)。来自 3D-GS 原始论文的致密化 (densification) + 剪枝启发式规则至关重要。
- **Topology issues.** 来自隐式场 (SDFs) 的 mesh 通常会有孔洞或自相交。在交付前运行 remesher（例如 Blender 的 voxel remesh）。
- **License of training data.** Objaverse 的许可证混杂；商业用途因模型而异。

## Use It

| Task | 2026 pick |
|------|-----------|
| 从照片进行场景重建 | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| 游戏用文本到 3D 物体 | Meshy 4 或 Rodin Gen-1.5 (PBR 输出) |
| 图像到 3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 从少量图像进行新视角合成 | CAT3D, SV3D |
| 动态场景重建 | 4D Gaussian Splatting |
| 虚拟形象 / 着装人体 | Gaussian Avatar, HUGS |
| 研究 / SOTA | 上周刚发布的新模型 |

对于在游戏或电商流程中交付生产级 3D：Meshy 4 或 Rodin Gen-1.5 输出可直接导入 Unity / Unreal 的 PBR mesh。

## Ship It

保存 `outputs/skill-3d-pipeline.md`。该 skill 接收一份 3D 简报（输入：文本 / 单张图像 / 少量图像；输出：mesh / splat / NeRF；用途：渲染 / 游戏 / VR），并输出：pipeline（多视图扩散 + 拟合，或直接 mesh 模型）、基础模型、迭代预算、拓扑后处理、所需材质通道。

## Exercises

1. **Easy.** 用 4、16、64 个 Gaussians 运行 `code/main.py`。报告最终 MSE 与目标的对比。
2. **Medium.** 扩展到彩色 Gaussians（RGB）。确认重建结果与目标颜色模式匹配。
3. **Hard.** 使用 gsplat 或 Nerfstudio，从 50 张照片捕获中重建真实物体。报告拟合时间和在保留视图上的最终 SSIM。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | 将场景表示为 3D 高斯集合；可微分的 alpha 合成渲染。 |
| NeRF | "Neural radiance field" | 输出 3D 点处颜色 + 密度的 MLP；通过光线积分渲染。 |
| Triplane | "Three 2-D planes" | 将 3D 分解为三个轴对齐的 2D 特征平面；比体渲染 (volumetric rendering) 更廉价。 |
| SDS | "Score distillation sampling" | 使用 2D 扩散分数 (score) 作为伪梯度来训练 3D 模型。 |
| Multi-view diffusion | "Many views at once" | 输出一批一致相机视图的扩散模型。 |
| PBR | "Physically-based rendering" | 具有 albedo、roughness、metallic、normal 通道的材质。 |
| Densification | "Grow splats" | 3DGS 训练启发式规则：在高梯度区域分裂 / 克隆 splats。 |

## Production note: 3D has no shared substrate yet

与图像（latent diffusion + DiT）和视频（时空 DiT）不同，3D 在 2026 年还没有单一的 dominant runtime。生产决策树因表示形式而分叉：

- **NeRF / triplane.** 推理是光线步进 (ray-marching) + 每个样本一次 MLP 前向传播。$512^2$ 渲染需要数百万次 MLP 前向传播。积极批量处理光线样本；可应用 SDPA/xformers。
- **Multi-view diffusion + LRM reconstruction.** 两阶段 pipeline。第一阶段（多视图 DiT）与 Lesson 07 中的扩散服务器完全相同。第二阶段（LRM transformer）是对视图的 one-shot 前向传播。整体延迟特征为 "diffusion + one-shot" —— 按阶段选择 serving 原语。
- **SDS / DreamFusion.** 按资产优化，而非推理。构建的是离线作业，而非请求处理器。

对于大多数 2026 年的产品，正确答案是在请求时运行多视图扩散模型，异步重建到 3DGS，然后为实时查看提供 3DGS 服务。这清晰地分割了 GPU 推理服务器（快）和离线优化器（慢）之间的工作负载。

## Further Reading

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF。
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS。
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS。
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123。
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion。
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM。
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D。
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D。
