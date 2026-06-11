---
name: 3d-pipeline
description: 根据输入类型、输出格式和使用场景选择 3D 生成或重建 pipeline。
version: 1.0.0
phase: 8
lesson: 12
tags: [3d, gaussian-splatting, nerf, mesh]
---

给定输入（文本提示词 / 单张图像 / 少量图像 / 照片捕获 / 视频）、目标输出（mesh / Gaussian splat / NeRF / point cloud）和使用场景（实时渲染、游戏引擎、AR / VR、影视），输出：

1. Pipeline. (a) 多视图扩散 (multi-view diffusion) + 3D 拟合（SV3D、CAT3D + 3DGS），(b) 直接单次推理（LRM、TripoSR、InstantMesh），(c) 带 PBR 的文本到 mesh（Meshy 4、Rodin Gen-1.5、Hunyuan3D 2.0），(d) 照片捕获 + 3DGS（Gsplat、Postshot、Scaniverse）。
2. Base model + hosting. 指定模型 + 开源 / 托管。包含商业用途的许可证相关性。
3. Iteration budget. 首次输出的预计时间、迭代成本、细化策略。
4. Topology + materials. 是否需要 remesh 流程？PBR 通道要求（albedo、roughness、metallic、normal）？UV 布局是自动还是手动？
5. Eval. 在保留视图上的 SSIM、CLIP score、mesh 水密性、多边形数量、纹理分辨率。
6. Platform target. Unity / Unreal / Blender / web（three.js / Babylon）/ AR（USDZ / glb）。

拒绝将 3DGS 直接交付到游戏引擎而不经过 mesh 转换流程（大多数引擎不支持原生 splat 渲染）。拒绝为复杂关节角色使用 text-to-3D —— 改用支持骨骼绑定的 pipeline。当下游工具无法渲染 NeRF 时（大多数 DCC 工具），标记任何仅 NeRF 的输出。
