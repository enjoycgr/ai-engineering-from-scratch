# 边缘推理 —— Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> 边缘设备的核心约束是内存带宽，而非算力。移动端 DRAM 为 50-90 GB/s；数据中心 HBM3 达到 2-3 TB/s —— 差距为 30-50 倍。decode 是内存带宽受限的，因此这一差距是决定性的。2026 年的格局分为四路。Apple M4/A18 Neural Engine 峰值 38 TOPS，采用统一内存（无 CPU↔NPU 拷贝）。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 达到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以 Q4 运行 Llama 3.1 8B 约 41 tok/s（约为原生的 70-80%）；GitHub 17.6k star，OpenAI 兼容 API，移动端覆盖率约 70-75%。NVIDIA Jetson Orin Nano Super（8GB）可容纳 Llama 3.2 3B / Phi-3；AGX Orin 通过 vLLM 运行 gpt-oss-20b 约 40 tok/s；Jetson T4000（JetPack 7.1）性能为 AGX Orin 的 2 倍。TensorRT Edge-LLM 支持 EAGLE-3、NVFP4、chunked prefill —— 由 Bosch、ThunderSoft、MediaTek 在 CES 2026 展示。

**类型:** 学习
**语言:** Python（标准库，玩具带宽受限 decode 模拟器）
**前置知识:** Phase 17 · 04（vLLM 服务内部原理），Phase 17 · 09（生产量化）
**时间:** 约 60 分钟

## 学习目标

- 解释为什么移动端 LLM 推理受内存带宽限制，算力是次要的。
- 列举四个边缘目标（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson）并将每个匹配到一个用例。
- 说出 2026 年 WebGPU 覆盖缺口（Firefox Android 正在追赶）和 Safari iOS 26 落地。
- 为每个目标选择量化格式（ANE 用 Core ML INT4 + FP16，Hexagon 用 QNN INT8/INT4，浏览器用 WebGPU Q4，Jetson Thor 用 NVFP4）。

## 问题背景

客户想要一个设备端聊天机器人：语音优先、默认私有、离线工作。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 约 55 tok/s —— 可以。在 iPhone 16 Pro 上，同一模型约 3 tok/s —— 不行。在中端 Android Snapdragon 8 Gen 3 上，7 tok/s。在 Chrome Android v121+ 的浏览器 WebGPU 上，4-8 tok/s，取决于设备。

吞吐量差异不是移植问题。它是带宽差距乘以量化格式乘以 NPU 是否可从用户空间访问。2026 年的边缘推理是四个不同的问题，有四个不同的解决方案。

## 核心概念

### 带宽是真正的天花板

decode 为每个 token 读取完整权重集。一个 7B 模型 Q4 为 3.5 GB。以 50 GB/s 读取 3.5 GB 需要 70 ms —— 理论上限约 14 tok/s。在 90 GB/s（高端移动 DRAM）时，上限提升到约 25 tok/s。再多的算力也无法突破这个数字。

数据中心 HBM3 以 3 TB/s 在 1.2 ms 内读取同样的 3.5 GB —— 上限为 830 tok/s。同一模型，同一权重。不同的内存子系统。

### Apple Neural Engine（M4 / A18）

- 最高 38 TOPS。统一内存（CPU 和 ANE 共享同一池）—— 无拷贝开销。
- 通过 Core ML + `.mlmodel` 编译模型访问，或通过 PyTorch 的 Metal Performance Shaders（MPS）。
- Llama.cpp Metal 后端使用 MPS，而非直接访问 ANE；原生 ANE 需要 Core ML 转换。
- 2026 年 iOS 应用的最佳实践路径：Core ML，INT4 权重 + FP16 激活。

### Qualcomm Hexagon（Snapdragon X Elite / 8 Gen 4）

- 最高 45 TOPS。与 SoC 中的 CPU 和 GPU 集成，但独立内存域。
- QNN（Qualcomm Neural Network）SDK 和 AI Hub 提供从 PyTorch/ONNX 的转换。
- Chat templates、Llama 3.2、Phi-3 都在 AI Hub 上作为一等制品提供。

### Intel / AMD NPU（Lunar Lake、Ryzen AI 300）

- 40-50 TOPS。软件落后于 Apple/Qualcomm；OpenVINO 正在改进但属小众。
- 最适合 Windows ARM copilot 应用；AMD/Intel 桌面上的本地优先原生方案。

### WebGPU + WebLLM

- 通过 WebGPU 计算着色器在浏览器中运行模型；无需安装。
- M3 Max 上 Llama 3.1 8B Q4 约 41 tok/s —— 通过同一后端约为原生的 70-80%。
- WebLLM GitHub 17.6k star；OpenAI 兼容 JS API；Apache 2.0。
- 2026 年覆盖：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追赶。整体移动端覆盖率约 70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super（8GB）：容纳 Llama 3.2 3B、Phi-3，tok/s 良好。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- Thor / T4000（JetPack 7.1）：AGX Orin 性能的 2 倍，支持 EAGLE-3 和 NVFP4。
- TensorRT Edge-LLM（2026）支持 EAGLE-3 投机解码、NVFP4 权重、chunked prefill —— 数据中心优化移植到边缘。

### 每个目标的量化选择

| 目标 | 格式 | 说明 |
|------|------|------|
| Apple ANE | INT4 权重 + FP16 激活 | Core ML 转换路径 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 转换器 |
| WebGPU / WebLLM | Q4 MLC（q4f16_1） | 使用 `mlc_llm convert_weight` + 编译 `.wasm`；不支持 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | 内存受限 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路径 |

### 边缘上的长上下文陷阱

Llama 3.1 的 128K 上下文是数据中心功能。在 8 GB RAM 的手机上，4 GB 模型 + 32K token 的 2 GB KV cache + 系统开销 = OOM。边缘部署将上下文保持在 4K-8K，除非接受激进的 KV 量化（Q4 KV）。

### 语音是杀手级应用

语音智能体对延迟敏感（首 token < 500 ms）。本地推理完全消除了网络延迟。结合语音转文本（Whisper Turbo 变体可在边缘运行），边缘推理成为生产级质量语音循环。

### 你应该记住的数字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 约 41 tok/s。
- AGX Orin：通过 vLLM 运行 gpt-oss-20b 约 40 tok/s。
- 数据中心-边缘带宽差距：30-50 倍。
- WebGPU 移动端覆盖：约 70-75%（Firefox Android 落后）。

## 动手实践

`code/main.py` 从带宽受限数学计算跨边缘目标的理论 decode 吞吐量上限。与实测基准比较，并突出带宽（而非算力）是瓶颈的地方。

## 交付成果

本课产出 `outputs/skill-edge-target-picker.md`。给定平台（iOS/Android/浏览器/Jetson）、模型和延迟/内存预算，选择量化格式和转换流水线。

## 练习

1. 运行 `code/main.py`。对于 Snapdragon 8 Gen 3（~77 GB/s 带宽）上的 7B Q4 模型，计算 decode 上限。与实测 6-8 tok/s 比较 —— 运行时是否高效？
2. Android 上的 WebGPU 需要 Chrome v121+。为旧浏览器设计回退方案 —— 通过同一 OpenAI 兼容 API 的服务器端。
3. 你的 iOS 应用需要 4K 上下文流式。哪种模型/格式组合让你在 iPhone 16 上保持 4 GB 活跃内存以下？
4. Jetson AGX Orin 以 40 tok/s 运行 gpt-oss-20b。Jetson Nano 只能容纳 3B。如果你的产品同时针对两者，如何统一推理栈？
5. 论证"WebLLM 在 2026 年是否生产就绪"。引用覆盖率、性能和 Firefox Android 缺口。

## 关键术语

| 术语 | 通常说法 | 实际含义 |
|------|---------|---------|
| ANE | "Apple neural engine" | M 系列和 A 系列中的设备端 NPU；统一内存 |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU；通过 QNN SDK 访问 |
| WebGPU | "browser GPU" | W3C 标准化浏览器 GPU API；Chrome/Safari 2026 |
| WebLLM | "browser LLM runtime" | MLC-LLM 项目；Apache 2.0；OpenAI 兼容 JS |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | "edge TensorRT" | 2026 年 TensorRT-LLM 的边缘移植；EAGLE-3 + NVFP4 |
| 统一内存 | "shared pool" | CPU 和 NPU 看到同一 RAM；无拷贝开销 |
| 带宽受限 | "memory limited" | decode 受读取权重的字节/秒限制 |
| Core ML | "Apple conversion" | Apple 用于 ANE 原生模型的框架 |
| QNN | "Qualcomm stack" | Qualcomm Neural Network SDK |

## 延伸阅读

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — 格局和基准。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 边缘移植公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 设计和基准。
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 原生转换。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon 预转换模型。
