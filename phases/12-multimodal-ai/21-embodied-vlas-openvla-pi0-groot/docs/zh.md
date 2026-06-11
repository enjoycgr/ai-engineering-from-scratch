# 具身 VLA：RT-2、OpenVLA、π0、GR00T

> 模型首次从网站读取食谱并在厨房机器人中执行是 RT-2 (Google DeepMind, 2023 年 7 月)。RT-2 将动作离散化为文本 token，在网页数据加机器人动作数据上共微调 VLM，并证明网页规模视觉-语言知识迁移到机器人控制。OpenVLA (2024 年 6 月) 发货了开放 7B 参考。Physical Intelligence 的 π0 系列 (2024-2025) 添加了 flow-matching 动作专家。NVIDIA 的 GR00T N1 (2025 年 3 月) 为规模上的人形机器人交付了双系统（System 1 / System 2）控制。VLA 原语——vision-language-action，一个能看、读和动的单一模型——是本阶段理解模型与 Phase 15 自主系统之间的桥梁。

**类型：** Learn
**语言：** Python (stdlib, action tokenizer + VLA inference skeleton)
**前置知识：** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**时间：** ~180 分钟

## 学习目标

- 描述动作 tokenization：离散 bin 编码 (RT-2)、FAST 高效动作 token、连续 flow-matching 动作 (π0)。
- 解释为什么在网页 + 机器人数据上共微调保留对新任务的通用知识迁移。
- 在同一机器人任务上对比 OpenVLA (开放 7B Llama+VLM)、π0 (flow-matching) 和 GR00T N1 (双系统)。
- 说出 Open X-Embodiment 数据集及其作为 RT-X 训练语料库的角色。

## 问题

做自然语言指令家务的机器人自 1970 年代以来就是研究目标。2020 年代答案：vision-language-action (VLA) 模型。与 VQA 相同的 VLM 架构，但输出是动作（关节扭矩、末端执行器姿态、离散命令）而非文本。

VLA 特有挑战：

1. 动作空间是连续的（关节角度、力）且高维（7-DOF 臂 + 3-DOF 夹爪 = 30 Hz 下 10 维）。
2. 机器人特定训练数据稀缺。Open X-Embodiment 有约 100 万轨迹；网页文本-图像是 50 亿+。
3. 控制频率重要。30 Hz 控制循环意味着每动作 33ms 预算。
4. 安全。错误动作损坏硬件、人类或财产。

## 概念

### 动作 tokenization (RT-2)

RT-2 的技巧：将每个关节目标表示为量化文本 token。将归一化 [-1, 1] 范围离散化为 256 bin，每个 bin 映射到词汇 ID。10-DOF 动作每控制步变成 10 个 token。

在混合物上共微调 PaLM-X VLM：

- 网页图像-文本对（标题生成、VQA）。
- 机器人演示，动作作为 token。

模型看到"捡起红方块"（语言）→ 图像（视觉）→ 10-token 动作序列（离散化关节目标）。网页预训练保留通用知识迁移：RT-2 能跟随"移向快速移动物体"即使"快速移动"不在训练数据中。

RT-2 论文中推理在 3-5 Hz，受 VLM 自回归解码限制。

### OpenVLA——开放 7B 参考

OpenVLA (Kim 等人, 2024 年 6 月) 是开放权重 RT-2 等价物。7B Llama 主干、DINOv2 + SigLIP 双视觉编码器、256 bin 上的动作 tokenization。

在 Open X-Embodiment（跨 22 个机器人的 97 万轨迹）上训练。附带 LoRA 微调支持以适应新机器人。

推理：A100 上量化后 4-5 Hz。对慢速操作足够快，对高频控制不够。

### FAST tokenizer——更快动作解码

Pertsch 等人 (2024) 表明离散 bin tokenization 效率低——大多数动作聚集在 bin-space 的小区域。FAST (Frequency-domain Action Sequence Tokenizer) 通过 DCT 压缩动作序列并量化系数。

30 步动作轨迹变成 ~10 个 FAST token 而非 300 个离散 bin token。推理加速 3-5 倍而无质量损失。

### π0 和 flow-matching 动作

Physical Intelligence 的 π0 (Black 等人, 2024 年 10 月) 用 flow-matching 动作专家替换离散动作 token：

- 小动作 transformer 读取 VLM 的 hidden state 并通过 rectified flow 输出连续 50 步动作序列。
- 动作 head 用 flow-matching loss 训练；VLM 预训练保持不变。
- 推理：完整动作序列在 ~5 去噪步中发出，有效 50 Hz 控制。

π0 声明：在广泛操作任务套件上击败 OpenVLA 和 Octo。连续动作公式保留离散化破坏的平滑性。

π0.5 和 π0-FAST 是增量升级。π0-FAST 结合 FAST tokenization 与 flow matching。

### GR00T N1——人形双系统

NVIDIA 的 GR00T N1 (2025 年 3 月) 为类人机器人构建（>30 DOF，全身）：

- System 2：大型 VLM 读取场景 + 指令，以 ~1 Hz 产生高级子目标。
- System 1：小动作 head transformer 产生条件于子目标的低级 50-100 Hz 关节命令。

拆分映射到 Kahneman 的快与慢思考：System 2 规划，System 1 执行。好处：慢 VLM 规模规划不阻塞快速控制；System 1 保持小以维持延迟。

GR00T N1.7 (2025 年末) 改进数据扩展。GR00T 用 Omniverse 的 sim-to-real 数据微调。

### Open X-Embodiment

训练数据。RT-X (2023 年 10 月) 组装了跨 22 个数据集的 22 个机器人 100 万轨迹。Open X-Embodiment 是每个人使用的语料库：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每样本：（机器人状态、摄像头视图、指令、动作序列）。
- 训练卫生：统一动作空间、归一化关节范围、resize 摄像头。

OpenVLA 和 π0 在 Open X-Embodiment 上训练。到任何特定机器人的领域差距通过 100-1000 任务特定演示的 LoRA 微调关闭。

### 共微调 vs 仅机器人

共微调混合网页 VQA 数据与机器人轨迹。比例重要：太多 VQA 模型遗忘动作；太多机器人数据模型丢失通用知识。

RT-2 比例：~1:1。OpenVLA：~0.5:1 网页到机器人。π0：类似。精确比例是按数据集大小调优的超参数。

仅机器人训练产生任务特定模型，在分布外指令上失败。共微调是"捡起红方块（在演示中）"与"捡起从左数第三大的物体（新颖措辞）"之间的差异。

### 安全和动作限制

每个生产 VLA 附带：

- 硬关节限制（不能超过规格扭矩）。
- 速度限制（软裁剪）。
- 工作空间边界（末端执行器不能离开桌子）。
- 新颖任务的人工在环批准。

这些坐在 VLA 外部作为控制层检查。VLA 的输出是建议，不是命令。

## 使用它

`code/main.py`：

- 实现 256-bin 动作 tokenization 和 de-tokenization。
- 勾勒基于 DCT + 量化的 FAST tokenizer。
- 对比每动作步 token 数（离散 bin、FAST、连续 flow）。
- 打印 RT-2 → OpenVLA → π0 → GR00T 的谱系摘要。

## 交付它

本课产生 `outputs/skill-vla-action-format-picker.md`。给定机器人任务（操作、导航、类人全身），在离散 bin + RT-2、FAST + OpenVLA、flow-matching + π0 或双系统 + GR00T 之间挑选。

## 练习

1. 10-DOF 臂在 30 Hz 控制率。256 bin 离散化每秒发出多少 token？7B VLM 能跟上吗？

2. FAST tokenization 将 30 步轨迹压缩到 ~10 token。如果轨迹有高频运动（如击鼓）用户丢失什么？

3. π0 的 flow-matching head 在 ~5 步去噪。对比 OpenVLA 自回归解码在 4-5 Hz 的吞吐。

4. GR00T 的 System 1 / System 2 拆分映射到 Kahneman。提出可能帮助双足行走的不同拆分（System 3？）。

5. 阅读 Open X-Embodiment Section 4 关于数据集整理。说出防止领域泄露的三个整理规则。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| VLA | "Vision-language-action" | 接收图像 + 指令并输出动作命令的模型 |
| Action tokenization | "Discrete bins" | 将连续关节目标量化为每维 256 bin，每个 bin 一个词汇 ID |
| FAST tokenizer | "Frequency action tokens" | DCT + 量化将 30 步轨迹压缩到 ~10 token |
| Co-fine-tune | "Mix web + robot" | 在网页 VQA 数据与机器人演示上训练以保留通用知识 |
| Flow-matching action head | "π0 continuous output" | 小动作 transformer 通过 rectified flow 输出 50 步动作序列 |
| System 1 / System 2 | "Dual-system control" | 大型 VLM 慢速规划，小动作 head 快速执行；GR00T 模式 |
| Open X-Embodiment | "RT-X dataset" | 100 万轨迹跨机器人数据集；训练语料库 |

## 延伸阅读

- [Brohan 等人 — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim 等人 — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black 等人 — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
