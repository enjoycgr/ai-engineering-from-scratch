# Scaling Laws（扩展定律）

> 2020 年 Kaplan 论文指出：模型越大，loss 越低。2022 年 Hoffmann 论文指出：你们训练不足。计算量（Compute）分为两个桶——参数量（parameters）和 token 量（tokens）——而两者的分配并非显而易见。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer），Phase 7 · 07（GPT）
**时间：** 约 45 分钟

## 问题背景

当你拥有 C FLOPs 的训练计算量，并希望获得最佳模型时，你面临两个旋钮：

1. **多少参数（N）？** 模型越大，容量越高。
2. **多少训练 token（D）？** 数据越多，容量利用越充分。

FLOPs 大致按 `6 × N × D` 缩放。你可以把 N 推高、D 压低，或者把 D 推高、N 压低。哪种更好？

2022 年之前，答案是"猛推 N"。GPT-3（2020）有 175B 参数，训练了约 300B token。比例约为每个参数 1.7 个 token。Kaplan 扩展定律支持这一点。

Hoffmann 等人（2022）训练了一个名为 Chinchilla 的小型模型族，发现了不同的结果：最优比例接近 **每个参数 20 个 token**。GPT-3 的训练量少了 10 倍。Chinchilla（70B 参数，1.4T token）在每个基准测试上都击败了 GPT-3（175B，300B token），而推理成本只有 2.5 分之一。

2026 年是 Chinchilla 的世界——但有一个重要的转折。Llama 3 8B 训练了 15 万亿 token，比例为每个参数 1,875 个 token。比 Chinchilla 最优值高出 94 倍。对于大规模使用的模型，推理成本比训练成本更重要，因此过度训练（超过 Chinchilla 最优值）以获得更小的可部署模型是 2026 年的默认做法。

## 核心概念

![Chinchilla 曲线：不同 N/D 比例下 loss 与计算量的关系](../assets/scaling-laws.svg)

### Hoffmann 定律

根据 Chinchilla 论文，loss 遵循：

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = 参数数量（非嵌入层）。
- `D` = 训练 token 数量。
- `α ≈ 0.34`，`β ≈ 0.28`（大致对称）。
- `E ≈ 1.69`，不可约 loss 上限。
- `A ≈ 406`，`B ≈ 411`。

两项在扩展时相互权衡。在固定计算量（C = 6ND）下对 `N` 求导并求解：

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

计算最优：每个参数 20 个 token。

### 为什么要过度训练

Chinchilla 最优值最小化每训练 FLOP 的训练 loss。但训练成本只付一次；推理成本永远持续。

对于每月服务一万亿 token 的聊天机器人，推理 dominates 总成本。Llama 的做法：训练更小的模型，更长时间。8B 参数训练 15T token 是深度推理优化的：

- 可放入消费级 GPU。
- 延迟是 70B Chinchilla 最优模型的一小部分。
- 质量对大多数任务来说足够接近。

DeepMind 2024 年的论文（"Over-training is the new optimal"）将这一点形式化。对于推理主导的工作负载，正确的比例更接近每个参数 100–500 个 token，具体取决于服务量。

### 涌现 vs 平滑性

观点：某些能力（算术、多步推理、思维链跟随）在某个规模下会"突然涌现"。

Schaeffer 等人（2023）认为这是一种测量伪影：涌现指标使用不连续的评分（精确匹配、阈值准确率），掩盖了底层 logits 的平滑改进。连续指标（交叉熵）显示平滑曲线。

2026 年的共识是：通过连续 loss 进行预测是可靠的。基准测试的跳跃通常是评分器伪影。按连续指标规划预算。

### 2026 年的图景

扩展定律仍然有效，但：

| 因素 | 变化方式 |
|------|---------|
| 数据质量 | 策划"好"token（Phi 风格）将曲线偏移 >2× 有效计算量 |
| MoE（Mixture of Experts，专家混合） | 总参数与活跃 FLOP 解耦；按活跃 FLOP 的扩展定律 |
| 后训练（Post-training） | 某些能力（指令跟随、代码）随 SFT+RLHF 的偏移大于预训练 |
| 多模态（Multimodality） | 图像 + 文本 token 一起扩展；每种模态有单独的曲线 |
| 合成数据（Synthetic data） | 模型生成训练数据；有效计算量可以复合 |

Muon 优化器（Kimi Moonlight，2024）在匹配数据下展示了约 2× 有效计算量增益超过 AdamW。一些 2026 年的训练运行默认使用 Muon。改变了扩展定律中的绝对常数，而非其形状。

## 动手实现

参见 `code/main.py`。我们实现了 Chinchilla loss 方程，并在多个计算预算下求解计算最优的 `(N, D)`。

### 步骤 1：Chinchilla loss

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 `C = 6ND` 下，将 `L` 绘制成 `(N, D)` 的等高线图。找到最小值。

### 步骤 2：计算最优前沿

对于从 `1e17` 到 `1e25` FLOPs 的计算预算，找到在约束 `6ND = C` 下最小化 loss 的 `(N, D)`。验证比例 `D/N ≈ 20`。

### 步骤 3：过度训练成本

计算训练 10 倍小模型（最优 N 的 1/10，最优 D 的 10 倍）所额外付出的 loss。报告换取的推理 FLOP 节省（与 N 成正比）。

### 步骤 4：与真实模型对比

放入已知的 `(N, D)` 对：GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（活跃参数），并比较预测 loss 与报告 loss。

## 如何应用

你不太可能自己训练前沿模型。但扩展定律告诉你：

1. **你的微调是否有足够数据。** 如果你的任务特定数据低于基础模型每个参数 20 个 token，预计会在某个 loss 下限处饱和。
2. **是否应该选择更大的基础模型。** 如果你将所有预算花在推理上，选择更小、训练时间更长的模型。
3. **收益递减点在哪里。** 超过 1000× Chinchilla 最优值后，log-loss 的变化变成噪声。

**2026 年的研究轨迹：**

- **数据受限 regime。** 网络上有有限数量的高质量 token（过滤后约 5–10 万亿英语 token）。前沿预训练正在接近这个上限。合成数据、多语言、多模态和 RLHF 扩展微调是下一个杠杆。
- **计算量倍增技巧。** Muon 优化器、MoE、更好的数据策划——每个都偏移绝对常数，而非渐近线。
- **RL 的扩展定律。** 开放问题。早期证据表明 RL 样本的幂律，但指数与预训练非常不同。

## 交付

参见 `outputs/skill-training-budget-estimator.md`。该 skill 根据计算预算、部署约束和目标 loss，为新训练运行选择 `(N, D, hours, GPU)`。

## 练习

1. **简单。** 运行 `code/main.py`。打印 Chinchilla 最优 `(N, D)` 对于计算预算 `1e20`、`1e22`、`1e24`。与真实模型表对比。
2. **中等。** 实现 Hoffmann loss-as-function-of-compute 曲线。为计算最优前沿绘制 loss vs `log10(C)`。确定定律预测我们需要 `>10^28` FLOPs 才能在下一次交叉熵降低 0.1 的时间点。
3. **困难。** 在 5 个微小模型（100K 到 10M 参数）上拟合你自己的扩展定律，使用相同数据集。估计 `α` 和 `E`。你的指数与已发表的多匹配？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| Parameters（参数）(N) | "模型大小" | 非嵌入层权重数量；决定容量。 |
| Tokens（token）(D) | "训练数据" | 看到的训练 token 数量；决定参数利用程度。 |
| Compute（计算量）(C) | "花费的 FLOPs" | 标准 transformer 大约 `6 × N × D`。 |
| Chinchilla-optimal（Chinchilla 最优） | "D/N ≈ 20" | 最小化每预训练 FLOP 的 loss 的比例。 |
| Over-training（过度训练） | "超过 Chinchilla" | 花费额外训练 FLOPs 以节省推理 FLOPs；D/N >> 20。 |
| Irreducible loss（不可约 loss） | "下限" | 扩展定律中的 `E` 项；数据本身的熵。 |
| Emergent capability（涌现能力） | "规模上的突然跳跃" | 通常是评分器伪影；连续 loss 是平滑的。 |
| Effective compute（有效计算量） | "训练效率倍增器" | 更好的数据/优化器/架构让 FLOP 走得更远。 |

## 延伸阅读

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇扩展定律论文；训练不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 涌现作为测量伪影。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 为什么 Llama 的过度训练对其工作负载是正确的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2× 计算量倍增器。
