---
name: prompt-alignment-method-selector
description: 为你的用例选择正确的对齐方法（SFT、RLHF、DPO、KTO、ORPO、SimPO）
version: 1.0.0
phase: 10
lesson: 8
tags: [alignment, dpo, rlhf, kto, orpo, simpo, preference-optimization, fine-tuning]
---

# 对齐方法选择器

在为语言模型选择对齐方法时，使用此框架评估你的数据、计算和质量要求，然后选择最符合你约束的方法。

## 输入要求

提供：
- **基础模型**（例如，Llama 3 8B、Mistral 7B、Qwen 2.5 72B）
- **起始点**（基础模型，还是已经 SFT 的？）
- **可用数据**（指令对、偏好对、未配对评分，或无）
- **计算预算**（GPU 小时数、GPU 数量）
- **质量目标**（原型够用、与开源竞争、state-of-the-art）
- **时间线**（天、周、月）

## 决策矩阵

### 快速选择

| 你的情况 | 推荐方法 | 原因 |
|---------------|-------------------|-----|
| 没有偏好数据，只有指令对 | 仅 SFT | 没有偏好信号就无法对齐 |
| < 5,000 偏好对，计算有限 | DPO | 管道更简单，小数据下效果好 |
| 未配对反馈（仅 thumbs up/down） | KTO | 唯一无需成对比较的方法 |
| 希望在单个训练运行中对齐 | ORPO | 结合 SFT + 对齐，无需 reference model |
| 内存受限（无法容纳 reference model） | SimPO | 无需 reference model |
| 大规模、多目标对齐 | RLHF (PPO) | 单独的 reward model 捕捉复杂偏好 |
| 迭代对齐与在线数据 | RLHF (PPO) | 可以生成、评分并循环重新训练 |
| Post-RLHF 细化 | DPO | 在针对性偏好上微调 RLHF 模型 |

### 详细比较

| 方法 | 数据要求 | 内存中的模型 | 训练循环 | 稳定性 | 最佳规模 |
|--------|-----------------|-----------------|----------------|-----------|------------|
| SFT | 指令对 (10K+) | 1 | 1 | 高 | 任何 |
| RLHF | 偏好对 (20K+) | 3-4 | 3 | 低 | 大 (70B+) |
| DPO | 偏好对 (5K+) | 2 | 2 (SFT + DPO) | 高 | 小-中 (7B-70B) |
| KTO | 未配对评分 (5K+) | 2 | 2 (SFT + KTO) | 高 | 任何 |
| ORPO | 偏好对 (10K+) | 1 | 1 | 高 | 小-中 |
| SimPO | 偏好对 (5K+) | 1 | 2 (SFT + SimPO) | 高 | 小-中 |

## 方法特定配置

### SFT

- **何时停止**：1-3 个 epoch 后或验证 loss 停止下降时
- **关键超参数**：Learning rate（1e-5 到 5e-5，更大的模型用更低的学习率）
- **关键细节**：在 loss 中 mask 指令 token
- **陷阱**：超过 3 个 epoch 会导致 memorization；混合 2-5% pre-training 数据

### RLHF (PPO)

- **何时使用**：你有 20K+ 比较对，需要多目标对齐，或想要迭代在线学习
- **关键超参数**：KL coefficient (0.01-0.05)、PPO clip ratio (0.1-0.3)、learning rate (5e-6 到 3e-5)
- **关键细节**：Reward model 应 >= policy model 大小
- **陷阱**：PPO 不稳定；持续监控 KL divergence 和 reward 曲线

### DPO

- **何时使用**：你有偏好对且想要比 RLHF 更简单的管道
- **关键超参数**：Beta (0.1-0.5；越低 = 允许偏离 reference 越多)
- **关键细节**：Reference model 必须是 SFT checkpoint 的冻结副本
- **陷阱**：对 beta 非常敏感；在 [0.05, 0.1, 0.2, 0.5] 上运行 sweep

### KTO

- **何时使用**：你只有 "good" 或 "bad" 标签，没有成对比较
- **关键超参数**：Beta（与 DPO 相同）、bad 响应上的 loss aversion 乘数 (1.5x)
- **关键细节**：需要大致平衡的好/坏示例（40-60% 分割）
- **陷阱**：没有成对数据，梯度信号更弱；可能需要比 DPO 更多的数据

### ORPO

- **何时使用**：你想跳过 SFT 直接从基础模型到对齐模型
- **关键超参数**：Lambda（偏好项 vs SFT 项的权重）
- **关键细节**：需要一个数据集中同时包含指令标签和偏好对
- **陷阱**：组合目标可能难以平衡；如果 SFT loss 占主导，对齐就很弱

### SimPO

- **何时使用**：内存受限的设置，无法容纳 reference model
- **关键超参数**：Beta、gamma（长度归一化指数）
- **关键细节**：长度归一化防止模型偏好短响应
- **陷阱**：没有 reference model 锚点，模型可以漂移更远；仔细监控

## 管道模板

### 模板 1：快速原型（1-2 天）

```
Base Model -> SFT (1 epoch, 10K examples) -> DPO (3 epochs, 5K pairs)
```

计算：7B 模型在 A100 上约 4 GPU 小时
质量：扎实的指令遵循，基本的偏好对齐

### 模板 2：生产质量（1-2 周）

```
Base Model -> SFT (2 epochs, 50K examples) -> DPO (5 epochs, 20K pairs) -> Eval -> Iterate
```

计算：7B 约 40 GPU 小时，70B 约 200 GPU 小时
质量：与开源 RLHF 模型竞争

### 模板 3：State-of-the-Art（1-3 个月）

```
Base Model -> SFT (2 epochs, 100K+ examples) -> RLHF (PPO, 50K+ pairs) -> DPO (targeted refinement) -> Eval -> Iterate
```

计算：70B 约 500+ GPU 小时
质量：接近前沿模型对齐

### 模板 4：最小数据（1-2 天）

```
Base Model -> SFT (1 epoch, 5K examples) -> KTO (unpaired thumbs up/down from users)
```

计算：7B 约 2 GPU 小时
质量：比仅 SFT 更好，数据收集开销最小

## 评估协议

对齐后，在这些维度上评估：

1. **偏好胜率**：在 200+ 测试 prompt 上比较对齐模型与 SFT 模型，由人类评判。目标：> 60% 胜率。
2. **Benchmark 保留**：MMLU、HumanEval 或领域特定 benchmark。与 SFT baseline 相比不应下降 > 5%。
3. **MT-Bench 或 AlpacaEval**：标准对齐质量 benchmark。与已发布的 baseline 比较。
4. **安全评估**：针对对抗性 prompt、jailbreaks 和有害请求类别进行测试。
5. **响应多样性**：测量 100 个 prompt 上响应的熵。低熵 = mode collapse。

## 常见失败模式

| 症状 | 原因 | 方法特定修复 |
|---------|-------|-------------------|
| 冗长、填充的响应 | Reward model / 隐式 reward 偏好长度 | DPO：增加 beta。RLHF：添加长度惩罚。SimPO：调整 gamma。 |
| 模型同意一切 | 偏好数据偏见导致的 sycophancy | 添加正确响应不同意用户的偏好对 |
| 拒绝良性请求 | 安全数据上过度对齐 | 减少安全示例比例，添加更多良性-拒绝对 |
| 输出与 SFT 几乎相同 | Beta 太高（DPO/KTO）或 KL coefficient 太高（PPO） | 降低 beta / KL coefficient；模型没有学习 |
| 训练 loss 振荡 | 学习率太高或数据不足 | 将 lr 降低 2-3 倍；增加偏好数据 |
