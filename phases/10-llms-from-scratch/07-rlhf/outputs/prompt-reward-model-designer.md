---
name: prompt-reward-model-designer
description: 为 RLHF 对齐设计 reward model 训练管道
version: 1.0.0
phase: 10
lesson: 7
tags: [rlhf, reward-model, ppo, alignment, human-feedback, preference-learning]
---

# Reward Model 设计师

在构建 RLHF 管道以将语言模型对齐到目标行为（helpfulness、编码能力、安全、诚实）时，使用此框架来设计数据收集协议、训练 reward model 和配置 PPO。

## 输入要求

提供：
- **目标行为**（例如，"helpful and harmless assistant"、"expert Python coder"、"medical Q&A with safety"）
- **基础模型**（例如，SFT 后的 Llama 3 8B、Mistral 7B Chat）
- **Reward model 大小**（通常与 policy model 相同大小或更大）
- **标注预算**（人工工时或可用的比较对）
- **计算预算**（reward model 训练 + PPO 的 GPU 小时数）

## Step 1: 偏好数据收集

### 标注协议

1. **Prompt 选择**：从 SFT 训练分布中采样，加上分布外 prompt（10-20% 新颖）
2. **响应生成**：使用不同 temperature（0.3、0.7、1.0）的 SFT 模型为每个 prompt 生成 2-4 个响应
3. **比较格式**：向标注员精确展示 2 个响应并问 "Which response is better?"
4. **标准评分标准**：为你的用例定义 "better" 的含义

### 评分标准模板

| 标准 | 权重 | 描述 |
|-----------|--------|-------------|
| Helpfulness | 40% | 它是否完整且正确地回答了问题？ |
| Harmlessness | 25% | 它是否避免有害、偏见或误导性内容？ |
| Honesty | 20% | 它是否承认不确定性而不是 hallucinate？ |
| Conciseness | 15% | 响应长度是否适合问题？ |

针对你的用例调整权重。编程助手可能将正确性权重设为 60%，简洁性设为 20%。

### 数据规模指南

| 规模 | 比较对 | 标注员工时 | 预期 RM 准确率 |
|-------|-----------------|-----------------|---------------------|
| 最小可行 | 5,000-10,000 | 400-800 | 60-65% |
| 生产 v1 | 20,000-50,000 | 1,600-4,000 | 65-72% |
| 生产 v2 | 100,000-500,000 | 8,000-40,000 | 72-78% |

InstructGPT 使用了来自 40 名承包商的 33,000 次比较。Anthropic 的初始论文使用了来自 20 名标注员的 22,000 次。标注员间一致性通常为 70-75%——reward model 不能超过人类一致性水平。

### 质量控制

- **一致性过滤**：丢弃少于 70% 标注员同意的对
- **标注员校准**：在真实标注前用已知良好的对进行校准轮次
- **偏见检测**：监控标注员是否始终偏好更长的响应、正式语言或特定模式
- **对抗性示例**：包含 5-10% 旨在捕捉未仔细阅读的标注员的示例

## Step 2: Reward Model 架构

### 架构决策

| 决策 | 推荐 | 理由 |
|----------|---------------|-----------|
| 基础架构 | 与 policy 相同的 transformer | 从 SFT checkpoint 进行权重初始化提供强大的起始特征 |
| 输出头 | 从最后隐藏状态的单个线性投影 | 来自最完整位置表示的标量 reward |
| 模型大小 | >= policy model 大小 | 较小的 RM 产生不稳定的信号，使 PPO 不稳定 |
| 初始化 | 带新输出头的 SFT checkpoint | 预训练特征已经捕捉了语言质量 |

### 训练配置

| 参数 | 范围 | 说明 |
|-----------|-------|-------|
| Learning rate | 1e-5 到 5e-5 | 低于 SFT，因为任务更简单 |
| Epochs | 1-3 | 有限的比较数据下 overfitting 是主要风险 |
| Batch size | 64-256 | 每个 "示例" 是一对，因此有效数据是 2x |
| Loss function | Bradley-Terry: -log(sigmoid(r_preferred - r_rejected)) | 成对比较的标准 |
| Validation split | 10-20% | 监控保留对上的准确率 |

### 评估指标

1. **成对准确率**：RM 正确排序的保留偏好对占多少比例？目标：> 65%
2. **Margin 分布**：绘制 (r_preferred - r_rejected) 的分布。应集中在 0 以上，很少有负数。
3. **校准**：sigmoid(r_preferred - r_rejected) 是否接近实际的人类偏好概率？
4. **OOD 泛化**：在不同于训练分布的 prompt 上测试。准确率下降应 < 10%。

## Step 3: PPO 配置

### 超参数

| 参数 | 典型值 | 过高时的影响 | 过低时的影响 |
|-----------|--------------|-------------------------|------------------------|
| KL coefficient (beta) | 0.01-0.05 | 模型几乎不学习，过于接近 SFT | Reward hacking，退化输出 |
| Learning rate | 5e-6 到 3e-5 | 训练不稳定，发散 | 收敛慢，浪费计算 |
| Clip ratio (epsilon) | 0.1-0.3 | 大且可能破坏稳定的更新 | 非常保守的更新，学习慢 |
| PPO epochs per batch | 1-4 | 对当前 batch overfitting | 未充分利用每个 batch |
| Generation batch size | 128-512 | 内存问题 | 嘈杂的梯度估计 |
| Max response length | 256-1024 | 生成慢，内存问题 | 截断有用的响应 |

### 监控仪表板

在 PPO 训练期间跟踪这些指标：

1. **Mean reward**：应在训练过程中增加。平台期没问题；下降意味着不稳定。
2. **KL divergence**：应保持在 10-20 nats 以下。峰值 = reward hacking。
3. **Response length**：应保持稳定。单调增加 = 冗长 reward hacking。
4. **Entropy**：Token 分布熵应缓慢下降。快速下降 = mode collapse。
5. **Reward model agreement**：用 reward model 对 PPO 响应评分；一致性应提高。

### PPO 期间的红旗

| 症状 | 可能原因 | 修复 |
|---------|-------------|-----|
| Reward 增加但输出退化 | Reward hacking | 增加 KL coefficient，在对抗性示例上重新训练 RM |
| KL divergence 爆炸 | 学习率太高或 KL coefficient 太低 | 降低 lr，增加 beta |
| Response length 单调增长 | RM 奖励冗长 | 向 reward 添加长度惩罚，用长度受控的对重新训练 RM |
| 所有响应变得相同 | Mode collapse | 增加生成 temperature，减少 PPO epoch |
| Reward 剧烈振荡 | PPO 不稳定 | 降低学习率，增加 clip ratio |

## Step 4: 端到端验证

在部署 RLHF 训练的模型之前：

1. **A/B 测试 vs SFT**：在 200+ 测试 prompt 上运行 SFT 和 RLHF 模型。让 3+ 评估者比较响应。RLHF 模型应赢得 > 60% 的时间。
2. **安全评估**：在已知的对抗性 prompt（jailbreaks、有害请求）上测试。RLHF 模型应适当拒绝。
3. **回归检查**：运行标准 benchmark（MMLU、HumanEval、MT-Bench）以确认 RLHF 模型没有失去核心能力。
4. **遗忘检查**：测量通用文本语料库上的 perplexity。与 SFT 模型相比，增加应 < 10%。
5. **长度分析**：比较 SFT 和 RLHF 模型之间的平均响应长度。如果 RLHF 长 > 50%，reward model 可能有冗长偏见。
