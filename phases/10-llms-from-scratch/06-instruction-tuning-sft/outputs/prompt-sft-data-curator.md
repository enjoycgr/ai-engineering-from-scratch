---
name: prompt-sft-data-curator
description: 为监督微调设计和策划指令数据集
version: 1.0.0
phase: 10
lesson: 6
tags: [sft, instruction-tuning, fine-tuning, data-curation, alignment]
---

# SFT 数据策划师

在为特定能力（代码生成、数学、对话、安全）设计指令微调数据集时，使用此框架来规划数据收集、定义质量标准并构建训练管道。

## 输入要求

提供：
- **目标能力**（例如，"Python 代码生成"、"医疗问答"、"多轮对话"）
- **基础模型**（例如，Llama 3 8B、Mistral 7B、Qwen 2.5 72B）
- **预算**（标注工时、用于合成生成的 API 成本）
- **格式偏好**（Alpaca、ShareGPT、ChatML）

## Step 1: 数据集设计

### 规模指南

| 质量级别 | 所需示例数 | 预期结果 |
|--------------|----------------|------------------|
| 研究原型 | 1,000-5,000 | LIMA 质量：如果示例是专家编写的，可与更大的数据集媲美 |
| 生产 v1 | 10,000-50,000 | Stanford Alpaca 级别：跨常见任务的扎实指令遵循 |
| 生产 v2 | 50,000-200,000 | Vicuna/Llama 2 Chat 级别：稳健的多轮、领域覆盖 |

质量总是胜过数量。1,000 个专家编写的示例（LIMA，2023 年 5 月）匹配了在 50,000+ 示例上训练的模型。优先考虑：

1. **多样性**——覆盖目标能力的全部范围
2. **准确性**——每个响应必须在事实上正确
3. **清晰性**——响应应简洁且结构良好
4. **难度梯度**——包含简单、中等和困难的示例

### 多样性检查清单

对于通用助手：
- 开放式问题 (20%)
- 事实问答 (20%)
- 创意写作 (10%)
- 代码生成 (15%)
- 推理和数学 (15%)
- 摘要 (10%)
- 带约束的指令遵循 (10%)

针对特定领域的模型调整百分比。编程助手可能会将 60% 分配给代码生成，20% 分配给代码解释。

## Step 2: 数据格式

### Alpaca 格式（单轮）

```json
{
  "instruction": "Write a function that reverses a string in Python.",
  "input": "",
  "output": "def reverse_string(s):\n    return s[::-1]"
}
```

使用场景：单轮任务、简单指令-响应对、快速原型设计。

### ShareGPT 格式（多轮）

```json
{
  "conversations": [
    {"from": "system", "value": "You are a Python expert."},
    {"from": "human", "value": "How do I reverse a string?"},
    {"from": "gpt", "value": "Use slicing: s[::-1]"},
    {"from": "human", "value": "What about for a list?"},
    {"from": "gpt", "value": "Same syntax works: my_list[::-1]"}
  ]
}
```

使用场景：对话应用、多轮上下文很重要的情况。

### ChatML 格式（带特殊 token）

```
<|im_start|>system
You are a Python expert.<|im_end|>
<|im_start|>user
How do I reverse a string?<|im_end|>
<|im_start|>assistant
Use slicing: s[::-1]<|im_end|>
```

使用场景：针对原生使用 ChatML 的模型（Qwen、Yi）。

## Step 3: 质量标准

### 逐条检查

1. **响应相关性**：响应是否真正回答了指令？
2. **事实准确性**：所有声明是否可验证且正确？
3. **完整性**：响应是否完全解决了指令？
4. **简洁性**：是否可以用更少的词传达相同的信息？
5. **格式一致性**：响应是否遵循预期的风格？

### 红旗（拒绝该示例）

- 响应自相矛盾
- 响应包含有害内容而没有拒绝
- 响应 hallucinate 事实或引用
- 指令含糊不清且响应没有澄清
- 响应是指令的改写副本

### 数据集级别检查

- 来自任何单一来源/模板的示例不超过 5%
- 至少 80% 的响应 token 是有意义的（不是填充词）
- 平均响应长度为 50-200 个 token（避免过短或过长）
- System prompt 多样性：至少包含 10 个不同的 system prompt

## Step 4: 训练配置

| 参数 | 推荐范围 | 说明 |
|-----------|------------------|-------|
| Learning rate | 1e-5 到 5e-5 | 较大的模型使用较低的学习率（70B 用 1e-5，7B 用 5e-5） |
| Epochs | 1-3 | 监控验证 loss，在首次出现增加迹象时停止 |
| Batch size | 32-128 | 如果 GPU 有限，随梯度累积扩展 |
| Warmup | 0-5% 的步数 | 不如 pre-training 关键 |
| Weight decay | 0.0-0.1 | 短 fine-tuning 运行可选 |
| Loss masking | 仅响应 token | Mask 指令和 system prompt token |
| Pre-training data mixing | 2-5% | 混合原始文本以防止 catastrophic forgetting |

## Step 5: 评估协议

训练后，评估：

1. **指令遵循率**：模型产生相关、完整响应的测试 prompt 百分比
2. **遗忘分数**：与基础模型相比，在保留的通用文本语料库上的 perplexity
3. **格式合规性**：遵循预期聊天格式的响应百分比
4. **MT-Bench 或 AlpacaEval**：指令微调模型的标准 benchmark
5. **领域特定评估**：针对目标能力的自定义评估

### 警告信号

- 验证 loss 在 epoch 1 后增加：你在 overfitting，减少 epoch 或增加数据
- 遗忘分数增加 > 15%：学习率太高或 epoch 太多
- 模型逐字复现训练示例：严重 overfitting，需要更多样化的数据
- 模型拒绝良性指令：在安全数据上训练过度，重新平衡数据集
