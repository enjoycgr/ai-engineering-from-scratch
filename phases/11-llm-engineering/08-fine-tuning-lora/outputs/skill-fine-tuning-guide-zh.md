---
name: skill-fine-tuning-guide
description: 何时以及如何使用 LoRA 和 QLoRA fine-tune LLM 的决策树
version: 1.0.0
phase: 11
lesson: 8
tags: [fine-tuning, lora, qlora, peft, llm-engineering]
---

# Fine-Tuning Decision Guide

在 fine-tuning 之前，按顺序尝试这些方法：

```
1. Prompt engineering (几分钟, $0)
2. Prompt 中的 few-shot examples (几分钟, $0)
3. RAG 用于知识检索 (几天, $10-100/月)
4. 使用 LoRA/QLoRA fine-tuning (几天, $5-50 每次实验)
5. Full fine-tuning (几周, $100-10,000 每次运行)
```

仅当上一个方法可测量地不足时才进入下一步。

## 何时 fine-tune

- 模型需要一种 prompting 无法实现的稳定输出风格或格式
- 你在将更大的模型蒸馏（从 8B 模型获得 GPT-4 质量）
- 延迟很重要，few-shot examples 增加了太多 token
- 你需要模型可靠地遵循复杂推理模式
- 你有 1,000+ 高质量的期望输入-输出行为示例

## 何时不 fine-tune

- 模型在正确的 prompt 下已经能做你想要的事
- 你需要模型知道事实（改用 RAG）
- 你的训练样本少于 500（可能 overfitting (过拟合)）
- 任务频繁变化（重新训练很昂贵）
- 你需要审计哪些数据影响了特定输出（fine-tuning 是黑箱）

## 方法选择

| GPU VRAM | 7B 模型 | 13B 模型 | 70B 模型 |
|----------|---------|----------|----------|
| 16GB (T4) | QLoRA | 不可行 | 不可行 |
| 24GB (3090/4090) | QLoRA 或 LoRA | QLoRA | 不可行 |
| 40GB (A100) | LoRA 或 Full | QLoRA 或 LoRA | QLoRA |
| 80GB (A100/H100) | Full | LoRA 或 Full | QLoRA 或 LoRA |

## LoRA 配置清单

1. 从 r=16, alpha=32 开始（大多数任务的安全默认值）
2. 首先针对 q_proj 和 v_proj（最小可行 LoRA）
3. QLoRA 使用学习率 2e-4，LoRA fp16 使用 5e-5
4. 设置 lora_dropout=0.05
5. 训练 1-3 个 epoch（更多会面临 overfitting 风险）
6. 每 100 步在 held-out 集上评估
7. 保存 checkpoint 并按 eval loss 选择最佳

## 常见错误

- 训练太多 epoch（小数据集在 epoch 2-3 后 overfitting）
- 使用与 full fine-tuning 相同的学习率（LoRA 需要更高 LR）
- 忘记设置 pad token（导致 Llama 模型出现 NaN loss）
- 没有冻结 base model（违背了 LoRA 的目的）
- 仅在训练数据上评估（始终保留 10-20% 用于 eval）
- 跳过 prompt engineering baseline（fine-tuning 一个 prompting 已经能解决的问题）

## 质量验证

训练后，在 200+ held-out 样本上对比：
1. 最佳 prompt 的 base model（baseline）
2. 带 LoRA adapter 的 base model（你的 fine-tuned 模型）
3. 相同 prompt 的 GPT-4 或 Claude（天花板）

如果 LoRA 模型没有击败 prompted baseline，你的训练数据或配置需要改进，而不是更多算力。

## Adapter 管理

- 为多任务服务保持 adapter 分离（每个请求交换 adapter）
- 为单任务部署将 adapter 合并到 base 权重中
- 在 Hugging Face Hub 上存储 adapter（10-100MB，易于版本控制和共享）
- 部署前测试合并后的模型输出与未合并的输出匹配
- 使用 TIES-Merging 或 DARE 将多个 adapter 合并为一个

## 调试训练

如果 loss 不下降：
1. 检查学习率（LoRA 太低，尝试 2e-4）
2. 验证 LoRA 层确实在接收梯度
3. 确认 base model 权重已冻结
4. 检查数据格式（tokenizer 必须匹配模型的预期格式）

如果 loss 下降但 eval 质量差：
1. 训练数据质量问题（垃圾进，垃圾出）
2. Overfitting（减少 epoch，增加 dropout，添加更多数据）
3. 错误的 target modules（为复杂任务添加 MLP 层）
4. Rank 太低（尝试 r=32 或 r=64）
