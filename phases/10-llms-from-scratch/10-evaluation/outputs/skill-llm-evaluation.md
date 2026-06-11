---
name: skill-llm-evaluation
description: 根据任务类型、预算和要求选择正确 LLM 评估策略的决策框架
version: 1.0.0
phase: 10
lesson: 10
tags: [evaluation, evals, benchmarks, llm-as-judge, elo, metrics]
---

# LLM 评估策略

在评估 LLM 系统时，应用此决策框架选择正确的方法。

## 何时使用每种评估类型

**Benchmarks (MMLU, HumanEval, SWE-bench):** 你正在进行初始模型选择。你需要将 10 个候选模型缩小到 3 个。Benchmark 以零成本提供粗略排名。不要将 benchmark 作为你的最终评估。

**Custom evals:** 你正在为生产构建。你有一个具有特定失败模式的特定任务。Custom evals 是唯一预测真实世界性能的评估。原型最少 50 个测试用例，生产 200+。

**LLM-as-judge:** 你的任务是开放式的（摘要、写作、对话）。Exact match 和 token overlap 指标太僵化。LLM-as-judge 每次判断花费约 $0.01，与人类约 80% 的时间一致。始终使用评分标准，而不是模糊的 prompt。

**Human evals:** 风险很高且自动化指标不一致。Human eval 是 ground truth，但花费 $0.10-$2.00 每次判断。保留给模糊案例和自动化指标的定期校准。

**来自成对比较的 ELO:** 你正在比较同一任务上的多个模型。成对比较比绝对评分更可靠，因为人类（和 LLM judge）更擅长相对判断。

## 评分函数选择

- **Exact match**: 分类、实体提取、具有已知答案的结构化输出
- **Token F1**: 部分信用重要的提取任务
- **ROUGE-L**: 摘要、翻译
- **BLEU**: 机器翻译
- **LLM-as-judge**: 开放式生成、对话质量、helpfulness
- **基于执行**: 代码生成（运行代码，检查测试是否通过）
- **Schema compliance**: 结构化输出（JSON 是否符合 schema？）

## 评估设计中的红旗

- 评估集小于 50 个用例：结果在统计上毫无意义
- 没有边缘情况：你正在测量快乐路径性能，这总是高于真实世界
- 单一指标：不同指标讲述不同故事，至少使用两个
- 没有版本控制：没有版本化的评估集就无法跟踪改进
- 评估集污染：绝不在 fine-tuning 数据或 few-shot prompt 中包含评估示例
- 只测试一个模型：你需要一个 baseline（即使是简单的启发式方法）进行比较

## 评估管道检查清单

1. 精确定义任务（不是 "answer questions" 而是 "classify support tickets into 5 categories"）
2. 跨快乐路径、边缘情况和已知回归创建测试用例
3. 选择 2-3 个适合任务类型的评分函数
4. 基于生产要求设置通过/失败阈值
5. 自动化执行：一个命令运行完整套件
6. 版本化一切：测试用例、评分函数、prompt、模型版本
7. 每次更改都运行：prompt 更新、模型交换、代码部署
8. 跟踪趋势：单一分数是噪声，趋势线是信号
9. 每季度针对人类判断进行校准
10. 每当发现生产故障时添加回归用例
