# Multi-Token Prediction (MTP)（多 token 预测）

> 从 GPT-2 到 Llama 3 的每个自回归 LLM 都在每个位置训练一个损失：预测下一个 token。DeepSeek-V3 为每个位置添加了第二个损失：预测再下一个 token。额外的 14B 参数（在 671B 模型上）通过梯度流蒸馏回主模型，训练好的 MTP head 在推理时被重新用作 speculative-decoding（推测解码）的 draft model（草稿模型），接受率 80%+。生成吞吐量免费提升了 1.8 倍。本课从 DeepSeek 技术报告构建顺序 MTP 模块，计算损失和共享 head 的参数布局，并解释为什么 MTP 保持因果链，而 Gloeckle 等人的原始并行 MTP 打破了它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Learning Objectives

- 说明 MTP 训练目标并推导跨预测深度的联合损失。
- 解释 Gloeckle 等人的并行 MTP head（2024）与 DeepSeek-V3 的顺序 MTP 模块之间的区别，以及为什么顺序设计保持因果链。
- 计算在预训练运行中添加 MTP 模块的参数和内存开销。
- 从头实现一个 MTP 模块：共享 embedding、逐深度 transformer block、投影和共享输出 head。

## The Problem

Next-token prediction（下一 token 预测）是标准 LLM 训练目标。每个隐藏状态被监督以精确预测一件事：紧随其后的 token。这是一个令人惊讶的弱信号。序列中的大部分信息延伸到一个 token 之外——结构、连贯性、事实性、算术流。模型必须通过数万亿 token 累积许多单 token 信号来学习这些。

MTP 问：如果每个隐藏状态被监督以同时预测多个未来 token 会怎样？Gloeckle 等人（Meta，2024）表明这有帮助。他们的实现在主干上放置了几个独立的输出 head，每个预测不同的偏移。并行、简单，但 head 看到相同的隐藏状态而没有任何层次细化——并且预测不因果链接，因此不能用于 speculative decoding。

DeepSeek-V3（2024 年 12 月）将 MTP 重新设计为在每个预测深度保持因果链的顺序模块。模型从 `h_i^(0)` 预测 `t+1`，然后从新的隐藏状态 `h_i^(1)` 预测 `t+2`，该状态结合了 `h_i^(0)` 与 `E(t+1)` embedding，依此类推。每个深度是它自己的小型 transformer block。共享 embedding 和共享输出 head 保持参数开销适度。在 DeepSeek-V3 的规模下，671B 主模型权重之上额外 14B 参数。这 2% 的开销买到了更密集的训练信号 AND 推理时现成的 speculative-decoding draft。

本课从头构建单个 MTP 模块和 D 深度损失。数学很简洁。实现是 150 行。

## The Concept

### 顺序 MTP 配方

DeepSeek-V3 在主模型之上添加 `D` 个 MTP 模块。每个模块 `k`（对于 `k = 1..D`）预测深度 `k` 的 token——即给定前缀到位置 `i` 的 `t_{i+k}`。

模块 `k` 包含：

- 一个具有自己 attention 和 MLP 的 transformer block `T_k`。
- 一个投影矩阵 `M_k`，结合前一深度的隐藏状态与下一深度 ground-truth token 的 embedding。
- 共享 embedding `E`（与主模型相同）。
- 共享输出 head `Out`（与主模型相同）。

训练时，对于前缀到位置 `i`，逐深度隐藏状态为：

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

逐深度预测为：

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

逐深度损失是对 ground-truth `t_{i+k}` 的交叉熵：

```
L_k = CE(logits_{i+k}, t_{i+k})
```

跨深度的联合损失：

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` 是一个小的加权因子——DeepSeek-V3 在前 10% 训练中使用 0.3，之后使用 0.1。总训练损失是 `L_main + L_MTP`。

### 为什么是顺序而非并行

Gloeckle 的原始并行 MTP 有 D 个输出 head，每个直接应用于 `h_i^(0)`。每个 head 从相同的 backbone 隐藏状态预测 `t_{i+k}`。这训练得很好，但预测不相互条件化。你不能使用 `head_1` 的输出帮助 `head_2`——head 并行触发。

DeepSeek-V3 的顺序设计从 `h_i^(k-1)` 加上实际下一 token embedding `E(t_{i+k})` 构建 `h_i^(k)`。这保持了因果链：要预测 `t_{i+k+1}`，深度 `k+1` 的模块看到 `t_{i+k}` 的内容。这在结构上与自回归解码器消耗自己输出的方式相同——使 MTP 模块可直接用作 speculative-decoding 的 draft model。

推理时：将 `h_i^(k-1)` 和 draft 的 `t_{i+k}` 输入模块 `k+1`，获得 `t_{i+k+1}` 的预测。重复。这正是一个 EAGLE 风格的 draft，使用训练的 MTP 模块作为 draft network。DeepSeek-V3 报告第一个 MTP 模块 80%+ 接受率和约 1.8 倍加速。

### 参数核算

对于隐藏层 `h` 和词表 `V` 的模型：

- 主模型：数十亿参数，加上一个大小为 `V * h` 的输出 head。
- 共享输出 head：重用主模型的 head。无额外参数。
- 共享 embedding：重用主模型的 embedding。无额外参数。
- 每个 MTP 模块：
  - 投影 `M_k`：`(2h) * h = 2h^2`。
  - Transformer block `T_k`：attention (`4h^2` for MHA) 加 MLP（SwiGLU 比例 8/3 时通常为 `8h^2`）。每个 block 约 `12h^2`。

每个模块额外总计：`~14h^2`。对于 DeepSeek-V3 的 `h = 7168`，D = 1 个模块：`~14 * 7168^2 = ~720M` 纸面参数。DeepSeek-V3 报告 14B——差异主要来自 MTP 模块中的 expert 层也是 MoE。

### Speculative-decoding 回报

预训练期间，MTP 模块使训练慢约 10%（更多前向计算，额外损失）。回报是双重的：

1. 更密集的训练信号。每个隐藏状态看到 D+1 个监督目标。在 MMLU、GSM8K、MATH、HumanEval 上的测量效果：DeepSeek-V3 的消融中一致提升几个百分点。

2. 推理时免费的 speculative decoding draft。MTP 模块已经训练好预测接下来的几个 token。重新用作 draft network 时，它提供 80%+ 接受率。在该水平下，N=3 或 N=5 的 spec decoding 给出 1.8 倍吞吐量。10% 的训练时间成本在第一次运行推理时就回本。

### 与 EAGLE 的关系

EAGLE 在预训练后单独训练一个小型 draft model。MTP 将 draft 烘焙到预训练中。两种方法在相似的接受率上收敛，但通过不同的流水线：

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible with existing weights | Yes | No (need to re-train) |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | Speculative decoding only | Denser training signal + speedup |

## Build It

`code/main.py` 端到端构建单个 MTP 模块：共享 embedding、投影、transformer block、共享输出 head。然后计算短合成序列上的逐深度交叉熵损失并打印按组件的参数计数。32 个 token 的玩具词表保持数字可读。

### Step 1: 共享 embedding 表

单个 `vocab_size x hidden` 表被主模型 AND 每个深度的每个 MTP 模块使用。不是第二份拷贝——字面意义上是相同的张量。

### Step 2: 逐深度组合

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真正的 DeepSeek-V3 将两个 RMSNormed 向量拼接成 `[2h]` 并用 `h x 2h` 矩阵投影。玩具使用向量加法以简化标准库。

### Step 3: 深度 k 的 transformer block

Self-attention 加 MLP。在玩具中，单层线性 attention block 和 SwiGLU MLP 保持结构可见而无需 numpy。

### Step 4: 共享输出 head

重用主模型的输出投影。词表上的 logits。

### Step 5: 逐深度损失

Softmax(logits) 对偏移 `k` 处 ground-truth token 的交叉熵。用 `lambda / D` 缩放因子聚合跨深度。

### Step 6: 参数核算

打印总参数计数、共享（embedding、head）计数和每个模块额外计数。展示 MTP 额外与主模型大小的比率。

## Use It

MTP 集成到 DeepSeek-V3（2024 年 12 月）和 DeepSeek-R1 系列中。推理时：

- DeepSeek 自己的服务栈开箱即用地将 MTP 模块作为 speculative decoder 消费。
- 截至 2026 年 4 月，vLLM 和 SGLang 有 DeepSeek-V3 MTP 的集成路径。
- AMD 的 ROCm SGLang 教程展示了一个特定的 MTP speculative-decoding 配置，在 V3 检查点上测量到 1.8 倍加速。

在新预训练运行中何时使用 MTP：

- 你控制完整的预训练流水线并希望储备更密集的训练信号。
- 你知道你将大规模服务该模型并希望免费的 speculative decoding。
- 你的隐藏大小至少为 4096。在 1B 规模下开销伤害大于收益帮助。

何时不使用：

- 对现有预训练密集模型进行 fine-tuning。MTP 模块未训练。
- 你想要干净基线进行比较的研究模型。MTP 改变了架构。

## Ship It

本课产出 `outputs/skill-mtp-planner.md`。给定预训练运行规范（模型大小、数据、计算），它返回 MTP 集成计划：深度数量 D、`lambda` 调度、内存开销和推理时 speculative-decoding 接线。

## Exercises

1. 运行 `code/main.py`。展示随着合成信号增强，逐深度损失单调递减。修改合成以使用固定模式并验证深度 1 和深度 2 损失都收敛。

2. 计算密集 70B 模型（隐藏 8192，80 层）带 D=1 MTP 模块的参数开销。与 DeepSeek-V3 报告的 14B 开销比较。解释为什么 DeepSeek 的数字更高：MTP transformer block 继承了相同的 MoE 结构，膨胀了每个模块的参数计数。

3. 在玩具中实现 D=2：添加第二个 MTP 模块，接收 h^(1) 并预测 `t_{i+2}`。验证联合损失和参数核算与 DeepSeek 论文的公式 19-21 匹配。

4. 将玩具切换到并行 MTP（Gloeckle 风格）：在主隐藏状态上添加 D 个输出 head，每个预测不同偏移。在相同合成信号上测量逐深度损失与顺序版本的比较。顺序版本应为 k > 1 产生更低的深度 k 损失，因为它条件化于中间预测。

5. 在推理时将训练的 MTP 模块用作 EAGLE 风格的 draft：调用模块 k 以在推理时提议 `t_{i+k}`。测量这些 draft token 对主模型在留出序列上预测的接受率。如果你在玩具上达到 50%+，你就复现了 MTP-as-draft 的经验特性。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MTP module | "Extra loss block" | 一个小型 transformer block 加投影，从主模型预测 `k` 个位置后的 token |
| Prediction depth | "Which offset" | 整数 `k` 使得模块 `k` 从前缀到位置 `i` 预测 `t_{i+k}` |
| Parallel MTP | "Gloeckle-style" | D 个独立 head 在相同 backbone 隐藏状态上，无条件链 |
| Sequential MTP | "DeepSeek-V3 style" | 每个模块条件化于前一深度的隐藏状态加下一 token 的 embedding；保持因果链 |
| Shared output head | "Reuse the main head" | MTP 模块调用主模型的 LM head，不是单独的输出投影 |
| Shared embedding | "Reuse the main table" | 相同的词表 embedding 表在各处使用；无重复参数 |
| Projection matrix M_k | "Combine hidden + next-token" | 将前一隐藏状态和目标 token embedding 折叠到下一深度输入的 `h x 2h` 线性层 |
| Joint loss L_MTP | "Averaged extra losses" | 逐深度交叉熵损失的算术平均，按 `lambda` 缩放 |
| Acceptance rate at depth 1 | "How often MTP draft is right" | D=1 MTP 模块的 top-1 预测等于主模型 top-1 预测的比率；DeepSeek-V3 上 80%+ |
| Lambda weighting | "Extra-loss importance" | 逐深度缩放因子；DeepSeek-V3 训练开始时 0.3，之后 0.1 |

## Further Reading

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — 完整顺序 MTP 描述（Section 2.2），包括联合损失方程和推理时 1.8 倍加速
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — DeepSeek 设计改进的并行 MTP 基线
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 685B 总计（671B 主模型 + 14B MTP），部署说明
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — MTP 适配的 speculative-decoding 框架
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — EAGLE 的 2025 draft 架构，MTP 竞争的对照组
