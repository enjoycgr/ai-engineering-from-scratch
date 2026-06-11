# 投机解码（Speculative Decoding）—— 起草、验证、重复

> 自回归解码（autoregressive decoding）是串行的：每个 token 必须等待前一个生成完毕。投机解码打破了这一链条：用一个廉价模型起草 N 个 token，再用昂贵模型在一次前向传播中全部验证。当起草正确时，你只用一次大模型前向就换来了 N 个生成 token。

**类型：** 构建（Build）
**语言：** Python
**前置知识：** Phase 7 · 07（GPT Causal LM），Phase 7 · 12（KV Cache & Flash Attention）
**时间：** 约 60 分钟

## 问题背景

一个 70B 参数的 LLM 在 H100 上采样一个 token 约需 30 ms。一个 3B 参数的起草模型（draft model）只需约 3 ms。如果我们让 3B 模型先起草 5 个 token，然后让 70B 模型**只运行一次**来验证这 5 个 token，总耗时为 `5×3 + 30 = 45 ms`，最多可接受 5 个 token —— 而直接串行生成需要 `5×30 = 150 ms`。这就是投机解码的全部核心：用少量额外的 GPU 显存（起草模型）换取 2–4 倍的解码延迟降低。

这个技巧必须保留目标分布。Leviathan 等人（2023）以及 Chen 等人同期提出的投机采样（speculative sampling）保证：输出序列与大型模型独立采样时的分布**完全一致**。没有质量损失，只是更快。

2026 年的推理场景中，四类起草-验证组合占据主导：

1. **经典投机解码（Vanilla speculative，Leviathan 2023）。** 独立的起草模型（如 Llama 3 1B）+ 验证模型（如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** 在验证模型上增加多个解码头（decoding heads），并行预测位置 `t+1..t+k` 的 token。无需独立起草模型。
3. **EAGLE 系列（Li 2024, 2025）。** 轻量起草模型复用验证模型的隐藏状态（hidden states）；接受率比经典方法更高；典型加速 3–4 倍。
4. **Lookahead 解码（Fu 2024）。** 使用 Jacobi 迭代；完全不需要起草模型。自投机（self-speculation）方式，适用面较窄但无额外依赖。

2026 年的每一个生产级推理栈都默认内置了投机解码。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少支持经典投机 + EAGLE-2。

## 核心概念

### 核心算法

给定验证模型 `M_q` 和更廉价的起草模型 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **起草（Draft）**：用 `M_p` 自回归地提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，对应起草概率为 `p_1..p_N`。
3. **并行验证（Verify in parallel）**：将 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 输入 `M_q` 运行一次，得到验证概率 `q_1..q_{N+1}`，对应位置 `k+1..k+N+1`。
4. **从左到右逐个接受/拒绝**：对每个位置 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 接受该起草 token。
5. 在首个拒绝位置 `j`：从"残差分布（residual distribution）" `(q_j - p_j)_+` 归一化后采样得到 `t_j`。位置 `j` 之后的所有起草 token 全部丢弃。
6. 若全部 `N` 个 token 都被接受：从 `q_{N+1}` 多采样一个额外 token `t_{N+1}`（免费的奖励 token）。

残差分布技巧是保持输出分布与 `M_q` 从头采样完全一致的数学关键。

### 什么决定了加速比

设 `α` = 每个起草 token 的期望接受率。设 `c` = 起草模型与验证模型的成本比。每步：

- 朴素生成：每个 token 调用一次大模型。
- 投机解码：当 `α` 较高时，每步约调用一次大模型可生成 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个 token。

经验法则：`α = 0.75`、`N = 5` 时，大模型调用次数减少约 3 倍。起草成本是 5 倍廉价。总 wall-clock 时间降低约 2.5 倍。

**α 取决于：**

- 起草模型对验证模型的近似程度。同一家族 / 同一训练数据能显著提升 α。
- 解码策略。贪心起草对贪心验证：α 高。温度采样（temperature sampling）：更难匹配；接受率下降。
- 任务类型。代码和结构化输出接受率更高（可预测）；自由创意写作接受率更低。

### Medusa —— 无需起草模型的投机解码

Medusa 用验证模型上的额外输出头（output heads）取代独立起草模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (标准 LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头输出自己的 logits。推理时从每个头采样得到候选序列，然后用 tree-attention 方案在一次前向传播中验证所有候选延续。

优点：无需第二个模型。缺点：增加了可训练参数；需要监督微调阶段（约 1B token）；接受率略低于搭配优质起草模型的经典投机解码。

### EAGLE —— 通过复用隐藏状态获得更好的起草模型

EAGLE-1/2/3（Li 等，2024–2025）将起草模型做成一个极小的 transformer（通常 1 层），输入是验证模型最后一层的隐藏状态（hidden states）。由于起草模型能看到验证模型的特征表示，其预测与验证模型的输出分布高度相关。接受率从经典方法的约 0.6 提升到 0.85+。

EAGLE-3（2025）增加了对候选延续的树搜索。vLLM 和 SGLang 将 EAGLE-2/3 作为 Llama 3/4 和 Qwen 3 的默认投机路径。

### KV Cache 的来回操作

验证时将 `N` 个起草 token 一次性输入验证模型。这会将验证模型的 KV cache 扩展 `N` 个条目。如果部分起草 token 被拒绝，必须将 cache 回滚到已接受前缀的长度。

生产实现（vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）使用临时 KV 缓冲区处理：先写入，接受后再提交。概念上不复杂，但实现上比较繁琐。

## 动手实现

见 `code/main.py`。我们实现了核心投机采样算法（拒绝步骤 + 残差分布），包括：

- 一个"大模型"，使用确定性 softmax 作用于手工设定的分布（以便解析验证接受率的数学正确性）。
- 一个"起草模型"，是大模型的扰动版本。
- 一个接受/拒绝循环，产生的边缘分布与直接采样一致。

### 步骤 1：拒绝步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均匀随机数。`q_prob` 是验证模型对该起草 token 的概率。`p_prob` 是起草模型的概率。Leviathan 定理指出：这个 Bernoulli 决策，加上拒绝时从残差分布采样，能精确保持验证模型的分布。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素用 `q` 减 `p`，负值截断为零，再重新归一化。任何拒绝时从此分布采样。

### 步骤 3：一次投机步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五个全接受 → 一个奖励 token → 一次验证调用产出六个 token。

### 步骤 4：测量接受率

运行 10,000 次投机步骤，在不同起草质量水平下。绘制接受率与起草和验证分布之间 KL 散度（KL divergence）的关系。应呈现清晰的单调关系。

### 步骤 5：验证分布等价性

经验验证：投机循环产生的 token 直方图应与直接从验证模型采样产生的直方图一致。这就是 Leviathan 定理的实践体现。卡方检验（chi-square test）确认在采样误差范围内一致。

## 实际使用

生产环境：

```bash
# vLLM 配合 EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM 配合经典起草模型
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至 2026 年中，TensorRT-LLM 拥有最快的 Medusa 路径。`faster-whisper` 为 Whisper-large 包装了投机解码，使用一个小型起草模型。

**选择起草策略：**

| 策略 | 何时选择 | 加速比 |
|------|----------|--------|
| 经典起草（1B/3B Llama 家族） | 快速原型，无需训练 | 1.8–2.3× |
| Medusa heads | 你可以微调验证模型 | 2–3× |
| EAGLE-2 / 3 | 生产环境，追求最大速度 | 3–4× |
| Lookahead | 无需起草模型、无需训练、无额外参数 | 1.3–1.6× |

**何时不应使用投机解码：**

- 单序列生成仅 1–5 个 token。开销占主导。
- 极度创意 / 高温度采样（α 下降）。
- 显存受限部署（起草模型增加 VRAM 占用）。

## 交付

见 `outputs/skill-spec-decode-picker.md`。该技能为新的推理工作负载选择投机解码策略（经典 / Medusa / EAGLE / lookahead）及调参（N、起草温度）。

## 练习题

1. **简单。** 运行 `code/main.py`。确认在 50,000 个 token 上，投机 token 分布与验证模型直接采样分布在卡方检验 p > 0.05 范围内一致。
2. **中等。** 绘制加速比（每次大模型前向的 token 数）随 `N` 变化的曲线，对应 `α = 0.5, 0.7, 0.85`。找出每个 α 下的最优 `N`。（提示：每次验证调用的期望 token 数 = `(1 - α^{N+1}) / (1 - α)`。）
3. **困难。** 实现一个微型 Medusa：取第 14 课的 capstone GPT，增加 3 个额外 LM head，分别预测位置 t+2、t+3、t+4。在 tinyshakespeare 上用联合多头损失训练。与截断同一模型得到的经典起草模型比较接受率。
4. **困难。** 实现回滚：从一个 10 token 前缀的 KV cache 开始，输入 5 个起草 token，模拟在位置 3 被拒绝。验证下一轮迭代时 cache 读取正确对应"前缀 + 前 2 个已接受起草 token"。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|----------|
| Draft model（起草模型） | "那个便宜的模型" | 提出候选 token 的较小模型；通常比验证模型便宜 10–50 倍。 |
| Verifier（验证模型） | "那个大模型" | 我们要保留其分布的目标模型；每轮投机步骤运行一次。 |
| Acceptance rate α（接受率） | "起草有多准" | 验证模型接受该起草 token 的逐 token 概率。典型值 0.7–0.9。 |
| Residual distribution（残差分布） | "拒绝时的后备方案" | `(q - p)_+` 归一化；从此分布采样可保持验证模型的分布。 |
| Bonus token（奖励 token） | "免费的那个" | 全部 N 个起草被接受时，从验证模型的下一步分布多采样一个 token。 |
| Medusa | "无起草模型的投机解码" | 验证模型上的多个 LM head 并行预测位置 t+1..t+k。 |
| EAGLE | "基于隐藏状态的起草" | 以验证模型最后一层隐藏状态为条件的极小型 transformer 起草模型。 |
| Lookahead decoding（前瞻解码） | "Jacobi 迭代" | 使用不动点迭代的自投机；无需起草模型。 |
| Tree attention（树注意力） | "同时验证多个候选" | 分支验证，同时考虑多个起草延续。 |
| KV rollback（KV 回滚） | "撤销被拒绝的起草" | 临时 KV 缓冲区；接受时提交，拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) —— 核心算法与等价定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) —— 同期提出；清晰的 Bernoulli 拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) —— Medusa 论文；tree-attention 验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) —— EAGLE-1；基于隐藏状态条件的起草。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) —— EAGLE-2；动态树深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) —— EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) —— lookahead，无需起草模型的方法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) —— 生产级标准参考，四种策略均已接入。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) —— EAGLE-1/2/3 的参考代码。
