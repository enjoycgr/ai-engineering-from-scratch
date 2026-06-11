# Speculative Decoding and EAGLE（推测解码与 EAGLE）

> 一个前沿 LLM 生成一个 token 需要数十亿参数上的完整前向传递。该前向传递是严重超配的：大多数时候一个小得多的模型可以正确猜测接下来的 3-5 个 token，而大模型只需要*验证*猜测。当猜测正确时，你以单个 token 的价格获得 5 个 token。Speculative decoding（Leviathan 等人 2023）使这变得精确，而 EAGLE-3（2025）将接受率推到约每次验证 4.5 个 token——在匹配输出分布的情况下 4-5 倍加速。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## The Problem

H100 上 70B 级模型的解码吞吐量通常为 40-80 token/秒。每个 token 需要完整前向传递，从 HBM 读取所有模型权重。你不能在不改变输出的情况下让模型更小。你不能超过内存增加 batch size。你卡住了——除非你能让模型每次前向传递输出多于一个 token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但有一个并发机会。如果你有一个廉价的预测器说"接下来 4 个 token 可能是 [a, b, c, d]"，你可以在**大模型的单次前向传递**中验证所有 5 个位置，并接受最长匹配前缀。

Leviathan, Kalai, Matias (2023, "Fast Inference from Transformers via Speculative Decoding") 通过一个巧妙的接受/拒绝规则使这变得精确，保留了目标模型的采样分布。相同的输出分布，2-4 倍更快。

## The Concept

### 双模型设置

- **Target model（目标模型）** `M_p`：你想要样本的大、慢、高质量模型。分布：`p(x)`。
- **Draft model（草稿模型）** `M_q`：一个小、快、低质量模型。分布：`q(x)`。小 5-30 倍。

每步：

1. Draft model 自回归提议 `K` 个 token：`x_1, x_2, ..., x_K ~ q`。
2. Target model 并行运行 ONE 前向传递覆盖所有 `K+1` 个位置，为每个提议 token 产生 `p(x_k)`。
3. 通过下面修改的拒绝采样规则从左到右接受/拒绝每个 token。接受最长匹配前缀。
4. 如果任何 token 被拒绝，从修正分布采样替换并停止。否则从 `p(· | x_1...x_K)` 采样一个 bonus token。

如果 draft 完美匹配 target，你每次 target-forward 获得 K+1 个 token。如果 draft 在位置 1 错误，你只获得 1 个 token。

### 精确性规则

Speculative decoding 在**分布上可证明等价于从 p 采样**。拒绝规则：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差的正部。当 draft 和 target 一致（`p ≈ q`）时接受接近 1。当它们不一致时，残差分布被构造为使整体样本仍然是精确的 `p`。

**贪婪情况。** 对于 temperature=0 采样只需检查 `argmax(p) == x_t`。如果是，接受；如果不是，输出 `argmax(p)` 并停止。

### 预期加速

如果 draft model 的 token 级接受率是 `α`，每次 target-forward 传递产生的预期 token 为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` 每次前向的 token。单次 target forward 大致成本为 `cost_q * K + cost_p`（K 个 draft 步骤加一次 target 验证）。如果 `cost_p >> cost_q * K` 则加速比为 `3.36× / 1 = 3.36×` 吞吐量。

唯一真正的参数是 `α`，它完全取决于 draft-target 对齐。好的 draft 就是一切。

### 训练 Draft：蒸馏

随机小模型是差的 draft。标准配方是从 target 蒸馏：

1. 选择小型架构（70B target 约 1B，7B target 约 500M）。
2. 在大型文本语料库上运行 target 模型；存储其下一 token 分布。
3. 用 KL 散度针对 target 分布（而非 ground-truth token）训练 draft。

结果：`α` 通常在代码上 0.6-0.8，自然语言聊天上 0.7-0.85。生产中 2-3 倍加速。

### EAGLE：树形 Draft + 特征重用

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") 观察到标准 speculative decoding 的两个低效：

1. Draft 做 K 个串行步骤，每个完整栈。但 draft 可以重用 target 最近验证的特征（隐藏状态）——target 已经计算了 draft 从头重新推导的丰富表示。
2. Draft 输出线性链。如果 draft 可以输出候选*树*（每个节点多个猜测），target 的单次前向传递可以通过 tree attention mask 并行验证多条候选路径，并选择最长接受分支。

EAGLE-1 改变：
- Draft 输入 = 位置 t 的 target 最终隐藏状态，而非原始 token。
- Draft 架构 = 1 个 transformer decoder 层（不是单独的小模型）。
- 输出 = 深度 K = 4-8 候选每深度，深度 4-6 的树。

EAGLE-2 (2024) 添加动态树拓扑：树在 draft 不确定的地方变宽，在自信的地方保持窄。在不增加验证成本的情况下提高 `α_effective`。

EAGLE-3 (Li 等人 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") 移除固定的顶层特征依赖，并用新的"test-time simulation"损失训练 draft——draft 在匹配 target test-time 分布的输出上训练，而非 teacher-forced 训练分布。接受率从 0.75 (EAGLE-2) 上升到 0.82 (EAGLE-3)，每次验证平均 token 从 3.0 到 4.5。

### Tree Attention Verification

当 draft 输出树时，target model 在单次前向传递中使用**tree attention mask**验证它——编码树拓扑而非纯线的 causal mask。每个 token 只 attends 树中的祖先。验证传递仍然是单次前向、单次 matmul；拓扑 mask 只花费少量额外 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的首 token 候选，`c, d, e, f` 是次 token 候选，所有六个位置在一次前向传递中验证。输出是沿任何接受路径的最长前缀。

### 何时赢，何时不赢

**赢：**
- 可预测文本的聊天/补全（代码、常见英语、结构化输出）。`α` 高。
- 解码期间有未使用 GPU 计算的设置（内存受限阶段）。树形 drafting 使用可用 FLOP。

**输/无赢：**
- 高度随机输出（高温创意写作）。`α` 降到 `1/|vocab|`。
- 非常高并发的 batch serving——batching 已经填满 FLOP，树验证空间很小。
- 非常小的 target model，draft 不会小多少。

生产环境通常报告聊天 2-3 倍 wall-clock 加速，代码生成 3-5 倍，创意写作接近零。

## Build It

`code/main.py`：

- 一个参考 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确拒绝规则并验证它保留 target 分布（经验 KL < 0.01 vs 纯 target 采样）。
- 一个 EAGLE 风格树 drafter，构建深度 K 树和 top-p 分支。
- 一个 tree attention mask builder，为验证器产生正确的 causal 模式。
- 一个接受率测试工具，在微型 LM 上运行两者（从 GPT-2-medium target 蒸馏一个 GPT-2-small draft）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Use It

- **vLLM** 和 **SGLang** 提供一流的 speculative decoding。标志：`--speculative_model`, `--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 标志支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考 draft 模型**：`Qwen/Qwen3-0.6B-spec`（Qwen3-32B 的 draft），`meta-llama/Llama-3.2-1B-Instruct-spec`（70B 的 draft）。
- **Medusa heads**（Cai 等人 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads")：替代 draft 模型，在 target 本身上添加 K 个并行预测 head。部署更简单，接受率略低于 EAGLE。

## Ship It

本课产出 `outputs/skill-speculative-tuning.md` — 一个 skill，分析 target 模型的工作负载并选择：draft 模型、K（draft 长度）、树宽度、温度，以及何时回退到朴素解码。

## Exercises

1. 实现精确拒绝规则并实证验证它。通过 `speculative_decode` 和纯 target 采样运行 10K 样本；计算两个输出分布之间的 TV 距离。应 < 0.01。

2. 计算加速公式。给定固定 `α` 和 `K`，绘制每次 target-forward 的预期 token。找到 α ∈ {0.5, 0.7, 0.9} 的最优 K。

3. 训练一个微型 draft。取 124M GPT-2 target 并在 100M token 上用 KL 损失蒸馏 30M GPT-2 draft。在留出文本上测量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格树 drafting。替代链，让 draft 在每深度输出 top-3 分支。构建 tree attention mask。验证 target 接受最长正确分支。

5. 测量失败模式。在 temperature=1.5（高随机性）下运行 speculative decode。展示 α 崩溃且算法因 draft 开销比朴素解码慢。

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Target model | "The big model" | 你想要样本的慢、高质量模型（p 分布） |
| Draft model | "The speculator" | 小、快预测器（q 分布）；小 5-30 倍 |
| K / draft length | "Look-ahead" | 每次验证传递推测的 token 数量 |
| α / acceptance rate | "Hit rate" | draft 提议被接受的 token 级概率 |
| Exact rejection rule | "The accept test" | 保留 target 分布的 r < p/q 比较 |
| Residual distribution | "Corrected p-q" | (p - q)+ / ||(p - q)+||_1，拒绝时从中采样的分布 |
| Tree drafting | "Branching speculation" | Draft 输出候选树，在单次传递中用树结构 attention mask 验证 |
| Tree attention mask | "Topological mask" | 编码树拓扑的 causal mask，使每个节点只 attends 其祖先 |
| Medusa heads | "Parallel heads" | 在 target 本身上添加 K 个额外预测 head；无单独 draft 模型 |
| EAGLE feature reuse | "Hidden-state draft" | Draft 输入是 target 的最后隐藏状态，非原始 token，缩小 draft |
| Test-time simulation loss | "EAGLE-3 training" | 在匹配 target test-time 分布的输出上训练 draft，非 teacher forcing |

## Further Reading

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind 的同期 speculative-sampling 论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — draft 模型的并行 head 替代方案
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 特征重用和树 drafting
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 动态树拓扑
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 训练时测试时匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead decoding，无 speculator 的替代方案
