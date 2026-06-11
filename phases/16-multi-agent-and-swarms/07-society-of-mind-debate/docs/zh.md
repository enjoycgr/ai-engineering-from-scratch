# 心智社会与多智能体辩论

> Minsky 1986 年的前提——智能是一个专家的社会——每十年都会被重新发现。2023 年，Du 等人将其转化为一个具体算法：多个 LLM 实例提出答案，阅读彼此的答案，进行批判，并更新。经过 N 轮后，它们收敛到一个共识，在六个推理和事实性任务上击败了 zero-shot CoT 和 reflection。两个发现很重要：**多个 agent (智能体)** 和 **多轮次** 各自独立地做出贡献。社会击败了单智能体独白；多轮次交换击败了一次性投票。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Problem

Self-consistency (自一致性)——从一个模型中采样多次并取多数答案——是你能附加的最便宜的推理改进。它有效，但很快饱和。你可以将样本数量翻倍，却看不到另一次有意义的提升。

Debate (辩论) 打破了这种饱和。不再是来自一个模型的 N 个独立样本，而是 N 个 agent 阅读彼此的推理并修订。样本之间的相关性下降（它们不再是 i.i.d.），而收敛点往往是正确的，而 i.i.d. 投票则自信地错了。

## Concept

### The Du et al. 2023 algorithm

来自 arXiv:2305.14325 (ICML 2024)：

1. N 个 agent 中的每一个都对问题产生一个初始答案。
2. 对于第 r = 2..R 轮：每个 agent 被展示其他 agent 的第 r-1 轮答案，并被问"考虑到这些，给出你的更新答案。"
3. 经过 R 轮后，对最终答案进行 majority-vote (多数投票)。

该论文在 MMLU、GSM8K、传记、MATH 和事实性基准上进行了测试。Debate 始终击败 CoT 和 Self-Reflection。

### Two independent knobs

来自同一论文的 ablations (消融实验)：

- **仅 agent 数量**（1 轮，N 个的多数投票）在大多数任务上击败单 agent，但会 plateau (饱和)。
- **仅轮次数**（1 个 agent 看到自己的先前推理）几乎没帮助——这是 reflection 的已知弱点。
- **两者结合**产生大幅提升。多个 agent 之间的多轮次交换推动了增益。

### Why it works

两种机制：

1. **Exposure to disagreement (暴露于分歧).** 当一个 agent 看到另一个 agent 的推理链得出不同结论时，它必须要么 justify (论证) 要么 update (更新)。无论哪种方式，第 r+1 轮的 context 都比第 r 轮更丰富。
2. **Correlated error reduction (相关错误减少).** 在 self-consistency 中，所有样本来自同一个模型，因此错误是相关的——你平均出一个自信的错误答案。不同的模型或不同的 seed 可以 decorrelate (去相关)。不同的*辩论观点*进一步去相关。

### Heterogeneous debate (异构辩论)

A-HMAD 和相关后续工作为不同的 agent 使用*不同的 base model (基础模型)*。Llama + Claude + GPT 辩论减少了 monoculture collapse (单一文化崩溃)（第 26 课），因为一个模型家族的相关错误不会被其他模型共享。

缺点：一个弱模型参与辩论可能将 consensus (共识) 拖向它的错误答案（参见 "Should we be going MAD?", arXiv:2311.17371）。

### NLSOM — the 129-agent extension

Zhuge 等人（"Mindstorms in Natural Language-Based Societies of Mind," arXiv:2305.17066）将这一想法扩展到 129 个成员的社会。结果：随着规模扩大，specialization (专业化) 和 self-organization (自组织) 涌现，系统在视觉问答等任务上优于单 agent。

### Failure modes

- **Sycophancy cascade (谄媚级联).** 所有 agent 都 defer (顺从) 于听起来最自信的那个 agent。辩论 collapse (崩溃) 为最大的声音。通过 prompting adversarial roles (对抗性角色)（"一个 agent 必须论证反方立场"）来缓解。
- **Topic drift (主题漂移).** 多轮辩论会从原始问题 drift (漂移)。缓解：每轮重新注入问题。
- **Compute blowup (计算爆炸).** N 个 agent × R 轮 = N·R 次 LLM 调用，每次的 context 都在增长。5 个 agent、5 轮辩论是 25 次调用，context 不断增长。每个问题的成本可能超过单次 CoT 调用的 10 倍。

## Build It

`code/main.py` 运行一个 3-agent × 3-round 的辩论，讨论一个数学问题，每个 agent 从一个不同的（可能是错误的）答案开始。Agent 是 scripted (脚本化的)——每个 agent 通过按 scripted confidence (脚本化置信度) 加权平均邻居的答案来"更新"。收敛过程在逐轮日志中可见。

Demo 展示了两个关键效果：

- 单轮交换使 agent 更接近正确答案。
- 超过第 2 轮的额外轮次显示 diminishing returns (收益递减)（与 Du 等人的 plateau 一致）。

Run:

```
python3 code/main.py
```

## Use It

`outputs/skill-debate-configurator.md` 为新任务配置辩论：agent 数量、轮次数、heterogeneity (同质性/异质性)（相同模型 vs 混合）、role assignment (角色分配)（symmetric (对称) vs one-adversarial (单一对抗性)）。它还在运行前估计 token 成本。

## Ship It

如果你要部署 debate：

- **Cap rounds at 3.** Du 等人表明 3 轮捕获了大部分收益。更多是成本，不是质量。
- **Cap agents at 5.** 超过 5 个，context bloat (膨胀) 和成本占主导。
- **Heterogeneous by default.** 池中至少有两个不同的 base model。
- **Adversarial slot.** 一个被提示无论如何都要 disagree (反对) 的 agent。打破 sycophancy。
- **Log every round.** 隐藏中间轮次的辩论系统无法被调试或审计。

## Exercises

1. 运行 `code/main.py`，然后将轮次数设为 5 并观察 diminishing returns。在哪一轮额外的收敛停止？
2. 添加一个具有 adversarial role (对抗性角色) 的第四个 agent：始终与当前多数意见 disagree。这会破坏还是改善收敛？
3. 绘制（打印）每轮的 agreement score (一致率)（落在多数答案上的 agent 比例）。它何时达到 1.0，这是否等同于"正确"？
4. 阅读 Du 等人第 4 节的 ablations。使用此代码复现"仅 agent" vs "仅轮次" vs "两者"的结果。
5. 阅读 "Should we be going MAD?" (arXiv:2311.17371) 并列出两种超越 round-robin (轮询) 的辩论变体——例如，judge-led (法官主导)、chain-of-debate (辩论链)、adversarial (对抗性)。

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Society of Mind | "Minsky's idea" | Intelligence as interacting specialists; 1986 framing now operationalized via LLM debate. |
| Multi-agent debate | "Agents argue" | N agents propose, critique each other, revise over R rounds, majority-vote. |
| Consensus | "They agree" | Not epistemic truth — just fraction-on-majority-answer. Can be confidently wrong. |
| Rounds | "Exchange steps" | One round = each agent reads the others and updates once. |
| Heterogeneous debate | "Mix model families" | Using different base models to decorrelate errors. |
| Sycophancy cascade | "Everyone agrees with the loud one" | Debate failure where agents defer to the most confident agent regardless of correctness. |
| NLSOM | "129-agent society" | Natural-language society of mind; Zhuge et al.'s scaled version. |
| Correlated error | "Same model, same bug" | Why self-consistency saturates; debate across different views decorrelates. |

## Further Reading

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — the reference paper, ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129-agent NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmarks debate variants
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du et al.'s code, demos, and ablation details
