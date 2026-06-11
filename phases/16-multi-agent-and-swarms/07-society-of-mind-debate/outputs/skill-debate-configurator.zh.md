---
name: debate-configurator
description: 为给定任务配置multi-agent debate (多智能体辩论)，在运行前估计quality gain (质量提升)和token cost (token成本)。
version: 1.0.0
phase: 16
lesson: 07
tags: [multi-agent, debate, society-of-mind, consensus]
---

给定一个问题或任务，生成一个可在任何agent framework (智能体框架)（LangGraph、AutoGen、custom loop (自定义循环)）上运行的debate configuration (辩论配置)。

生成内容：

1. **Task-fit check (任务适配检查)。** 该任务是否能通过consensus (共识)改进？Debate (辩论)有助于reasoning (推理)、factuality (事实性)和decomposition (分解)；它无助于已经是deterministic (确定性的)任务（arithmetic (算术)、code compilation (代码编译)）或纯粹generative (生成性的)任务（creative writing (创意写作)）。
2. **Agent count (智能体数量)。** 3、4或5个。默认3个；只有cost-insensitive (成本不敏感)且任务需要更多diverse views ( diverse views (多样视角)）时才用4+。
3. **Round count (轮次)。** 2或3轮。默认3轮；很少更多。引用Du et al.的plateau (平台期)。
4. **Heterogeneity (异质性)。** Same base model (相同基础模型)（更简单、更便宜、错误更相关）或mixed family (混合家族)（Llama + Claude + GPT；decorrelates (去相关)；更昂贵，需要routing layer (路由层)）。
5. **Role assignment (角色分配)。** Symmetric (对称)（所有agent具有相同角色）vs one-adversarial (单一对抗)（一个agent被指示disagree ( disagree (反对)））。Adversarial slot (对抗位)是防止sycophancy cascades (谄媚级联)的廉价保险。
6. **Aggregation method (聚合方法)。** Majority vote (多数投票)（离散答案）、weighted average (加权平均)（数值）或LLM-judge synthesis (LLM裁判合成)（开放式）。
7. **Cost estimate (成本估计)。** N个agent × R轮 × 每轮中位数tokens。根据当前provider pricing (提供商定价)给出dollar estimate (美元估算)。

Hard rejects (硬性拒绝)：

- 任何没有具体cost-justification (成本论证)就配置超过5个agent或超过3轮的配置。
- 对已知有sycophancy risk (谄媚风险)的任务使用symmetric-only (仅对称)的debates (辩论)。
- 对具有deterministic verifier (确定性验证器)的任务使用debate (辩论)（compile (编译)、test (测试)、exact math (精确数学)）——直接运行验证器。

Refusal rules (拒绝规则)：

- 如果任务是简单的factual lookup (事实查找)，拒绝并推荐retrieval-augmented single-agent (检索增强单智能体)。
- 如果任务是generative (生成性的)（write a poem (写诗)），拒绝——debate (辩论)会将输出drag toward the mean (拉向平均值)。
- 如果用户未设置token/dollar budget (预算)，拒绝并要求提供一个。Debate (辩论)的成本是single-agent (单智能体)的5-15倍。

Output (输出)：一页config brief (配置简报)。以task-fit check (任务适配检查)开头，以total cost estimate (总成本估算)收尾。
