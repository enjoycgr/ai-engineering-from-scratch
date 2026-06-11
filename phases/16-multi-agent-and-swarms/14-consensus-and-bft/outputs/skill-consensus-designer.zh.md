---
name: consensus-designer
description: 为multi-agent ensemble (多智能体集合)设计BFT-aware consensus protocol (拜占庭容错共识协议)。选择clustering (聚类)、weighting (加权)、threshold (阈值)和escalation policy (升级策略)；针对byzantine (拜占庭)、sycophancy (谄媚)和monoculture (单一文化)模式进行attack-test (攻击测试)。
version: 1.0.0
phase: 16
lesson: 14
tags: [multi-agent, consensus, BFT, voting, confidence]
---

给定一个N个agent回答共同问题的ensemble (集合)，设计一个consensus protocol (共识协议)，能够抵御三种典型的LLM-agent attacks (LLM智能体攻击)：byzantine lie (拜占庭谎言)、sycophantic conformity (谄媚性从众)、correlated-error monoculture (相关错误单一文化)。

生成内容：

1. **Clustering strategy (聚类策略)。** 如何分组答案？String canonicalization (字符串规范化)（lowercase + strip punct (去除标点)）、embedding similarity with threshold (带阈值的嵌入相似度)或explicit structural canonicalization (显式结构规范化)（JSON schema）。说明预期的cluster-granularity error rate (聚类粒度错误率)。
2. **Weighting strategy (加权策略)。** Plurality (计数)、confidence-probe weighted (CP-WBFT)、quality-plus-trust (WBFT)或score-based with geometric-median robustness (基于分数配合几何中值鲁棒性)（DecentLLMs）。根据attack profile (攻击特征)说明选择理由。
3. **Threshold (阈值)。** 触发acceptance (接受)的总weight (权重)比例是多少？低于threshold时会发生什么：retry (重试)、escalate (升级)还是abstain (弃权)？
4. **Diversity requirement (多样性要求)。** ensemble需要多少base models (基础模型)、prompt families (提示词家族)或temperature settings (温度设置)？Monoculture (单一文化)是plurality (多数表决)无法恢复的attack (攻击)；diversity (多样性)是structural mitigation (结构性缓解)。
5. **Independent verifier (独立验证器)。** 是否有具有ground truth (真实值)（当可用时）或rubric (评分标准)的read-only agent (只读智能体)？verifier的输出去哪里？它不能重新进入voting pool (投票池)。
6. **Round bounding (轮次限制)。** 升级前的最大rounds (轮次)。大多数任务默认2-3轮。更长的rounds会放大sycophancy (谄媚)。
7. **Attack-test table (攻击测试表)。** 对于每种攻击（byzantine (拜占庭)、sycophancy (谄媚)、monoculture (单一文化)），展示预期的protocol behavior (协议行为)和residual risk (残余风险)。如果协议承认已知的failure mode (故障模式)，用一句话说明。

Hard rejects (硬性拒绝)：

- 任何在single base model (单一基础模型)上只使用plurality-only (仅多数表决)的设计。Monoculture (单一文化)使其静默失败。
- 任何具有unbounded rounds (无界轮次)或"keep debating until agreement (一直辩论直到达成一致)"的设计。这奖励conformity (从众)。
- 任何让verifier's output (验证器输出)反馈回voting pool (投票池)的设计。这会poison (污染)验证器。
- 声称BFT "solves" disagreement (BFT "解决"分歧)。BFT对齐输出；correctness (正确性)是一个单独的问题。

Refusal rules (拒绝规则)：

- 如果任务没有ground truth (真实值)（opinion (观点)、synthesis (综合)、creative (创意)），说明这一点并推荐"consensus as advisory (共识作为建议), human as decider (人类作为决策者)"。
- 如果可用agent少于3个，consensus (共识)不适用；推荐single agent plus verifier (单智能体加验证器)。
- 如果所有agent共享一个base model且用户无法更改，明确标记monoculture ceiling (单一文化上限)。

Output (输出)：一页design brief (设计简报)。以single-sentence summary (单句摘要)开头（"Confidence-weighted voting over 5 agents (3 base models), semantic-cluster threshold 0.55, independent verifier re-fetches sources, max 2 rounds."），然后是上述七个部分。以attack-test table (攻击测试表)收尾。
