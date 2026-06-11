---
name: economy-designer
description: 设计一个minimal agent economy (最小智能体经济)——identity (身份)、credit attribution (信用归属)、payment mechanism (支付机制)、reputation (信誉)。选择能解决用户multi-agent incentive problem (多智能体激励问题)的最小栈。
version: 1.0.0
phase: 16
lesson: 21
tags: [multi-agent, economy, Shapley, auctions, reputation, DePIN]
---

给定一个需要incentive alignment (激励对齐)的multi-agent scenario (多智能体场景)（open network (开放网络)、heterogeneous operators (异构运营者)、tokenized rewards (代币化奖励)或reputation-based routing (基于信誉的路由)），设计economy layer (经济层)。

生成内容：

1. **Identity layer (身份层)。** 对于portable identity (可移植身份)使用W3C DIDs，如果系统是closed (封闭的)则使用platform-internal IDs (平台内部ID)。根据网络的openness (开放性)说明理由。
2. **Credit attribution (信用归属)。** Equal split (均分)、last-contributor-takes-all (最后贡献者全拿)、contribution-weighted (按贡献加权)、Shapley (精确或sampled (采样))或none (无)（pay-per-call (按调用付费)）。当coalitions (联盟)重要时推荐Shapley sampling；对simple pay-per-call推荐equal split。
3. **Payment mechanism (支付机制)。** Second-price auction (二价拍卖)用于task assignment (任务分配)（truthful under monotone aggregation (单调聚合下真实)）、first-price (一价)用于speed (速度)、posted-price (标价)用于simplicity (简单)。如果payoffs依赖quality verification (质量验证)，使用Escrow (托管)。
4. **Reputation rule (信誉规则)。** Exponential decay constant (指数衰减常数)、slashing policy (削减策略)、minimum floor (最低下限)、maximum ceiling (最高上限)。Reputation reads cheaply (读取廉价)（O(1)用于routing (路由)）并在verification (验证)后writes (写入)。
5. **Verification (验证)。** 谁验证contribution quality (贡献质量)？Separate agent (独立智能体)、human review (人工审核)、on-chain oracles (链上预言机)、cross-agent attestation (跨智能体认证)？没有verification，credit attribution是guesswork (猜测)。
6. **Sybil mitigation (女巫攻击缓解)。** 什么阻止一个operator运行N个fake agents？Reputation cost-to-forge (伪造信誉成本)、proof-of-humanity attestation (人性证明认证)、stake requirement (质押要求)或capped reputation per DID (每DID信誉上限)。
7. **Legal and jurisdictional check (法律与管辖权检查)。** Token-denominated payments (代币计价支付)在大多数jurisdictions (司法管辖区)触及financial regulation (金融监管)。如果适用，标记它并推荐legal review (法律审查)。

Hard rejects (硬性拒绝)：

- 任何没有verification of contribution quality (贡献质量验证)的设计。Credit将accumulate (积累)给fastest-but-wrongest agents (最快但最错的智能体)。
- 没有decay (衰减)的reputation (信誉)。Stale reputation (陈旧信誉)奖励那些过去做得好但现在broken (故障)的agent。
- 对N > 6的Shapley exact computation (精确计算)。Computation time按N!增长；改用sample (采样)。
- aggregation function不是monotone (单调)的second-price auctions。Truthfulness不成立。
- 没有regulatory check (监管检查)的token distribution (代币分发)。许多jurisdictions将其视为securities activity (证券活动)。

Refusal rules (拒绝规则)：

- 如果系统完全是internal (内部的)（一家公司、一个运营者），推荐更简单的allocation (分配)（managers assign (经理分配)、metrics are internal (指标是内部的)）。Economic mechanisms (经济机制)是overkill (过度设计)。
- 如果没有验证contribution quality的方法，推荐在economy design前添加verification。没有它，economy只是ornamental (装饰性的)。
- 如果用户想要tokenized system (代币化系统)但没有legal team (法律团队)，标记risk并推荐从reputation (非代币)开始。

Output (输出)：两页brief (简报)。以one-sentence summary (单句摘要)开头（"Reputation-only system with DIDs, Shapley-sampled credit on 3-agent pipelines, second-price auction for slot assignment, slashing on verification failure."），然后是上述七个部分。以30-day pilot plan (30天试点计划)收尾：warmup phase (预热阶段)、verification pipeline setup (验证流水线搭建)、reputation-weighted rollout (信誉加权上线)、audit schedule (审计计划)。
