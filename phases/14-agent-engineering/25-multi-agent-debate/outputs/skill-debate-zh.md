---
name: debate
description: 搭建一个 multi-agent debate（多智能体辩论），包含 N 个 debaters、R 轮、configurable topology（可配置拓扑）（full mesh 全连接、star 星型、ring 环形）和 convergence rule（收敛规则）。
version: 1.0.0
phase: 14
lesson: 25
tags: [debate, multi-agent, society-of-minds, sparse-topology]
---

给定一个 question class（问题类别）和 accuracy target（准确率目标），搭建一个 debate protocol（辩论协议）。

生成：

1. `Debater` with different prompts（不同的提示词）（and ideally different models 理想情况下不同的模型）以避免 homogenization（同质化）。
2. Round runner（轮次运行器）：full mesh（全连接）、star（星型）或 ring（环形）topology（拓扑）。
3. Convergence rule（收敛规则）：majority-vote（多数投票）、weighted by confidence（按置信度加权）或 supermajority-with-fallback（超多数回退）。
4. Round 1 forced disagreement（第一轮强制分歧）：如果可能，每个 debater 返回 distinct proposal（不同的提议）。
5. Cost accounting（成本核算）：total critique ops（总批评操作数）+ token cost per question（每问题 token 成本）。

Hard rejects（硬性拒绝）：

- 所有 debaters 使用相同的 prompt AND same model（相同模型）。保证 groupthink（群体思维）。
- 未检查成本就使用 N >= 6 的 full mesh。Debate ops 以 O(N*R) 规模扩展。
- 没有 convergence rule（收敛规则）。返回 debater 0 的 round-R answer 不是收敛。

Refusal rules（拒绝规则）：

- 如果产品是 latency-sensitive（延迟敏感）（<1s budget），拒绝 debate。使用 Self-Refine（Lesson 05）或 parallel voting（Lesson 12）代替。
- 如果 question class 是 simple factual lookup（简单事实查找）（capital 首都、date 日期、definition 定义），拒绝 debate。Lookup + CRITIC（Lesson 05）更便宜。
- 如果 debaters 在 eval set（评估集）的任何问题上第一轮后没有 disagreement（分歧），拒绝该协议。你需要 model/prompt diversity（模型/提示词多样性）。

输出：`debater.py`、`topology.py`、`convergence.py`、`runner.py`、`README.md`，解释 N/R choice（N/R 选择）、topology rationale（拓扑原理）和 cost-vs-accuracy measurements（成本 vs 准确率测量）on the eval set。最后以 "what to read next" 指向 Lesson 12（workflow patterns 工作流模式）如果任务更简单，或 Lesson 28（orchestration patterns 编排模式）了解如何将 debate 嵌入更大的系统。
