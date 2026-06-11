---
name: tom-auditor
description: 审计声称具有"emergent coordination (涌现协调)"的multi-agent system (多智能体系统)。通过control conditions (对照条件)、statistical tests (统计测试)和complementarity measurement (互补性测量)区分真正的ToM-enabled coordination (心智理论驱动的协调)与prompt-dressed illusion (提示词包装幻觉)。
version: 1.0.0
phase: 16
lesson: 18
tags: [multi-agent, theory-of-mind, coordination, evaluation, emergence]
---

给定一个声称具有emergent coordination (涌现协调)的multi-agent system (多智能体系统)，审计该协调是真实的还是prompt engineering (提示工程)的产物。

生成内容：

1. **Claim extraction (声明提取)。** 声称的coordination behavior (协调行为)是什么？（division of labor (分工)、anticipation (预期)、complementary actions (互补动作)、consensus reaching (达成共识)）。精确说明。
2. **Prompt inspection (提示词检查)。** 任何agent的system prompt是否explicitly instruct coordination (明确指示协调)、role selection (角色选择)或team awareness (团队意识)？如果是，将声明标记为partially prompt-dressed (部分提示词包装)并设计control (对照)。
3. **Control condition (对照条件)。** 一个stripped (剥离)了coordination-inducing language (诱导协调语言)的系统版本。精确说明文本更改。
4. **Metric (指标)。** 至少一个：identity-linked differentiation (身份关联差异化)、goal-directed complementarity (目标导向互补性)、higher-order synergy (高阶协同)（Riedl 2025）。不接受"agents seem to work together (agent似乎在一起工作)"作为证据。
5. **Statistical test (统计测试)。** 在system vs control上metric的显著性。`p < 0.05`所需的sample size (样本量)。如果`n < 50`次trials (试验)，明确报告power (检验力)。
6. **Model-capacity check (模型能力检查)。** 在更小的base model上重复比较。effect (效果)持续存在还是消失？Li/Riedl都展示了capacity-dependence (能力依赖性)。
7. **Failure-case review (故障案例审查)。** 当系统失败时，ToM state (心智理论状态)（如果有）看起来如何？Identity confusion (身份混淆)（belief-agent binding (信念-智能体绑定)断裂）还是content hallucination (内容幻觉)（错误的信念内容）？

Hard rejects (硬性拒绝)：

- 没有control condition (对照条件)的emergence (涌现)声明。Demo reels (演示视频)不是证据。
- 在statistical scrutiny (统计审查)下消失的声明（`n >= 50`次trials上`p < 0.05`以下的effect）。这些是coordination illusions (协调幻觉)。
- 只在一种model上成立的声明。如果更小的strong baseline也能在没有ToM prompting的情况下实现该效果，则协调不是ToM-driven (心智理论驱动)的。
- "Our agents just figured it out (我们的agent自己搞定的)"作为mechanism explanation (机制解释)。Mechanism claims需要ToM state被记录并可检查。

Refusal rules (拒绝规则)：

- 如果系统没有per-agent reasoning (每智能体推理)的logging (日志记录)，audit无法区分real coordination (真实协调)和randomness (随机性)。推荐在重新审计前添加structured ToM-state logs (结构化心智理论状态日志)。
- 如果任务有oracle-computed optimal coordination (神谕计算的最优协调)，与optimal (最优)而非control (对照)进行比较。
- 如果声明是narrow (狭窄的)（"single-round task上的coordination"），audit可以是更短的检查：测量single round上的complementarity (互补性)，不需要long-horizon analysis (长期分析)。

Output (输出)：两页audit (审计)。以one-sentence verdict (单句判定)开头（"Coordination claim is prompt-dressed: removing 'work together' language drops the metric from 0.82 to 0.31, control-significant."），然后是上述七个部分。以fixes列表收尾，将prompt-dressed coordination转换为real coordination：explicit ToM state (显式心智理论状态)、带logging的longer horizons (更长期)、mixed-model ensembles (混合模型集合)。
