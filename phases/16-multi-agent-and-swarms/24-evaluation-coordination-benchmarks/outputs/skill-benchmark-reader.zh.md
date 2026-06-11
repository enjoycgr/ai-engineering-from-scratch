---
name: benchmark-reader
description: 批判性地阅读multi-agent benchmark claim (多智能体基准声明)。在benchmark selection (基准选择)、contamination (污染)、baselines (基线)、statistical significance (统计显著性)、task diversity (任务多样性)和cost disclosure (成本披露)方面对声明进行评分。
version: 1.0.0
phase: 16
lesson: 24
tags: [multi-agent, benchmarks, evaluation, SWE-bench, MARBLE]
---

给定一个已发布或内部的multi-agent benchmark performance claim (多智能体基准性能声明)，对声明进行评分并标出caveats (注意事项)。

生成内容：

1. **Benchmark + split identification (基准+划分识别)。** 哪个benchmark（MARBLE、COMMA、MedAgentBoard、AgentArch、SWE-bench Pro、SWE-bench Verified、custom (自定义)）？哪个split（full (完整)、held-out (留出)、contamination-cleaned (去污染)）？Unknown splits (未知划分)是disqualifying (不合格的)。
2. **Contamination status (污染状态)。** 该benchmark对测试下的model是否是post-training-cutoff (训练截止后)的？如果benchmark predates training cutoff (早于训练截止)，标记contamination risk (污染风险)并discount the claim (打折声明)。
3. **Baseline quality (基线质量)。** Vs single-LLM、vs random、vs prior multi-agent work。Vs untuned-same-system不算；它是ablation (消融)，不是baseline (基线)。
4. **Statistical significance (统计显著性)。** N trials (试验次数)、confidence interval (置信区间)或standard error (标准误)、p-value或equivalent (等价物)。N < 50 trials上的claims without statistics (无统计)是under-supported (支持不足的)。
5. **Task diversity (任务多样性)。** 一个任务、一个domain (领域)还是多个？Single-task claims (单任务声明)不意味着generalization (泛化)。
6. **Cost disclosure (成本披露)。** Tokens per task (每任务token数)、wall-clock per task (每任务挂钟时间)、dollar cost per task (每任务美元成本)。90% solution at 20x cost (20倍成本的90%方案)是business decision (商业决策)；没有成本，声明是incomplete (不完整)的。
7. **Letter grade + one-sentence verdict (字母等级+单句判定)。**

   - **A：** 六项检查全部通过；声明可能是robust (稳健的)。
   - **B：** 一个weakness (弱点)；声明是plausible (可信的)且有标出的caveats (注意事项)。
   - **C：** 两个weaknesses；声明是suggestive (有提示性的)但需要replication (复现)。
   - **D：** 三个或更多weaknesses；声明不是evidence (证据)。
   - **F：** Disqualifying issue (不合格问题)（undisclosed split上的contamination、no statistics (无统计)、no baseline (无基线)）。

Hard rejects (硬性拒绝)：

- 引用"SWE-bench"但未指定Verified vs Pro的claims。40+ point gap使这种ambiguous reporting (模糊报告)不可接受。
- 没有baseline comparison (基线比较)的claims。"Our system does X%"是一个数字，不是result (结果)。
- 基于少于20次trials的multi-agent systems claims。Variance太高。
- Multi-agent systems的cost-unreported claims (未报告成本的声明)。Coordination tax是material (实质性)的。

Refusal rules (拒绝规则)：

- 如果benchmark不是publicly available (公开可用)且用户没有internal audit trail (内部审计追踪)，无法assign grade (评分)。推荐releasing evaluation artifacts (发布评估产物)。
- 如果声明来自under peer review (同行评审中)的paper（arXiv preprint、unsubmitted (未提交)），作为precaution (预防措施)downgrade one letter grade (降一级)，直到replication (复现)。
- 如果用户是claimant (声明者本人)并请求audit，直接运行audit；标记claim尚未ready for publication (准备好发表)的情况。

Output (输出)：一页grade card (评分卡)。以one-sentence summary (单句摘要)开头（"Grade: C — good benchmark choice, adequate baselines, but no contamination check and no cost disclosure."），然后是上述七个部分。以prioritized list (优先列表)收尾："what to fix to raise the grade (如何提高等级)"。
