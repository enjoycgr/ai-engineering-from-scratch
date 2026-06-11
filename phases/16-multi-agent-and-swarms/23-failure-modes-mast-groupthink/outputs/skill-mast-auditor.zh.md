---
name: mast-auditor
description: 对multi-agent system (多智能体系统)运行MAST-style failure-mode audit (MAST风格故障模式审计)。将execution-trace failures (执行追踪故障)分类为Specification / Coordination / Verification和Groupthink families (家族)；按expected failure reduction (预期故障减少)对mitigations (缓解措施)进行排序。
version: 1.0.0
phase: 16
lesson: 23
tags: [multi-agent, failure-modes, MAST, groupthink, circuit-breaker, audit]
---

给定一个multi-agent system (多智能体系统)和sampled execution traces (采样执行追踪)，运行failure-mode audit (故障模式审计)。

生成内容：

1. **Sample construction (样本构建)。** 至少200个来自production (生产环境)的traces (追踪)，在task types (任务类型)和time windows (时间窗口)上均匀采样。记录sampling method (采样方法)和bias risks (偏差风险)。
2. **Classification pass (分类遍历)。** 对每个trace，标记`success | failure`。对failures，分配一个MAST category（spec / coord / verify）以及（如适用）一个或多个Groupthink family tags（monoculture / conformity / tom / mixed-motive / cascade）。
3. **Distribution table (分布表)。** 按MAST category和Groupthink tag的counts (计数)和percentages (百分比)。与Cemri 2025的reference distribution (参考分布)（41.77 / 36.94 / 21.30）比较。严重偏离reference的系统通常有特定的weak layer (薄弱层)。
4. **Top failure patterns (主要故障模式)。** 识别3个最常见的specific patterns (具体模式)（例如，"two agents both review (两个agent都进行审查)"）。记录reproduction steps (复现步骤)。
5. **Mitigation ranking (缓解排序)。** 对每个top pattern，从standard library (标准库)提出mitigation：explicit role contracts (显式角色契约)、versioned shared state (版本化共享状态)、independent verifier (独立验证器)、circuit breaker (熔断器)、detection-diagnosis-validation (检测-诊断-验证)（STRATUS）trio。按给定pattern频率的expected failure reduction (预期故障减少)排序。
6. **Risk of silent failures (静默故障风险)。** 多少failures产生plausible-but-wrong outputs (看似合理但错误的输出) vs loud errors (显式错误)？Silent rate (静默率)驱动verification-layer investment (验证层投入)。
7. **Slow-failure proxies (慢故障代理指标)。** 推荐2-3个live metrics (实时指标)，在drift (漂移)变成loud error (显式错误)前发现它：agreement rate (一致率)、retry-rate (重试率)、output-length distribution (输出长度分布)、inter-agent edit distance (智能体间编辑距离)。

Hard rejects (硬性拒绝)：

- 没有random or stratified sample (随机或分层样本)的audits。Hand-picked failures (手动挑选的故障)over-represent dramatic cases (过度代表戏剧性案例)并遗漏slow-failure drift (慢故障漂移)。
- 没有baseline measurement (基线测量)的mitigation recommendations (缓解建议)。"Add a verifier"没有意义，除非知道当前的failure rate。
- 忽略MAST-unknown incidents (MAST未知事件)。如果trace不符合category，taxonomy (分类法)不完整；提出extension (扩展)而不是force a category (强行归类)。
- 声称quarterly audit (季度审计)足够而没有operational slow-failure monitoring (运营慢故障监控)。Quarterly遗漏audits之间的drift。

Refusal rules (拒绝规则)：

- 如果traces缺少per-agent attribution (每智能体归属)（谁写了什么、谁读了什么），audit无法区分coordination failures (协调故障)和role conflicts (角色冲突)。推荐在重新审计前添加structured per-agent logging (结构化每智能体日志)。
- 如果系统总共少于50个failed traces (故障追踪)，sample太小无法产生distribution estimates (分布估计)。推荐更长的observation window (观察窗口)。
- 如果traces包含PII，分析前mask (脱敏)。

Output (输出)：三页report (报告)。以one-sentence summary (单句摘要)开头（"41% spec failures, 12% coordination, 39% verification gaps, 8% unknown; top pattern is dual-reviewer conflict; highest-ROI mitigation is explicit role contracts."），然后是上述七个部分。以prioritized action list (优先行动列表)收尾：三个mitigations (缓解措施)附带estimated implementation cost (预计实施成本)和expected failure-rate reduction (预期故障率降低)。
