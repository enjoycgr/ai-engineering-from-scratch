---
name: topology-picker
description: 为给定任务选择multi-agent debate topology (多智能体辩论拓扑)：star (星形) / chain (链形) / tree (树形) / graph (图形)，agent数量N，heterogeneity profile (异质性特征)和round bound (轮次上限)。
version: 1.0.0
phase: 16
lesson: 15
tags: [multi-agent, debate, topology, voting, self-consistency]
---

给定一个task description (任务描述)，推荐一个multi-agent topology (多智能体拓扑)和sizing (规模)。

生成内容：

1. **Task fingerprint (任务指纹)。** Research (研究)（long-horizon (长周期)、open-ended (开放式)）、fast-factual (快速事实)（closed-form answer (闭式答案)）、stepwise-refinement (逐步细化)（staged pipeline (分阶段流水线)）或opinion (观点)（无ground truth (真实值)）。选择一个；如果跨越两个，选择主导形状。
2. **Topology (拓扑)。** Star (星形)、chain (链形)、tree (树形)或graph (图形)。从fingerprint (指纹)说明理由：
   - research → graph (任意对任意的critique (批评))
   - fast-factual → star (hub aggregates (中心聚合))
   - stepwise-refinement → chain (或tree (树形)如果使用divide-and-conquer (分而治之))
   - opinion → 以上都不是；推荐single agent + human decision (单智能体+人类决策)
3. **N of agents (智能体数量)。** 3是最便宜的有用ensemble (集合)；5是常见的sweet spot (最佳点)；7+是specialty (特殊用途)。graph topology (图形拓扑)上超过5个时，警告coordination tax (协调税)。
4. **Heterogeneity profile (异质性特征)。** 如果monoculture (单一文化)很重要（research (研究)、reasoning (推理)），至少有一个agent必须来自不同的base model family (基础模型家族)。N=5时优先选择3个不同的base models。
5. **Round bound (轮次上限)。** 1轮 = vote (投票)。2轮 = one refinement (一次细化)。3轮 = 在conformity (从众)主导前的最大值。Never unbounded (永无界)。
6. **Aggregation (聚合)。** Plurality (多数表决)（便宜）、confidence-weighted (置信加权)（第14课的CP-WBFT）、geometric median (几何中值)（DecentLLMs）或judge-scored (裁判评分)。除非cost constraints (成本限制)要求plurality (多数表决)，否则默认confidence-weighted (置信加权)。
7. **Escalation (升级)。** Below-threshold consensus (低于阈值的共识) → 升级到哪里？Human (人类)、another ensemble with different base models (使用不同基础模型的另一个集合)或abstention (弃权)？

Hard rejects (硬性拒绝)：

- 任何在graph topology (图形拓扑)上推荐10+个agent的建议。Coordination tax (协调税)占主导；先测量。
- 用于open research questions (开放式研究问题)的star topology (星形拓扑)。Star (星形)失去了any-to-any critique (任意对任意批评)的好处。
- 任何将相同的base model运行N次并称之为multi-agent (多智能体)的建议。这是disguised self-consistency (伪装的自洽性)；正确标记它。
- Unbounded rounds (无界轮次)。奖励conformity (从众)；debate (辩论)运行时间越长，agent通过pressure (压力)而非logic (逻辑)达成一致越多。

Refusal rules (拒绝规则)：

- 如果任务没有ground truth (真实值)（opinion (观点)、synthesis (综合)、creative (创意)），说明voting (投票)仅供参考。推荐single agent + human decision (单智能体+人类决策)。
- 如果用户无法访问多个base models，标记monoculture ceiling (单一文化上限)并推荐self-consistency with temperature variation (温度变化的自洽性)作为fallback (回退)。
- 如果任务简单（single factual lookup (单一事实查找)、< 100 tokens of reasoning (推理token)），推荐single agent with self-consistency N=5 (自洽性N=5的单智能体)。

Output (输出)：一页brief (简报)。以single-sentence recommendation (单句建议)开头（"Graph topology, N=5 agents from 3 different base models, 2 rounds, confidence-weighted aggregation, escalate to human on below-threshold."），然后是上述七个部分。以budget estimate (预算估算)收尾：每次查询的expected tokens (预期token)和expected latency in seconds (预期延迟秒数)。
