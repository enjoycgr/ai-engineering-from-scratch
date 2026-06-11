---
name: simulation-designer
description: 为给定场景设计generative-agent simulation (生成式智能体仿真)（Smallville风格）。指定memory schema (记忆模式)、reflection cadence (反思节奏)、plan horizon (规划范围)、spatial/social constraints (空间/社会约束)和evaluation metrics (评估指标)。
version: 1.0.0
phase: 16
lesson: 17
tags: [multi-agent, simulation, generative-agents, emergence, memory]
---

给定一个需要agent population (智能体群体)产生emergent behavior (涌现行为)的场景（social simulation (社会仿真)、game NPCs (游戏NPC)、policy rehearsal (政策推演)、market dynamics (市场动态)），设计simulation (仿真)。

生成内容：

1. **Population size and heterogeneity (群体规模与异质性)。** N个agent；哪些共享base model (基础模型) vs 不同；prompt families (提示词家族)；role distribution (角色分布)。Smallville使用了25个homogeneous agents (同质智能体)配合individualized personas (个性化人设)；更大的群体受益于heterogeneity (异质性)。
2. **Memory schema (记忆模式)。** 每条目的字段：`(ts, kind, content, importance, embedding_ref, source_ids)`。Recency-decay constant (近因衰减常数)；importance scoring procedure (重要性评分程序)；relevance metric (相关性指标)（cosine with embedding model X）。Retention policy (保留策略)用于compaction (压缩)。
3. **Reflection cadence (反思节奏)。** Trigger (触发)：unprocessed importance (未处理重要性)之和 > threshold (阈值)，或每N个observations (观测)，或periodic tick (周期性滴答)。每次trigger的reflections (反思)数量。Reflection prompt template (反思提示词模板)。
4. **Plan horizon (规划范围)。** Day (日) / hour (时) / action (动作)级别。哪些是强制的；哪些是可选的。Revision trigger (修订触发)：importance > threshold的新observation与active plan (活跃计划)矛盾时。
5. **World model (世界模型)。** Spatial grid (空间网格)、social graph (社交图)、resource constraints (资源约束)。什么构成observation (观测)（line-of-sight (视线范围)、conversation (对话)、notification (通知)）。architecture未学习而必须显式编码的normative constraints (规范约束)（capacity limits (容量限制)、closed hours (闭店时间)、private spaces (私人空间)）。
6. **Seed goals (种子目标)。** 哪些agent被seeded (植入)了哪些priorities (优先级)。可能compete (竞争)的overlapping goals (重叠目标)；应该coexist (共存)的non-competing goals (非竞争目标)。
7. **Budget (预算)。** 每tick每个agent的LLM calls（observe + retrieve + reflect + plan + act）。每tick每个agent的预期tokens。T ticks的总simulation cost (仿真成本)。
8. **Evaluation metric (评估指标)。** Believability (可信度)（human-rater (人工评分员)）、goal achievement rate (目标达成率)、coordination events counted (协调事件计数)、spatial-norm violations (空间规范违规)作为failure signal (失败信号)。

Hard rejects (硬性拒绝)：

- 没有explicit spatial / social norm encoding (显式空间/社会规范编码)的设计。架构会违反它们（Park 2023中的closed-store (闭店)、single-bathroom (单卫生间)失败）。
- 具有mutable memory (可变内存)的设计。Memory必须是append-only (仅追加)；corrections (修正)是new entries (新条目)。
- 每tick都运行reflection (反思)的设计。这是budget-inefficient (预算低效)的；reflection很昂贵，triggers应该是threshold-based (基于阈值的)。
- 大规模N (> 50)没有memory-compaction strategy (内存压缩策略)的simulations。Retrieval cost随stream length (流长度)增长。

Refusal rules (拒绝规则)：

- 如果场景需要emergent *task execution (任务执行)* 而非emergent *social behavior (社会行为)*，推荐supervisor / roles / primitives patterns (监督者/角色/原语模式)（第16阶段 · 05-08）。Smallville用于social simulation (社会仿真)。
- 如果预算允许每tick少于100次LLM calls total，推荐N = 3-5配合dense interactions (密集交互)而非更大的群体。
- 如果场景不受益于emergence (涌现)（tightly-scripted task (紧密编排的任务)），推荐single-agent + tools (单智能体+工具)。

Output (输出)：一页design brief (设计简报)。以single-sentence summary (单句摘要)开头（"Smallville-style simulation: 15 heterogeneous agents, reflection at importance sum > 120, 3-level plan horizon, spatial grid with capacity constraints, measured by believability + coordination events."），然后是上述八个部分。以expected emergent behaviors (预期涌现行为)和first three failure modes to watch for (前三个需关注的故障模式)收尾。
