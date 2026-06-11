---
name: reflexion-buffer
description: 维护一个用于 verbal RL（语言强化学习）的 episodic-memory buffer（情节记忆缓冲区），带 TTL、dedup（去重）和 scope policy（作用域策略）。
version: 1.0.0
phase: 14
lesson: 03
tags: [reflexion, episodic-memory, self-healing, verbal-rl, sleep-time]
---

给定一个任务类别（重复类型的智能体运行——例如 "重构函数"、"关闭支持工单"），维护一个 reflection（反思）的 episodic-memory buffer（情节记忆缓冲区）。每条 reflection 记录一个失败模式和自然语言中的纠正性洞察。该缓冲区被 prepend 到同一任务类别的下一次试验。

产出内容：

1. Reflection capture（反思捕获）。在试验以低于阈值的评估器分数结束后，发出一行格式的 reflection："I failed to do X because Y; next time, Z."。丢弃外部失败（网络、上游 500）上的 reflection，除非它们是可复现的。
2. TTL 和 dedup（去重）。Reflections 默认在 N 次试验后过期（建议 10）。精确重复项折叠。Near-duplicates（小 embedding 模型上余弦相似度 >0.9，或共享子串 >= 80%）只保留最近的。
3. Scope policy（作用域策略）。三种 scope：task-class（按任务名称）、user（同一用户的跨任务）、agent（跨所有用户）。默认是 task-class。仅在 reflection 引用用户特定偏好时升级到 user scope；永远不要自动升级到 agent scope。
4. Compaction（压缩）。当缓冲区超出预算时，运行 sleep-time compaction（睡眠时压缩）：聚类 near-duplicates、总结、合并。Compaction 在 hot path 之外运行——不要延迟主智能体的响应。
5. Prompt integration（Prompt 集成）。发出一个标题为 "What I learned from prior trials" 的单一块，带 bullet list。Prompt 中上限为 6 项；溢出放入单独摘要项（"... and 4 older reflections about timeouts"）。

Hard rejects（硬拒绝）：

- 将 reflection 写成 "be more careful next time."。这不是可操作的。用强制具体 next-time instruction 的 prompt 重新运行 reflector。
- 基于 wall-clock 时间而非试验次数使 reflection 过期。TTL 应该是试验范围的，不是时间范围的，以支持离线可重放运行。
- 存储引用 secret（API 密钥、token、PII）的 reflection。在提交到缓冲区之前用特定的 "contains secret" 类错误拒绝。

Refusal rules（拒绝规则）：

- 如果没有附加评估器，拒绝并推荐 Lesson 05（Self-Refine/CRITIC）—— reflection 需要信号，不是 gut feeling（直觉）。
- 如果任务类别是 one-shot（永不重复），拒绝；episodic memory 对永不重复的任务无效。

输出：结构化缓冲区文件（JSON，含 reflection 对象：trial id、task class、scope、text、created_at、ttl_remaining）、下一次试验的 prompt block，以及列出即将过期条目的 "stale reflections" 报告。

结尾附 "what to read next" 注释：如果缓冲区总是达到上限，指向 Lesson 06（context compression）；如果将 compaction 移出 hot path，指向 Lesson 08（Letta sleep-time compute）。
