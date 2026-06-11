---
name: tripwire-design
description: 在首次自主运行之前，审核拟议智能体检测器栈（kill switch, circuit breakers, canary tokens）并标记缺失的绊线。
version: 1.0.0
phase: 15
lesson: 14
tags: [kill-switch, circuit-breaker, canary, honeytoken, detection-and-response]
---

给定一个拟议的智能体部署检测器栈，根据三检测器参考标准（kill switch, circuit breaker, canary）进行审核，并标记缺失、调优不当或暴露给智能体的部分。

产出：

1. **Kill-switch 审核。** 开关位于何处（功能标志、Redis、签名配置）？确认智能体的凭证无法将其关闭。确认每次 consequential 行动都检查开关，而不仅仅在启动时。确认重新启用是显式的人类操作。
2. **Circuit-breaker 清单。** 列出断路器监视的每个模式（重复、连续失败、速率、out-of-trust 读取后的特定工具）。说明每个模式的阈值和冷却期。阈值高于 10 通常过松。
3. **Canary 设计。** 列出环境中的每个 canary token。对每个 token：它是什么（假凭证、假 DB 记录、假文件、假记忆条目）、它位于何处、什么访问触发警报、谁被 paging。确认没有 canary 有正当理由被接触。
4. **统计 + 硬性分层。** 确认栈至少使用一个硬性限制（第 17 课宪法风格）以及任何统计检测器（EWMA, z-score）。纯统计检测器接受缓慢漂移。
5. **隔离路径。** 检测器触发时会发生什么？完全停止智能体、路径特定暂停、流量重定向（eBPF / Cilium 蜜罐）、仅警报。确认路径至少经过一次端到端测试。

硬性拒绝：
- 任何没有外部 kill switch 的部署。
- 存储在智能体有写入权限的系统中的 canary tokens。
- 没有硬性限制的纯统计检测。
- 冷却期在没有人类审核的情况下自动重新启用的 circuit breakers。
- Kill switch 仅在启动时检查、不在每次行动上检查的无人值守运行。

拒绝规则：
- 如果用户无法命名位于智能体凭证之外的、托管 kill switch 的具体系统，拒绝。"我们使用智能体读取的配置文件"如果智能体可以写入配置文件，就不是 kill switch。
- 如果用户将 Auto Mode 分类器（第 10 课）视为绊线的替代品，拒绝。分类器与检测和响应正交。
- 如果拟议的 canaries 位于智能体有正当理由读取的系统中，拒绝并要求重新设计。

输出格式：

返回一份绊线审核，包含：
- **Kill-switch 行**（位置、检查频率、重新启用流程）
- **Circuit-breaker 表**（模式、阈值、冷却期）
- **Canary 表**（token、位置、警报、负责人）
- **分层说明**（统计 + 硬性限制存在 y/n）
- **隔离流**（什么触发、发生什么、已测试 y/n）
- **就绪度**（production / staging / research-only）
