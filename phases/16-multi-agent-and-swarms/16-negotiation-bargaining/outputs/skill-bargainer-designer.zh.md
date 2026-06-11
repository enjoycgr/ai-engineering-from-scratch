---
name: bargainer-designer
description: 设计negotiation protocol (协商协议)：哪个agent narrates (叙述)、哪个component generates offers (生成报价)、private scratchpads (私有草稿本)如何与public messages (公共消息)分离、round bound (轮次上限)是什么、以及如何监控deal rate (成交率)。
version: 1.0.0
phase: 16
lesson: 16
tags: [multi-agent, negotiation, bargaining, contract-net, OG-Narrator]
---

给定一个negotiation (协商)或task-market (任务市场)场景（two-party bargain (双方议价)、N-party auction (N方拍卖)、contract-net broadcast (合同网广播)），设计protocol (协议)。

生成内容：

1. **Mechanism (机制)。** Two-party bargain (双方议价)、N-bidder auction (N方竞标)、contract-net broadcast (合同网广播)或multi-party coalition (多方联盟)。命名game (博弈)。
2. **Offer generator (报价生成器)。** Deterministic (确定性的)（Zeuthen-style concession (Zeuthen式让步)、Rubinstein equilibrium (Rubinstein均衡)、simple linear schedule (简单线性时间表)）或LLM-prompted (LLM提示)。默认：deterministic (确定性的)，除非offer必须是qualitative structure (定性结构)（proposal (提议)、role assignment (角色分配)）。
3. **Narration layer (叙述层)。** LLM贡献什么：human-friendly framing (人性化框架)、persuasion tactics (说服策略)、persona (人设)。明确说明LLM不决定什么。
4. **Private vs public channels (私有与公共通道)。** Reasoning traces (推理痕迹)如何保持在counterpart's context (对方上下文)之外。"Private scratchpad (私有草稿本)" + "public message (公共消息)"作为两个字段。根据arXiv:2503.06416，这是non-negotiable (不可协商的)。
5. **Round bound (轮次上限)。** Two-party (双方)最多3-5轮。Unbounded (无界)不是选项；它奖励conformity (从众)并鼓励emotional offers (情绪化报价)。
6. **Reservation and BATNA discipline (保留价和BATNA纪律)。** 双方必须知道他们的reservation price (保留价)。如果对方probe (试探)，LLM narrator不得泄露它。根据此规则验证每条outgoing message (发出的消息)。
7. **Deal-rate monitoring (成交率监控)。** 该protocol的预期baseline deal rate (基线成交率)（从negotiation benchmarks (协商基准)引用一个数字：根据LLM role在27%-89%范围内）。Regression (退化)的alert threshold (告警阈值)。
8. **Escalation (升级)。** Below-threshold rounds (低于阈值的轮次)、ZOPA violations (ZOPA违规)或counterpart-side rule-breaking (对方违规)路由到mediator agent (调解智能体)或human (人类)。

Hard rejects (硬性拒绝)：

- 任何LLM在没有deterministic fallback (确定性回退)的情况下计算numerical offer (数字报价)的设计。arXiv:2402.15813显示这产生约27%的deal rates (成交率)。
- 任何没有separate private and public channels (独立私有和公共通道)的设计。Counterparts会读取你的推理。
- 任何具有unbounded rounds (无界轮次)的设计。保证conformity-driven outcomes (从众驱动结果)。
- 让single agent同时持有buyer and seller state (买方和卖方状态)的设计（roleplay bargaining (角色扮演议价)）。Private-information property (私有信息属性)是mechanism (机制)；合并角色会移除它。

Refusal rules (拒绝规则)：

- 如果任务没有numerical payoff (数值收益)（qualitative negotiation (定性协商)、contract terms (合同条款)），OG-Narrator decomposition可能不适用。推荐structured proposal + schema validation (结构化提议+模式验证)。
- 如果用户无法实现separate scratchpad (独立草稿本)（single-LLM-call architecture (单LLM调用架构)），明确标记leak risk (泄露风险)并推荐two-call architecture (双调用架构)。
- 如果negotiation是adversarial (对抗性的)且对方可能lie (撒谎)，推荐mediator agent plus logged offers for audit (调解智能体+记录报价以供审计)。

Output (输出)：一页brief (简报)。以single-sentence summary (单句摘要)开头（"Two-party bargain: Zeuthen offer generator + LLM narrator, 5-round bound, separate scratchpad, deal-rate alert below 85%."），然后是上述八个部分。以sample message (示例消息)收尾：counterpart看到什么 vs private scratchpad (私有草稿本)持有什么。
