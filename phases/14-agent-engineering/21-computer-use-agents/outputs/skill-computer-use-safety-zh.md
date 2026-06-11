---
name: computer-use-safety
description: 为 computer-use agent 构建 per-step safety classifier（逐步安全分类器）+ confirmation gate（确认门），包含 allowlist navigation（允许列表导航）和 injection-marker filtering（注入标记过滤）。
version: 1.0.0
phase: 14
lesson: 21
tags: [computer-use, safety, claude, openai-cua, gemini]
---

给定一个 computer-use agent 和目标应用列表，生成一个 safety layer（安全层），在执行前对每个动作进行分类。

生成：

1. `SafetyClassifier.assess(action, screen) -> SafetyVerdict`，包含字段 `allow`、`reason`、`needs_confirmation`。
2. Allowlist（允许列表）of element labels agent 可以点击；否则拒绝。
3. Allowlist（允许列表）of URLs agent 可以导航到；在 redirect（重定向）出列表时拒绝。
4. Injection-marker filter（注入标记过滤器）on DOM text、retrieved content（检索内容）和 typed text。任何匹配都会阻止动作。
5. Confirmation gate（确认门）for sensitive actions（敏感动作）（login 登录、purchase 购买、delete 删除、publish 发布）。Human-in-the-loop（人机协同）callback interface（回调接口）。
6. Trace emitter（追踪发射器）：每个决策都被记录为 (action, verdict, reason)。

Hard rejects（硬性拒绝）：

- Safety classifier 只在第一个动作上运行。每个动作都必须被分类。
- Allowlist 形式为 `*`。允许一切的 allowlist 不是 allowlist。
- 跳过确认因为模型"看起来很有信心"。Confidence（信心）不是 safety（安全）。

Refusal rules（拒绝规则）：

- 如果 agent 拥有 computer-use 访问权限但没有 per-step safety（逐步安全），拒绝发布。
- 如果 agent 可以导航到任意 URL，拒绝。需要 allowlist 或 blocklist（阻止列表）。
- 如果 sensitive actions（敏感动作）在任意模式下绕过 confirmation gate（确认门），拒绝。

输出：`classifier.py`、`allowlist.py`、`confirmation.py`、`trace.py`、`README.md`，解释 gate policy（门策略）、injection markers（注入标记）和 allowlist maintenance process（允许列表维护流程）。最后以"what to read next"指向 Lesson 27（prompt injection 提示注入）和 Lesson 23（OTel span attribution for safety decisions 安全决策的 OTel span 归因）。
