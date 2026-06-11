---
name: injection-defense
description: 为任何智能体运行时构建 PVE（Prompt-Validator-Executor）层，包含来源标记内容、注入标记扫描和白名单导航。
version: 1.0.0
phase: 14
lesson: 27
tags: [security, prompt-injection, pve, greshake, source-tag]
---

给定一个具有工具访问和检索能力的智能体，为其生成注入防御层。

产出：

1. 每条内容的来源标签：`user_message`、`tool_output`、`retrieved_web`、`retrieved_memory`、`retrieved_file`。将标签传播到消息历史。
2. `Validator.assess(tool_call, contents)` —— 拒绝参数或检索内容呈注入形态的工具调用；仅当来源标签匹配声明的信任级别时才允许。
3. 导航白名单 / 黑名单：智能体可以访问的 URL、域名、文件路径。
4. 记忆写入护栏：拒绝看起来像指令的写入。
5. 内容捕获规范（Lesson 23）：将检索内容存储在外部；跨轮次携带引用 ID 而非原文；事故可审计。
6. 测试套件：将 Greshake 的五种漏洞类别作为红队测试用例。

硬性拒绝：

- 没有来源标签的工具使用表面。没有来源就无法区分权限级别。
- 仅在最终输出上运行的验证器。晚期验证无关——模型已经行动了。
- "相信我，系统提示能搞定。" 系统提示卫生不是控制措施。

拒绝规则：

- 如果智能体有任何检索能力但没有来源标记，拒绝发布。检索内容是典型的注入向量。
- 如果敏感工具（发送消息、执行 shell、写入 / 下文件）没有人力确认回路，拒绝。
- 如果记忆写入无护栏，拒绝。Persistent memory poisoning 会在下一轮会话中再中毒。

输出：`validator.py`、`source_tag.py`、`allowlist.py`、`memory_guard.py`、`red_team.py`、`README.md` 解释六层控制栈、残余风险和持续审查节奏。结尾附上"接下来读什么"，指向 Lesson 21（计算机使用安全）和 Lesson 23（通过 OTel 进行内容捕获）。
