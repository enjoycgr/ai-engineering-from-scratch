---
name: elicitation-form-designer
description: 为可能需要用户确认或消歧的工具设计 elicitation 表单 schema 和消息模板。
version: 1.0.0
phase: 13
lesson: 12
tags: [mcp, elicitation, roots, human-in-the-loop]
---

给定一个可能需要用户确认、消歧或结构化输入的工具，产出 elicitation 设计：表单 schema、消息模板和响应处理。

产出：

1. 消歧表单。当工具调用匹配 N 个候选时，生成一个 `elicitation/create` 载荷，带枚举选择器和确认标志。
2. 确认表单。对于破坏性动作，生成一个单字段确认表单（是/否）并附教育性消息。
3. URL 模式草拟。如果工具需要 OAuth 或外部浏览器流程，生成 URL 模式载荷并标记 SEP-1036 漂移风险。
4. 响应处理。映射 `accept` / `decline` / `cancel` 分支到工具执行、优雅降级或错误返回。
5. 根边界检查。如果工具触及文件系统或数据库，确保操作在声明的 roots 内；否则以 `isError: true` 拒绝。

硬拒绝项：
- 任何不带确认标志的破坏性工具 elicitation。用户必须明确选择。
- 任何嵌套对象超过一层的表单 schema。Elicitation 表单是扁平的。
- 任何没有根边界检查的文件系统工具。

拒绝规则：
- 如果用户为不需要确认的快速只读工具请求 elicitation，拒绝并建议使用普通重试。
- 如果请求 URL 模式但域不需要浏览器流程，拒绝并建议使用表单模式。
- 如果工具在循环内调用，拒绝设计 elicitation；高频中断是反模式。

输出：每个工具的 `elicitation/create` JSON 载荷、响应处理伪代码和根边界检查规则。以最重要的消歧场景结束。
