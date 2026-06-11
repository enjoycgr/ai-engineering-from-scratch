---
name: agent-bundle
description: 为工作流产出可移植的 SKILL.md + AGENTS.md + MCP-server 蓝图，可在 Claude Code、Cursor、Codex 和兼容智能体中加载。
version: 1.0.0
phase: 13
lesson: 22
tags: [skills, agents-md, apps-sdk, cross-agent, portability]
---

给定工作流描述，产出智能体包。

产出：

1. SKILL.md。带 `name` 和 `description` 的 YAML frontmatter，带编号步骤的 markdown 主体。如果主体很长，包含渐进披露子资源引用。
2. AGENTS.md 条目。添加到仓库 AGENTS.md 的几行，反映技能依赖的任何约定（linter 命令、测试命令）。
3. MCP 服务器蓝图。技能通过 MCP 调用的工具；名称、描述（Use-when 模式）和输入 schema。
4. 跨智能体翻译。SkillKit 风格的注释，说明此 SKILL.md 如何映射到 Cursor 规则、Codex `.codex.md`、Windsurf 规则。
5. 加载路径。智能体将在何处发现此包：`~/.anthropic/skills/`、`./skills/`、`~/.claude/skills/`。

硬拒绝项：
- 任何 `name` 不是 `kebab-case` 的 SKILL.md。破坏发现。
- 任何 frontmatter 中没有 `description` 的 SKILL.md。智能体运行时跳过它。
- 任何其 MCP 工具未按 Phase 13 · 05 规则命名的包。

拒绝规则：
- 如果工作流是单次提示，拒绝产出技能；推荐内联提示工程。
- 如果工作流需要 OAuth（例如 Slack 发布），标记 MCP 服务器的首次运行 elicitation 必须处理它。
- 如果目标智能体不支持 SKILL.md（某些 IDE），推荐通过 SkillKit 或类似工具翻译。

输出：一份一页包，包含三个文件的草图、跨智能体翻译注释和加载路径。以首先测试该包的单一智能体结束。
