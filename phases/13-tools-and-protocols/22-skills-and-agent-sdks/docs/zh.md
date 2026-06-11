# 技能和智能体 SDK —— Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说 "存在什么工具。" 技能说 "如何做任务。" 2026 年栈同时层叠两者。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以渐进披露交付 SKILL.md。OpenAI 的 Apps SDK 是 MCP 加小部件元数据。AGENTS.md（现已在 60,000+ 仓库中）位于仓库根作为项目级智能体上下文。本课命名每个覆盖的内容并构建一个跨智能体工作的最小 SKILL.md + AGENTS.md 包。

**类型：** Learn
**语言：** Python（stdlib，SKILL.md 解析器和加载器）
**前置要求：** Phase 13 · 07（MCP 服务器）
**时间：** ~45 分钟

## 学习目标

- 区分三层：AGENTS.md（项目上下文）、SKILL.md（可复用 know-how）、MCP（工具）。
- 编写带 YAML frontmatter 和渐进披露的 SKILL.md。
- 将技能文件系统风格加载到智能体运行时。
- 将技能与 MCP 服务器和 AGENTS.md 组合，使一个包在 Claude Code、Cursor 和 Codex 中工作。

## 问题

工程师将一个发布说明撰写工作流提炼为多步骤提示："读取最新合并的 PR。按区域分组。总结每个。按照团队风格撰写 changelog 条目。发布到 Slack 草稿。" 他们将其放入团队的 Notion 文档中。

现在他们想要从 Claude Code、Cursor 和 Codex CLI 中使用此工作流。每个智能体有不同的加载指令方式：Claude Code 斜杠命令、Cursor 规则、Codex `.codex.md`。工程师复制工作流三次并维护三个副本。

AGENTS.md 和 SKILL.md 一起修复了这一点：

- **AGENTS.md** 位于仓库根。每个兼容智能体在会话开始时读取它。"这个项目如何工作？约定是什么？哪些命令运行测试？"
- **SKILL.md** 是可移植包：YAML frontmatter（名称、描述）+ markdown 主体 + 可选资源。支持技能的智能体按名称按需加载它们。
- **MCP**（Phase 13 · 06-14）处理技能需要调用的工具。

三层，一个可移植工件。

## 概念

### AGENTS.md（agents.md）

2025 年末推出，到 2026 年 4 月被 60,000+ 仓库采用。仓库根中的一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

智能体在会话开始时读取此文件，并用于校准其在该项目上的行为。2026 年的每个编码智能体都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（作为开放标准于 2025 年 12 月发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter 声明技能的身份。主体是技能加载时展示给模型的提示。

### 渐进披露

技能可以引用智能体仅在需要时才获取的子资源。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说 "查看 style-guide.md 了解样式规则。" 智能体仅在技能主动运行时拉取 style-guide.md。这避免了用模型可能不需要的细节膨胀提示。

### 文件系统发现

智能体运行时扫描已知目录中的 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- Project `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

加载通过文件夹名称和 frontmatter `name` 进行。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨智能体）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载技能，将它们作为运行时内部可调用的 "智能体" 暴露。智能体循环在用户调用时将任务分发给技能。

### OpenAI Apps SDK

2025 年 10 月推出；直接构建在 MCP 上。统一 OpenAI 之前的 Connectors 和 Custom GPT Actions 到一个开发者表面。一个 Apps SDK 应用是：

- 一个 MCP 服务器（工具、资源、提示）。
- 加上 ChatGPT UI 的小部件元数据。
- 加上可选的 MCP Apps `ui://` 资源用于交互式表面。

相同协议，更丰富的 UX。

### 通过 SkillKit 的跨智能体可移植性

SkillKit 和类似的跨智能体分发层将单个 SKILL.md 转换为 32+ AI 智能体（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）中每个的原生格式。一个真实来源；许多消费者。

### 三层栈

| 层 | 文件 | 何时加载 | 目的 |
|-------|------|-------------|---------|
| AGENTS.md | 仓库根 | 会话开始 | 项目级约定 |
| SKILL.md | 技能目录 | 技能调用时 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用动作 |

三者组合：智能体在会话开始时读取 AGENTS.md，用户调用技能，技能的指令包含 MCP 工具调用，智能体通过 MCP 客户端分发。

## 使用它

`code/main.py` 交付了一个 stdlib SKILL.md 解析器和加载器。它在 `./skills/` 下发现技能，解析 YAML frontmatter 加 markdown 主体，并生成按技能名称键入的 dict。然后它模拟一个按名称调用 `release-notes-writer` 的智能体循环。

看点：

- YAML frontmatter 用最小 stdlib 解析器解析（无 `pyyaml` 依赖）。
- 技能主体逐字存储；智能体在调用时将其前置到系统提示。
- 通过 `read_subresource` 函数演示渐进披露，按需拉取引用文件。

## 交付它

本课产出 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能产出组合的 SKILL.md + AGENTS.md + MCP-server-blueprint 包，跨智能体可移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能并确认加载器拾取它。

2. 为此课程仓库编写 AGENTS.md。包含测试命令、样式约定和 Phase 13 心智模型。

3. 将你团队内部文档中的多步骤工作流移植到 SKILL.md。验证它在 Claude Code 中加载。

4. 手工将技能转换为 Cursor 和 Codex 的原生规则格式。计数格式之间的差异——这是 SkillKit 自动化的翻译表面。

5. 阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 有此课加载器未覆盖的一个功能。（提示：智能体子调用。）

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| SKILL.md | "技能文件" | YAML frontmatter 加 markdown 主体，由智能体运行时加载 |
| AGENTS.md | "仓库根智能体上下文" | 会话开始时读取的项目级约定文件 |
| 渐进披露 | "延迟加载子资源" | 技能主体引用仅在需要时才拉取的文件 |
| Frontmatter | "顶部 YAML 块" | `---` 分隔符中的元数据（名称、描述） |
| Claude Agent SDK | "Anthropic 的技能运行时" | `@anthropic-ai/claude-agent-sdk`，加载技能并路由 |
| OpenAI Apps SDK | "MCP + 小部件元" | 构建在 MCP 加上 ChatGPT UI 钩子的 OpenAI 开发者表面 |
| 技能发现 | "文件系统扫描" | 遍历已知目录查找 SKILL.md，按名称键入 |
| 跨智能体可移植性 | "一个技能多个智能体" | 通过 SkillKit 风格工具将一个 SKILL.md 转换为 32+ 智能体 |
| Agent Skill | "可移植 know-how" | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | "MCP 加 ChatGPT UI" | 在 MCP 上统一的 Connectors 和 Custom GPTs |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) —— 2025 年 12 月发布
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) —— SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) —— ChatGPT 的基于 MCP 的开发者平台
- [agents.md](https://agents.md/) —— AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) —— 官方技能示例
