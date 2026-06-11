# 巅峰 —— 构建完整工具生态系统

> Phase 13 教授了每个部分。此巅峰将它们连接成一个生产形状的系统：具备工具 + 资源 + 提示 + 任务 + UI 的 MCP 服务器，边缘 OAuth 2.1，RBAC 网关，多服务器客户端，A2A 子智能体调用，OTel 追踪到收集器，CI 中的工具中毒检测，以及 AGENTS.md + SKILL.md 包。到最后你可以捍卫每个架构选择。

**类型：** Build
**语言：** Python（stdlib，端到端生态系统 harness）
**前置要求：** Phase 13 · 01 到 21
**时间：** ~120 分钟

## 学习目标

- 组合暴露工具、资源、提示和带 `ui://` 应用的任务的 MCP 服务器。
- 用强制执行 RBAC 和固定哈希的 OAuth 2.1 网关为服务器做前端。
- 编写端到端追踪的 OTel GenAI 属性的多服务器客户端。
- 将部分工作负载委托给 A2A 子智能体；验证不透明性是否保留。
- 用 AGENTS.md + SKILL.md 打包整个栈，以便其他智能体可以驱动它。

## 问题

交付 "研究和报告" 系统：

- 用户询问："总结 2026 年关于智能体协议的三个最引用 arXiv 论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要委托给专门的写作智能体；聚合结果；将交互式报告渲染为 MCP Apps `ui://` 资源；记录每一步到 OTel。

Phase 13 的所有原语都出现。这不是玩具——2026 年 Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPTs）和第三方交付的生产研究助手系统具有这种确切的形状。

## 概念

### 架构

```
[用户] -> [客户端] -> [网关 (OAuth 2.1 + RBAC)] -> [研究 MCP 服务器]
                                                      |
                                                      +- MCP 工具: arxiv_search (pure)
                                                      +- MCP 资源: notes://recent
                                                      +- MCP 提示: /research_topic
                                                      +- MCP 任务: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A 调用: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI span
```

### Trace 层次结构

```
agent.invoke_agent
 ├── llm.chat (启动)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task 转换 (不透明内部)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status 轮询
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (最终综合)
```

一个 trace id。每个 span 都有正确的 `gen_ai.*` 属性。

### 安全态势

- 带资源指示器固定受众到网关的 OAuth 2.1 + PKCE。
- 网关持有上游凭证；用户永远看不到它们。
- RBAC：`alice` 拥有 `research:read`、`research:write`，可以调用所有工具。`bob` 拥有 `research:read`，不能调用 `generate_report`。
- 固定描述清单：丢弃任何工具哈希变更的服务器。
- Rule of Two 审计：没有工具组合不受信任输入、敏感数据和后果性动作。

### 渲染

最终 `generate_report` 任务返回内容块加 `ui://report/current` 资源。客户端的宿主（Claude Desktop 等）在沙盒 iframe 中渲染交互式仪表板。仪表板包含排序的论文列表、引用计数和一个按钮，为任何用户点击的论文调用 `host.callTool('summarize_paper', {arxiv_id})`。

### 打包

整个东西作为：

```
research-system/
  AGENTS.md                     # 项目约定
  skills/
    run-research/
      SKILL.md                  # 顶级工作流
  servers/
    research-mcp/               # MCP 服务器
      pyproject.toml
      src/
  agents/
    writer/                     # A2A 智能体
  gateway/
    config.yaml                 # RBAC + 固定清单
```

用户以 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动系统。

### Phase 13 每课的贡献

| 课 | 巅峰使用什么 |
|--------|------------------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、schema、linting |
| 06-10 | MCP 原语、服务器、客户端、传输、资源 + 提示 |
| 11-14 | 采样、roots + elicitation、异步任务、`ui://` 应用 |
| 15-17 | 工具中毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子智能体委托 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 使用它

`code/main.py` 将以前课程的模式连接到一个可运行的演示中。全部标准库，全部进程内，以便你可以从头到尾阅读。它为研究和报告场景运行完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、作为任务的 generate_report、对 writer 的 A2A 调用、返回的 ui:// 资源、发出的 OTel span。

看点：

- 一个 trace id 贯穿每个跳跃。
- 网关策略阻止第二个用户写入。
- 任务生命周期从 working 到 completed，返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器不透明。
- AGENTS.md 和 SKILL.md 是另一个智能体重现工作流所需的唯一文件。

## 交付它

本课产出 `outputs/skill-ecosystem-blueprint.md`。给定产品需求（研究、摘要、自动化），该技能产出完整架构：哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、哪些打包。

## 练习

1. 运行 `code/main.py`。注意单一 trace id 和 span 如何嵌套。计数演示触及多少 Phase 13 原语。

2. 扩展演示：添加第二个后端 MCP 服务器（例如 `bibliography`）并确认网关将其工具合并到同一命名空间。

3. 用子进程中运行的真实智能体替换假 A2A writer 智能体。使用第 19 课的 harness。

4. 在编排器和 LLM 之间的路由网关中添加 PII 编辑步骤。确认用户查询中的电子邮件被清除。

5. 为将维护此系统的队友编写 AGENTS.md。它应该阅读时间不到五分钟，并给他们需要的一切以在 Cursor 或 Codex 中驱动巅峰。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么 |
|------|----------------|------------------------|
| 巅峰 | "Phase-13 集成演示" | 使用每个原语的端到端系统 |
| 研究和报告 | "场景" | 搜索、摘要、渲染模式 |
| 生态系统 | "所有部分在一起" | 服务器 + 客户端 + 网关 + 子智能体 + 遥测 + 包 |
| Trace 层次结构 | "单一 trace id" | 每个跳跃的 span 共享 trace；通过 span id 父子关联 |
| 网关颁发 token | "传递性认证" | 客户端只看到网关的 token；网关持有上游凭证 |
| 合并命名空间 | "一个扁平列表中的所有工具" | 网关处的多服务器合并，前缀-冲突 |
| 不透明性边界 | "A2A 调用隐藏内部" | 子智能体的推理对编排器不可见 |
| 三层栈 | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| 纵深防御 | "多层安全" | 固定哈希、OAuth、RBAC、Rule of Two、审计日志 |
| 规范合规矩阵 | "我们交付的规范要求什么" | 将交付物映射到 2025-11-25 要求的清单 |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) —— 合并参考
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) —— 协议的发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) —— A2A v1.0 参考
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) —— 规范追踪约定
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) —— 生产智能体运行时模式
