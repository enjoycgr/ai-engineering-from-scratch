---
name: primitive-splitter
description: 将 MCP 服务器草稿中的每个能力分类为工具、资源或提示并给出理由。
version: 1.0.0
phase: 13
lesson: 10
tags: [mcp, primitives, resources, prompts]
---

给定一个提议的 MCP 服务器的能力（以纯英文或草稿工具列表形式），将每个能力分类为工具、资源或提示，并附一句理由。

产出：

1. 每个能力分类。对于每个条目，返回 `{name, primitive: tool | resource | prompt, rationale}`。
2. 资源 URI scheme。如果任何能力成为资源，提议一个 URI scheme（`notes://`、`gh://`、`db://`）和一个模板模式。
3. 提示参数骨架。如果任何能力成为提示，提议参数列表和 required/optional 标志。
4. 订阅候选。标记经常变更且会从 `resources/subscribe` 受益的资源。
5. 反模式标记。指出旧设计将读取包装在工具中的情况（例如 `notes_read(id)`），而资源会服务得更好。

硬拒绝项：
- 任何被分类为 "既是工具又是资源" 而无拆分的能力。选一个或搭建一对。
- 任何没有识别出 required 参数的提示。在斜杠命令 UI 中展示需要参数 schema。
- 任何不可寻址的资源 URI scheme（自由格式字符串，而非 URI）。

拒绝规则：
- 如果所有能力都归为工具，拒绝并询问服务器是否有可成为资源的只读数据。
- 如果没有能力适合提示，那没关系；提示是可选的。不要发明它们。
- 如果服务器的域更适合 A2A（智能体到智能体协作、不透明状态），拒绝并重定向到 Phase 13 · 19。

输出：一份一页决策报告，包含分类表、URI scheme 提议、提示骨架和订阅标志。以该服务器最具影响力的 工具 -> 资源 转换结束。
