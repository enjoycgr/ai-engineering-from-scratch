# 记忆块与睡眠时计算 (Letta)

> MemGPT 于 2024 年演变为 Letta。2026 年的演进新增了两个理念：智能体可直接编辑的离散功能性 memory blocks (记忆块)，以及在主智能体空闲时异步 consolidation (合并) 记忆的 sleep-time agent (睡眠时智能体)。这就是将记忆扩展至单轮对话之外的方式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 分钟

## 学习目标

- 说出 Letta 使用的三层记忆（core、recall、archival）以及每层的作用。
- 解释 memory-block (记忆块) 模式：Human block、Persona block 以及用户自定义块作为一等 typed object。
- 描述什么是 sleep-time compute (睡眠时计算)，为什么它位于 critical path (关键路径) 之外，以及为什么它可以运行比主智能体更强的模型。
- 实现一个脚本化的双智能体循环：主智能体响应用户，sleep-time agent 在轮次间合并块。

## 问题背景

MemGPT（第 07 课）解决了虚拟内存控制流。三个生产问题随之浮现：

1. **Latency (延迟)。** 每次记忆操作都在 critical path (关键路径) 上。如果智能体必须在用户等待时修剪、摘要或调和，尾部延迟会爆炸。
2. **Memory rot (记忆腐化)。** 写入不断累积。被否定的事实仍留在那里。检索淹没在陈旧内容中。
3. **Structure loss (结构丢失)。** 扁平的归档存储无法表达"Human block 始终在 prompt 中；Persona block 始终在 prompt 中；Task block 按会话切换"。

Letta (letta.com) 是 2026 年的重写。Memory blocks 使结构显式化；sleep-time compute 将 consolidation 移出 critical path。

## 核心概念

### 三层架构

| 层级 | 范围 | 存储位置 | 写入者 |
|------|-------|----------------|------------|
| Core (核心) | 始终可见 | 主 prompt 内部 | Agent tool call + sleep-time 重写 |
| Recall (回忆) | 对话历史 | 可检索 | 自动轮次日志 |
| Archival (归档) | 任意事实 | Vector + KV + graph | Agent tool call + sleep-time 摄入 |

Core 是 MemGPT 的核心。Recall 是带驱逐尾巴的对话缓冲区。Archival 是外部存储。这一拆分清理了 MemGPT 两层架构的过度负载。

### Memory blocks (记忆块)

Block 是 core 层级中一个 typed、persistent、可编辑的段落。原始 MemGPT 论文定义了两个：

- **Human block** —— 关于用户的事实（姓名、角色、偏好、目标）。
- **Persona block** —— 智能体的自我概念（身份、语气、约束）。

Letta 泛化为任意用户自定义块：用于当前目标的 `Task` 块、用于代码库事实的 `Project` 块、用于硬约束的 `Safety` 块。每个块有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道何时编辑它）。

块可通过工具表面编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` —— 当块接近上限时进行压缩。

### Sleep-time compute (睡眠时计算)

2025 年 Letta 新增：在后台运行第二个智能体，位于 critical path 之外。Sleep-time agents 处理对话转录和代码库上下文，将 `learned_context` 写入共享块，并合并或失效归档记录。

衍生的属性：

- **无延迟成本。** 主响应无需等待记忆操作。
- **允许更强的模型。** Sleep-time agent 可以使用更昂贵、更慢的模型，因为它不受延迟约束。
- **自然的合并窗口。** 当用户不等待时，去重、摘要、失效矛盾事实。

这一形态契合人类工作方式：你完成任务，睡一觉，长期记忆在夜间沉淀。

### Letta V1 与原生推理 (Native reasoning)

Letta V1 (`letta_v1_agent`, 2026) 弃用了 `send_message`/heartbeat 和内联 `Thought:` 词元，转而采用 native reasoning (原生推理)。Responses API（OpenAI）和带 extended thinking (扩展思考) 的 Messages API（Anthropic）在独立通道上发出推理，跨轮次传递（生产环境中跨供应商加密）。控制循环仍是 ReAct。思维轨迹是结构性的，而非 prompt 形状。

### 该模式何时出错

- **Block bloat (块膨胀)。** 无限 `block_append` 很快触及上限。在写入即将超出上限前，接入 block summarizer。
- **Silent drift (静默漂移)。** Sleep-time agent 重写了一个块，而主智能体从未察觉。对块进行版本控制，并在 trace 中展示 diff。
- **Poisoned consolidation (投毒式合并)。** Sleep-time agent 将攻击者可触及的内容处理进 core。第 27 课同样适用于 sleep-time 表面。

## 动手实现

`code/main.py` 实现了：

- `Block` —— id、label、value、limit、description。
- `BlockStore` —— CRUD + `near_limit(label)` 辅助函数。
- 两个脚本化智能体 —— `PrimaryAgent` 服务一轮，`SleepTimeAgent` 在轮次间合并。
- 一个 trace，展示三轮对话中的块写入，以及一次 sleep-time pass：摘要一个块并失效一个陈旧事实。

运行：

```
python3 code/main.py
```

transcript 展示了拆分：主轮次快速产生原始写入；sleep pass 压缩并清理。

## 如何使用

- **Letta** (letta.com) 作为参考实现。自托管或托管云。
- **Claude Agent SDK skills** 作为块形状的知识 —— skill 是一个命名、版本化、可按需检索的指令块，由智能体加载。
- **Custom builds** 适用于希望控制存储后端的团队。使用 Letta API 契约，以便日后迁移。

## 产物输出

`outputs/skill-memory-blocks.md` 生成一个 Letta 风格的块系统，带 sleep-time hooks，适用于任何 runtime，包含安全规则和引用布线。

## 练习

1. 添加 `block_summarize` 工具，当 `near_limit` 返回 true 时，用模型生成的摘要替换块值。哪个触发阈值能同时最小化 summarization 调用次数和块溢出？
2. 在归档上实现 sleep-time dedup：两个文本词元重叠 >90% 的记录合并为一个。仅在 sleep pass 中执行，绝不在 critical path 上。
3. 对块进行版本控制。每次写入记录旧值和 diff。暴露 `block_history(label)`，以便运维人员调试"智能体为什么忘记了 X"。
4. 将 sleep-time agents 视为不可信写入者。当它们触及 Persona 或 Safety 块时，要求第二个智能体审查后才能提交。
5. 将示例移植到使用 Letta API (`letta_v1_agent`)。块模式有哪些变化，native reasoning 如何改变 trace 形态？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Memory block | "可编辑 prompt 段落" | core 内存中 typed、persistent、LLM 可编辑的段 |
| Human block | "用户记忆" | 关于用户的事实，钉在 core 中 |
| Persona block | "智能体身份" | 自我概念、语气、约束，钉在 core 中 |
| Sleep-time compute | "异步记忆工作" | 第二个智能体在 critical path 外做合并 |
| Core / Recall / Archival | "层级" | 三层记忆拆分：始终可见 / 对话 / 外部 |
| Block limit | "上限" | 每块字符上限；强制摘要 |
| Native reasoning | "思考通道" | 供应商级别的推理输出，不是 prompt 级别的 `Thought:` |
| Learned context | "睡眠输出" | Sleep-time agent 写入共享块的事实 |

## 延伸阅读

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) —— 块模式
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) —— 异步合并
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) —— 原生推理重写
- [Packer 等, MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) —— 起源
