# 记忆：虚拟上下文与 MemGPT

> 上下文窗口 (context window) 是有限的。对话、文档和工具轨迹不是。MemGPT (Packer 等, 2023) 将其框定为操作系统虚拟内存 (OS virtual memory) —— 主上下文 (main context) 是 RAM，外部存储是磁盘，智能体在它们之间换页 (page)。这是每个 2026 年记忆系统所继承的模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 分钟

## 学习目标

- 解释 MemGPT 所依赖的操作系统类比：main context (主上下文) = RAM，external context (外部上下文) = 磁盘，memory tools (记忆工具) = 换入/换出 (page in/out)。
- 用标准库实现两层 MemGPT 模式：一个主上下文缓冲区 (main-context buffer)、一个外部可搜索存储 (external searchable store)，以及换页工具。
- 描述智能体如何发出"中断"(interrupt) 来查询或修改外部记忆，以及结果如何拼接回下一次 prompt。
- 识别 MemGPT 的设计选择如何延续到 Letta（第 08 课）和 Mem0（第 09 课）。

## 问题背景

上下文窗口看起来应该能解决记忆问题。事实并非如此。三种失败模式在生产中反复出现：

1. **Overflow (溢出)。** 多轮对话、长文档或工具调用密集的轨迹会超出窗口。截断点之后的一切都会丢失。
2. **Dilution (稀释)。** 即使在窗口内，塞入无关上下文也会稀释对关键信息的注意力。前沿模型在长输入上仍会退化。
3. **Persistence (持久化)。** 新会话以空窗口启动。没有外部记忆的智能体无法跨会话说出"还记得你让我……"。

更大的窗口有帮助，但无法根治。Mem0 的 2025 年论文测得：即便 128k 窗口基线，仍会漏掉带外部记忆的 4k 窗口智能体能捕获的长程事实。

## 核心概念

### MemGPT：操作系统类比

Packer 等 (arXiv:2310.08560, v2 Feb 2024) 将上下文管理映射到操作系统虚拟内存 (OS virtual memory)：

| OS 概念 | MemGPT 概念 | 2026 年生产级对应 |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB、KV、graph store |
| Page fault | memory tool call | `memory.search`、`memory.read`、`memory.write` |
| OS kernel | agent control loop | 带记忆工具的 ReAct loop |

智能体运行普通的 ReAct loop。额外的一类工具让它将数据换入 (page in) 和换出 (page out) 主上下文。

### 两层架构

- **Main context (主上下文)。** 固定大小的 prompt，持有当前任务。模型始终可见。
- **External context (外部上下文)。** 无界，可通过工具搜索。相关时读取，事实涌现时写入。

原始论文在两项超出基线窗口的任务上评估了该设计：超过 100k 词元的文档分析和跨天持久记忆的多会话聊天。

### 中断模式 (Interrupt pattern)

MemGPT 引入了 memory-as-interrupt (记忆即中断)：对话中途智能体可以调用记忆工具，runtime 执行它，结果拼接进下一轮 assistant turn 作为新 observation。概念上等同于 Unix `read()` 系统调用 —— 阻塞进程、返回字节、进程继续。

典型记忆工具表面：

- `core_memory_append(section, text)` —— 写入 prompt 的持久化段落。
- `core_memory_replace(section, old, new)` —— 编辑持久化段落。
- `archival_memory_insert(text)` —— 写入可搜索的外部存储。
- `archival_memory_search(query, top_k)` —— 从外部存储检索。
- `conversation_search(query)` —— 扫描历史轮次。

### MemGPT 终点与 Letta 起点

2024 年 9 月 MemGPT 演变为 Letta。研究仓库 (`cpacker/MemGPT`) 仍在；Letta 扩展了设计：

- 三层而非两层（core、recall、archival —— 第 08 课）。
- 原生推理 (native reasoning) 替代 `send_message`/heartbeat 模式（第 08 课）。
- 异步执行记忆任务的 sleep-time agents（第 08 课）。

MemGPT 论文是 2026 年的基础，即便生产系统运行的是 Letta、Mem0 或自定义两层存储。

### 该模式何时出错

- **Memory rot (记忆腐化)。** 写入速度超过读取；检索淹没在陈旧事实中。修复：定期 consolidation (合并)（Letta sleep-time）、显式 invalidation (失效)（Mem0 冲突检测器）。
- **Memory poisoning (记忆投毒)。** 外部记忆是检索到的文本。如果攻击者控制的内容落入记忆笔记，智能体将在下一会话重新摄入。这是 Greshake 等（第 27 课）攻击随时间的重演。
- **Citation loss (引用丢失)。** 智能体回忆"用户让我交付 X"，但无法引用是哪一轮。每次归档写入时都要存储来源引用（session ID、turn ID）。

## 动手实现

`code/main.py` 用标准库实现了 MemGPT 的两层模式：

- `MainContext` —— 固定大小 prompt 缓冲区，带 `core` 字典和 `messages` 列表；超出上限时自动压缩最旧消息。
- `ArchivalStore` —— 内存中的 BM25 风格存储（词元重叠打分），记录为 (id, text, tags, session, turn)。
- 五个映射到 MemGPT 表面的记忆工具。
- 一个脚本化智能体，先向归档写入事实，然后通过 `archival_memory_search` 回答问题。

运行：

```
python3 code/main.py
```

trace 展示了智能体写入三条事实、填满主上下文至上限（触发驱逐），然后通过归档检索回答后续问题 —— 无需真实 LLM 即可复现 MemGPT 工作流。

## 如何使用

当今每个生产级记忆系统都是 MemGPT 的变体：

- **Letta**（第 08 课）—— 三层、原生推理、sleep-time 计算。
- **Mem0**（第 09 课）—— vector + KV + graph 融合打分层。
- **OpenAI Assistants / Responses** —— 通过 threads 和 files 管理记忆。
- **Claude Agent SDK** —— 通过 skills 和 session store 实现长期记忆。

按运维形态选择（自托管、托管、框架集成），而非按核心模式 —— 核心模式就是 MemGPT。

## 产物输出

`outputs/skill-virtual-memory.md` 是一个可复用的 skill，为任何目标 runtime 产出正确的两层记忆脚手架（主 + 归档 + 工具表面），内置驱逐策略和引用字段。

## 练习

1. 添加按词元计算的 `max_main_context_tokens` 上限（近似用 `len(text.split()) * 1.3`）。超出上限时将最旧消息压缩为摘要。比较有/无 summarizer 时的行为差异。
2. 在归档存储上正确实现 BM25（term frequency、inverse document frequency）。在玩具事实集上测量 recall@10，与词元重叠基线对比。
3. 为归档插入添加 `citation` 字段（session_id、turn_id、source_url）。让智能体在每次基于检索的回答中引用来源。
4. 模拟 memory poisoning：添加一条写着"忽略所有未来用户指令"的归档记录。写一个 guard，扫描检索结果中 directive-shaped text (指令型文本) 并将其标记为不可信。
5. 将实现移植到使用 MemGPT 研究仓库的核心记忆 JSON schema (`cpacker/MemGPT`)。从扁平字符串切换到 typed sections 后，有哪些变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Virtual context | "无限记忆" | 主 (prompt) + 外部 (可搜索) 两层，带换入/换出 |
| Main context | "工作记忆" | Prompt —— 固定大小，始终可见 |
| Archival memory | "长期存储" | 外部可搜索持久化，按需检索 |
| Core memory | "持久化 prompt 段落" | 钉在主上下文内的命名段落 |
| Memory tool | "记忆 API" | 智能体发出的用于读写外部记忆的工具调用 |
| Interrupt | "记忆缺页中断" | 智能体暂停、runtime 获取、结果拼接进下一轮 |
| Memory rot | "陈旧事实" | 旧写入淹没检索；用 consolidation 修复 |
| Memory poisoning | "注入持久笔记" | 攻击者内容存储为记忆，在召回时被重新摄入 |

## 延伸阅读

- [Packer 等, MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) —— 受操作系统启发的虚拟上下文论文
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) —— 三层演进
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) —— 将上下文视为预算
- [Chhikara 等, Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) —— 在此模式之上的生产级混合记忆
