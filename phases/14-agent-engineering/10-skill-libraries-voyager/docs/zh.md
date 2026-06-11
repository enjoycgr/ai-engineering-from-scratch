# 技能库与终身学习 (Voyager)

> Voyager (Wang 等, TMLR 2024) 将可执行代码视为 skill (技能)。Skills 是命名的、可检索的、可组合的，并通过环境反馈精炼。这是 Claude Agent SDK skills、skillkit 以及 2026 年 skill-library (技能库) 模式的参考架构。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 分钟

## 学习目标

- 说出 Voyager 的三个组件 —— automatic curriculum (自动课程)、skill library (技能库)、iterative prompting (迭代提示) —— 以及各自的作用。
- 解释为什么 Voyager 将 action space (动作空间) 设为代码，而非原始命令。
- 用标准库实现一个 skill library：支持注册、检索、组合和失败驱动的 refinement (精炼)。
- 将 Voyager 的模式映射到 2026 年的 Claude Agent SDK skills 和 skillkit 生态。

## 问题背景

每次会话都从头重建所有能力的智能体会做错三件事：

1. **浪费词元。** 每项任务都重复诱发相同的推理。
2. **丢失进度。** 会话 A 中学到的修正无法转移到会话 B。
3. **长程组合失败。** 复杂任务需要能力层级；单条 prompt 无法表达它们。

Voyager 的答案：将每个可复用能力视为一个命名代码块，存储在 library (库) 中，按相似度检索，与其他 skill 组合，并通过执行反馈精炼。

## 核心概念

### 三个组件

Voyager (arXiv:2305.16291) 围绕以下结构构建智能体：

1. **Automatic curriculum (自动课程)。** 一个 curiosity-driven (好奇心驱动) 的提议器，基于智能体当前的技能集和环境状态挑选下一个任务。探索是自底向上的。
2. **Skill library (技能库)。** 每个 skill 是可执行代码。任务成功时添加新 skill。Skills 按 query-to-description similarity (查询到描述的相似度) 检索。
3. **Iterative prompting mechanism (迭代提示机制)。** 失败时，智能体接收执行错误、环境反馈和 self-verification (自验证) 输出，然后精炼 skill。

Minecraft 评估（Wang 等, 2024）：独特物品 3.3 倍、石制工具快 8.5 倍、铁制工具快 6.4 倍、地图遍历远 2.3 倍。数字是 Minecraft 特有的，但模式可迁移。

### Action space = code (动作空间 = 代码)

大多数智能体发出原始命令。Voyager 发出 JavaScript 函数。一个 skill 是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由 sub-skills (子技能) 组合而成。以 description 和 embedding 为键存储。作为程序检索，而非 prompt。

这就是 2026 年的 Claude Agent SDK skill：一个命名的、可检索的代码块加指令，智能体按需加载。

### Skill retrieval (技能检索)

新任务"制作钻石镐"。智能体：

1. 嵌入任务描述。
2. 向 skill library 查询 top-k 相似 skills。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 从检索到的原语 + 新逻辑组合出新 skill。

这就是 MCP resources（第 13 阶段）和 Agent SDK skills 实现的模式：在知识/代码表面上检索，范围限定于当前任务。

### Iterative refinement (迭代精炼)

Voyager 的反馈循环：

1. 智能体编写一个 skill。
2. Skill 针对环境运行。
3. 返回三种信号之一：`success`、`error`（带 stack trace）、`self-verification failure`。
4. 智能体使用信号作为上下文重写 skill。
5. 循环直到成功或达到最大轮数。

这是 Self-Refine（第 05 课）应用于代码生成，带环境锚定的验证。CRITIC（第 05 课）是同一模式，以外部工具作为 verifier。

### 课程与探索

Voyager 的 curriculum module (课程模块) 基于智能体已有什么和尚未做什么，提议诸如"在湖边建一个庇护所"的任务。提议器使用环境状态 + skill inventory 挑选一个略高于当前能力的任务 —— 探索 sweet spot (最佳点)。

对生产级智能体，这转化为一个"缺什么"操作员：给定当前 skill library 和一个领域，我们尚未覆盖哪些 skills？团队通常手动实现这一点，作为 curriculum review。

### 该模式何时出错

- **Skill library rot (技能库腐化)。** 同一 skill 以略有不同的描述被添加 10 次。写入时去重；检索只返回一个。
- **Composed-skill drift (组合技能漂移)。** 父 skill 依赖已被精炼的子 skill。对 skills 做版本控制；钉在 v1 的父 skill 不会 magically pick up v3。
- **Retrieval quality (检索质量)。** 对 skill descriptions 的向量检索在 library 增长到几百以上时退化。辅以 tag filters 和硬约束（"仅 `category=tooling` 的 skills"）。

## 动手实现

`code/main.py` 用标准库实现了 skill library：

- `Skill` —— name、description、code（字符串）、version、tags、dependencies。
- `SkillLibrary` —— register、search（词元重叠）、compose（依赖的拓扑排序）、refine（更新时 bump version）。
- 一个脚本化智能体，注册三个原语 skill，组合第四个，遭遇失败，然后精炼。

运行：

```
python3 code/main.py
```

trace 展示了 library 写入、检索、组合、一次失败执行和一次 v2 精炼 —— Voyager 的端到端循环。

## 如何使用

- **Claude Agent SDK skills**（Anthropic）—— 2026 年参考：每个 skill 有 description、code 和 instructions；在智能体会话期间按需加载。
- **skillkit**（npm: skillkit）—— 面向 32+ AI coding agents 的跨 agent skill 管理。
- **Custom skill libraries** —— 领域专用（数据 agent 的 SQL skills、基础设施 agent 的 Terraform skills）。Voyager 模式可向下缩放。
- **OpenAI Agents SDK `tools`** —— 轻量端；每个 tool 是一个轻量 skill。

## 产物输出

`outputs/skill-skill-library.md` 生成一个 Voyager 风格的 skill library，为任何目标 runtime 内置注册、检索、版本控制和精炼机制。

## 练习

1. 在 `compose()` 中添加 dependency-cycle detector (依赖循环检测器)。当 skill A 依赖 B，B 又依赖 A 时，报错还是警告？
2. 实现 per-skill version pinning (版本锁定)。当父 skill 组合子 skill `crafting@1` 时，对 `crafting@2` 的精炼不得静默升级父 skill。
3. 将词元重叠检索替换为 sentence-transformers embeddings（或 BM25 标准库实现）。在 50-skill 玩具 library 上测量 retrieval@5。
4. 添加一个"curriculum" agent：给定当前 library 和一个领域描述，提议 5 个缺失的 skills。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK skill 文档。将玩具 library 移植到 SDK 的 skill schema。可发现性 (discoverability) 有哪些变化？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Skill | "可复用能力" | 命名代码块 + 描述，可按相似度检索 |
| Skill library | "智能体 how-to 记忆" | 可搜索、可组合的 skills 持久化存储 |
| Curriculum | "任务提议器" | 由当前能力缺口驱动的自底向上目标生成器 |
| Composition | "Skill DAG" | Skills 调用 skills；执行时拓扑排序 |
| Iterative refinement | "自我纠正循环" | 环境反馈 + 错误 + 自验证折叠进下一版本 |
| Action-space-as-code | "程序化动作" | 发出函数而非原始命令，实现时间延展行为 |
| Dedup on write | "技能坍缩" | 近似重复描述坍缩为单一规范 skill |

## 延伸阅读

- [Wang 等, Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) —— 原始技能库论文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) —— 2026 年的产品化
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— 实践中的 skills 和 subagents
- [Madaan 等, Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) —— Voyager 底层的精炼循环
