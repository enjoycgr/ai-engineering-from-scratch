# 浏览器智能体与长时程网页任务

> ChatGPT agent（2025 年 7 月）将 Operator 与 deep research 合并为一个浏览器/终端智能体，并在 BrowseComp 上取得 SOTA 68.9%。OpenAI 于 2025 年 8 月 31 日关闭了 Operator —— 产品层面的整合。Anthropic 收购 Vercept 后，Claude Sonnet 在 OSWorld 上的得分从不到 15% 提升至 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中约 11.3 个百分点的假阴性率，并发布了包含 258 个任务的 Hard 子集。这些数字是真实的。攻击面也是真实的：OpenAI 的准备负责人公开声明，针对浏览器智能体的间接提示注入（indirect prompt injection）"不是一个可以完全修补的 bug"。2025–2026 年记录在案的攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持（one-click hijack）。

**类型：** 学习
**语言：** Python（标准库，间接提示注入攻击面模型）
**前置：** Phase 15 · 10（权限模式），Phase 15 · 01（长时程智能体）
**时间：** ~45 分钟

## 问题

浏览器智能体（browser agent）是一种读取不可信内容并采取 consequential 行动的长时程智能体。智能体访问的每个页面都是用户未亲自撰写的输入。每个页面上的每个表单都是潜在的命令通道。2025–2026 年的攻击库表明这并非假设：Tainted Memories 允许攻击者通过精心构造的页面将恶意指令绑定到智能体的记忆；HashJack 将命令隐藏在智能体访问的 URL 片段中；Perplexity Comet 在一次点击中完成劫持。

防御图景令人不安。OpenAI 的准备负责人说出了那句不愿被公开的话：间接提示注入"不是一个可以完全修补的 bug"。这是因为攻击存在于智能体的读取与行动边界，而该边界在架构上本身就是模糊的 —— 原则上，模型读取的每个 token 都可能被解读为指令。

本课命名了攻击面、基准测试图景（BrowseComp、OSWorld、WebArena-Verified），并建模了一个最小的间接提示注入场景，以便你在第 14 课和第 18 课中推理真实防御措施。

## 概念

### 2026 年各系统一览，每系统一段

**ChatGPT agent（OpenAI）。** 2025 年 7 月推出。统一了 Operator（浏览）与 Deep Research（多小时研究）。2025 年 8 月 31 日关闭了独立 Operator。BrowseComp SOTA 68.9%；OSWorld 和 WebArena-Verified 上也有强劲数据。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 聚焦于计算机使用能力。Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 以工具 API 形式提供。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成提供计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门追踪 ML R&D 领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有据可查的问题：原始 WebArena 有约 11.3% 的假阴性率（实际已解决的任务被标记为失败）。Verified 版本采用人工策划的成功标准重新评分，并增加了 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准测试 | 测量内容 | 时程 |
|---|---|---|
| BrowseComp | 在时间压力下在开放网络上查找特定事实 | 分钟级 |
| OSWorld | 智能体操控完整桌面（鼠标、键盘、shell） | 数十分钟级 |
| WebArena-Verified | 模拟站点中的事务性网页任务 | 分钟级 |
| Hard 子集 | WebArena-Verified 中涉及多页面状态转换的任务 | 数十分钟级 |

不同维度。高 BrowseComp 分数说明智能体能查找事实；并不能说明它能预订航班。OSWorld 分数更接近"它在我的桌面上是否可用"。WebArena-Verified 更接近"它能否完成一个流程"。任何生产决策都需要与任务分布匹配的基准测试。

### 攻击面，逐条命名

1. **Indirect prompt injection（间接提示注入）。** 不可信页面内容包含指令。智能体读取它们。智能体执行它们。公开示例：2024 年 Kai Greshake 等，2025 年 Tainted Memories 论文，2026 年 HashJack（Cato Networks）。
2. **URL fragment / query injection（URL 片段 / 查询注入）。** 爬取 URL 的 `#fragment` 或查询字符串包含命令。对用户不可见；但仍处于智能体的上下文中。
3. **Memory-binding attacks（记忆绑定攻击）。** 页面指示智能体将持久化记忆写入（第 12 课涵盖 durable state）。下一次会话，该记忆在没有可见触发的情况下触发载荷。
4. **CSRF-shaped attacks on authenticated sessions（认证会话上的类 CSRF 攻击）。** Tainted Memories 类：智能体在某处已登录；攻击者的页面向智能体发出状态变更请求，智能体使用用户的 cookie 执行。
5. **One-click hijack（一键劫持）。** 一个视觉上无害的按钮搭载智能体跟随的载荷。Comet 类。
6. **Content-Security-Policy holes in the agent's host surface（智能体宿主表面的 CSP 漏洞）。** 渲染层和工具层本身可能成为攻击向量；浏览器中的浏览器智能体堆栈很宽。

### 为何"无法完全修补"

攻击与智能体的能力同构。智能体必须读取不可信内容才能完成工作。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能与用户实际请求不一致。防御措施（信任边界、分类器、工具白名单、consequential 行动上的人类介入）提高了攻击成本并减小了爆炸半径。它们无法关闭整个攻击类别。

这与 Löb 定理（第 8 课）的推理模式相同：智能体无法证明下一个 token 是安全的；它只能建立一个使不安全 token 更易被检测的系统。

### 实际部署的防御姿态

- **Read / write boundary（读写边界）。** 读取从不产生后果。写入（提交表单、发布内容、调用具有副作用的工具）如果发起内容来自信任边界之外，则需要新的人类批准。
- **Tool allowlist per task（每任务工具白名单）。** 智能体可以浏览；除非该工具被显式启用用于该任务，否则它不能发起电汇。第 13 课涵盖预算。
- **Session isolation（会话隔离）。** 浏览器智能体会话仅以限定凭证运行。没有生产认证，没有个人邮箱。保留每个 HTTP 请求的日志以供审计。
- **Content sanitizer（内容消毒器）。** 在将获取的 HTML 拼接进模型上下文之前，先去除已知的恶意模式。（减少简单攻击；无法阻止复杂载荷。）
- **HITL on consequential actions（consequential 行动上的人类介入）。** Propose-then-commit 模式（第 15 课）。
- **Canary tokens on memory（记忆上的金丝雀令牌）。** 如果一条记忆触发，用户会看到它（第 14 课）。

## 使用

`code/main.py` 对一个微型浏览器智能体在三张合成页面上进行建模。一张页面是良性的，一张在可见文本中包含直接提示注入块，一张包含 URL 片段注入（不可见但在智能体的上下文中）。脚本展示 (a) 天真智能体会做什么，(b) 读写边界能捕获什么，(c) 消毒器能捕获什么，(d) 两者都捕获不了什么。

## 交付

`outputs/skill-browser-agent-trust-boundary.md` 界定一个拟议的浏览器智能体部署：它触及哪些信任区域、被授权写入什么，以及首次运行前必须具备哪些防御。

## 练习

1. 运行 `code/main.py`。确定哪种攻击消毒器能捕获但读写边界不能，以及哪种攻击只有读写边界能捕获。
2. 将消毒器扩展为检测一类 HashJack 风格的 URL 片段注入。在带有合法片段的良性 URL 上测量误报率。
3. 挑选一个你知道的真实浏览器智能体工作流（例如"预订航班"）。列出每次读取和每次写入。标记哪些写入需要 HITL 以及为什么。
4. 阅读 WebArena-Verified ICLR 2026 论文。确定一类原始 WebArena 评分不可靠的任务，并解释 Verified 子集如何解决它。
5. 为浏览器智能体场景设计一个记忆金丝雀。你会存储什么、在哪里存储，以及什么触发警报？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| Indirect prompt injection | "坏页面文本" | 智能体读取的页面中的不可信内容包含智能体执行的指令 |
| Tainted Memories | "记忆攻击" | 智能体将攻击者提供的指令写入 durable memory；下次会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段 / 查询字符串中的载荷处于智能体上下文中但不可见 |
| One-click hijack | "坏按钮" | 可见的 affordance 搭载智能体执行的后续载荷 |
| BrowseComp | "网络搜索基准" | 在开放网络上查找特定事实；分钟级时程 |
| OSWorld | "桌面基准" | 完整操作系统控制；多步 GUI 任务 |
| WebArena-Verified | "修复的网页任务基准" | ServiceNow 重新评分的 WebArena 含 Hard 子集 |
| Read/write boundary | "副作用门" | 读取从不产生后果；如果内容来自信任边界之外，写入需要新批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 与 deep research 的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 谱系及演变为 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准测试。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含计算机使用智能体的攻击面讨论。
