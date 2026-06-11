# 多模态 Agent 与计算机使用（顶点）

> 2026 年前沿产品是一个多模态 agent，读取截屏、点击按钮、导航网页 UI、填写表单、端到端完成工作流。SeeClick 和 CogAgent (2024) 证明了 GUI 定位原语。Ferret-UI 添加了移动端。ChartAgent 引入了图表的视觉工具使用。VisualWebArena 和 AgentVista (2026) 是前沿追逐的基准——甚至 Gemini 3 Pro 和 Claude Opus 4.7 在 AgentVista 困难任务上也只有 ~30%。本顶点汇集 Phase 12 的每条线：感知（高分辨率 VLM）、推理（带工具使用的 LLM）、定位（坐标输出）、长程记忆和评估。

**类型：** 顶点
**语言：** Python (stdlib, action schema + agent loop skeleton)
**前置知识：** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**时间：** ~240 分钟

## 学习目标

- 设计多模态 agent 循环：感知 → 推理 → 行动 → 观察 → 重复。
- 构建 VLM 可作为 JSON 发出的 GUI 定位输出 schema（点击坐标、输入文本、滚动、拖拽）。
- 对比纯截屏 agent vs 无障碍树 agent vs 混合 agent。
- 在小型 VisualWebArena 切片上设置多模态 agent 基准评估。

## 问题

预订网站工作流："给我找一趟 4 月 15 日去东京的航班，过道座位，低于 800 美元，预订它。"

多模态 agent 需要：

1. 截取浏览器截屏。
2. 将截屏 + URL + 目标解析为计划。
3. 发出结构化动作：在 (x,y) 点击、在元素 E 输入"Tokyo"、向下滚动、选择（单选按钮）。
4. 将动作应用于浏览器。
5. 观察新状态（下一个截屏）。
6. 重复直到任务完成。

每步是多模态 VLM 调用。VLM 输出必须是可解析 JSON。错误跨步累积，因此恢复重要。

## 概念

### GUI 定位——原语

GUI 定位是：给定截屏和自然语言指令，输出点击的 (x, y) 坐标（或其他动作）。

SeeClick (arXiv:2401.10935) 是首批开放大规模结果：在合成 + 真实 GUI 数据上微调 VLM，将坐标作为纯文本 token 输出。有效。

CogAgent (arXiv:2312.08914) 为密集 UI 添加 1120x1120 高分辨率编码。分数：网页导航 ~84%。

Ferret-UI (arXiv:2404.05719) 聚焦移动端 UI，整合 iOS 无障碍数据。

输出格式通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 帮助恢复：如果坐标在截屏间漂移，语义提示让系统重新定位。

### 动作 schema

典型动作 schema 有 6-10 种动作类型：

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent 每步发出一个动作。浏览器包装器执行并返回新状态。

### 纯截屏 vs 无障碍树

两种输入模式：

- 纯截屏：完整图像，无结构信息。最通用；任何应用都工作。
- 无障碍树：结构化 DOM / iOS 无障碍信息。定位可靠得多；树可用的地方工作。
- 混合：两者，树作为可靠原子动作的定位器，截屏用于语义上下文。

生产 agent 尽可能使用混合。浏览器自动化（Selenium + 无障碍）总是有树；桌面应用有时有。

### 长程记忆

20 步工作流生成 20 个截屏。VLM 上下文快速填满。三种压缩策略：

- 摘要链：每 5 步后总结发生了什么，丢弃旧截屏。
- 跳帧：保留第一个、最后一个和每第 3 个截屏。
- 工具记录日志：执行动作，保留所做文本日志；不重新查看旧截屏。

Claude 的 computer-use API 使用日志模式。更简单，更可靠。

### 视觉工具使用

ChartAgent (arXiv:2510.04514) 引入图表理解的视觉工具使用：裁剪、缩放、OCR、调用外部检测。Agent 可以发出"裁剪到区域 (100, 200, 300, 400) 然后调用 OCR"作为工具调用。工具返回文本；VLM 继续推理。

此模式泛化：set-of-mark prompting、区域注释和外部检测工具都适用相同的"发出工具调用、接收结构化响应" schema。

### 2026 年基准

- ScreenSpot-Pro。~1k 网页截屏上的 GUI 定位。开放 SOTA Qwen2.5-VL-72B ~85%。前沿 ~90%。
- VisualWebArena。端到端网页任务（购物、论坛、分类广告）。开放 SOTA ~20%。Gemini 3 Pro ~27%。
- AgentVista (arXiv:2602.23166)。2026 年最难基准。12 个域的真实工作流。前沿模型 27-40%；开放模型 10-20%。
- WebArena / WebShop。旧基准；前沿饱和。

### 为什么仍然难

Agent 性能瓶颈：

1. 细尺度视觉定位。"点击小 X"在移动端分辨率上经常失败。
2. 长程规划。10 步动作后 agent 偏离目标。
3. 错误恢复。点击失败（错误按钮）时，检测 + 恢复很少是训练数据。
4. 跨页上下文。在标签页间跳转或长表单丢失状态。

研究方向：记忆架构、显式重新规划、多模态验证（动作成功的截屏匹配）。

### 顶点构建

顶点任务：构建一个 computer-use agent，它：

1. 读取预订网站模拟页面的 HTML + 截屏。
2. 计划多步序列：搜索 → 选择 → 填写表单 → 提交。
3. 发出匹配动作 schema 的 JSON 动作。
4. 在固定 10 任务切片上评估端到端成功率。

课程提供易于扩展到真实浏览器的脚手架代码。

## 使用它

`code/main.py` 是顶点脚手架：

- 动作 schema JSON 定义（10 个动作）。
- 模拟浏览器状态为 dict。
- Agent 循环骨架：接收状态、发出动作、应用、循环。
- 10 任务迷你基准（合成页面）测量端到端成功率。
- 动作失败时的错误恢复钩子。

## 交付它

本课产生 `outputs/skill-multimodal-agent-designer.md`。给定 computer-use 产品（领域、动作集、评估目标），设计完整 agent 循环、记忆策略、定位模式和预期基准分数。

## 练习

1. 用 `screenshot_region` 工具（裁剪 + 缩放）扩展动作 schema。什么任务受益？

2. 阅读 AgentVista (arXiv:2602.23166)。描述最困难的任务类别及为什么前沿模型仍然失败。

3. 长程记忆压缩：设计带 ≤4 个活动截屏的摘要链、任意数量已记录。

4. 构建错误恢复钩子：动作失败（未找到按钮）时，agent 下一步做什么？

5. 在 10 个网页任务上对比纯截屏 Claude 4.7 与混合截屏 + 无障碍树 Qwen2.5-VL。各在什么任务上获胜？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| GUI grounding | "Click coordinates" | 模型在截屏上输出指令目标的 (x,y) |
| Action schema | "Tool definitions" | 有效动作的 JSON 描述（点击、输入、滚动、拖拽） |
| Accessibility tree | "Structured DOM" | 来自浏览器/iOS API 的机器可读 UI 层级 |
| Hybrid agent | "Screenshot + tree" | 同时使用图像和结构化信息；比单独任一更可靠 |
| Visual tool use | "Zoom/crop/detect" | Agent 在计划中调用外部视觉工具（OCR、检测） |
| Summary-chain | "Memory compression" | 定期文本摘要替换长截屏历史 |
| VisualWebArena | "E2E web bench" | 2024 端到端网页任务基准 |
| AgentVista | "2026 hard bench" | 12 域真实工作流；甚至 Gemini 3 Pro 也只有 ~30% |

## 延伸阅读

- [Cheng 等人 — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong 等人 — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You 等人 — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh 等人 — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
