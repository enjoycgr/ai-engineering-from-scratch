# Benchmarks: WebArena and OSWorld

> WebArena 在四个自托管应用上测试 web-agent (网页智能体) 能力。OSWorld 在 Ubuntu、Windows、macOS 上测试 desktop-agent (桌面智能体) 能力。在发布时（2023–2024），两者都显示了 best-in-class agent (顶尖智能体) 与人类之间的巨大差距。差距正在缩小；failure mode (失效模式) 没有改变。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 分钟

## Learning Objectives

- 描述 WebArena 的四个自托管应用以及为什么 execution-based evaluation (基于执行的评估) 重要。
- 解释为什么 OSWorld 使用真实 OS screenshot (操作系统截图) 而非 accessibility APIs (无障碍 API)。
- 说出 OSWorld 的两个主要 failure mode (失效模式)：GUI grounding (GUI grounding/ grounding) 和 operational knowledge (操作知识)。
- 总结 OSWorld-G 和 OSWorld-Human 在基础 benchmark (基准测试) 之上添加了什么。

## The Problem

通用 agent (智能体) 可以调用工具。它们能否在浏览器中点击 20 次完成购物结账？能否仅使用键盘和鼠标配置 Linux 机器？这些是 WebArena 和 OSWorld 回答的问题。

## The Concept

### WebArena（Zhou et al., ICLR 2024）

- 812 个 long-horizon (长期) 任务跨越四个自托管 web apps (网页应用)：购物站点、论坛、类 GitLab 开发工具、商业 CMS (内容管理系统)。
- 加上工具：map (地图)、calculator (计算器)、scratchpad (草稿本)。
- 通过 gym APIs 进行 execution-based evaluation (基于执行的评估)——订单是否已下单、issue (问题) 是否已关闭、CMS page (页面) 是否已更新？
- 发布时：最佳 GPT-4 agent (智能体) 成功率 14.41% vs 人类 78.24%。

自托管的 framing (框架) 很重要——benchmark (基准测试) 不会 flaky (不稳定)，因为目标应用是固定且可重现的。

### Extensions (扩展)

- **VisualWebArena** —— visually grounded tasks (视觉 grounded 任务)，成功取决于解释图像（screenshot (截图) 作为 first-class observation (一级观察)）。
- **TheAgentCompany**（2024 年 12 月）—— 添加 terminal (终端) + coding (编码)；更像真实的 remote-work (远程工作) 环境。

### OSWorld（Xie et al., NeurIPS 2024）

- 369 个真实计算机任务跨越 Ubuntu、Windows、macOS。
- 对真实应用的自由键盘和鼠标控制。
- 1920×1080 screenshots (截图) 作为 observation (观察)。
- 发布时：最佳模型 12.24% vs 人类 72.36%。

### Primary failure modes (主要失效模式)

1. **GUI grounding (GUI grounding/ grounding)。** Pixel (像素) → element (元素) mapping (映射)。模型难以在 1920×1080 中可靠地 localize (定位) UI elements (元素)。
2. **Operational knowledge (操作知识)。** 哪个菜单有该设置、哪个 keyboard shortcut (键盘快捷键)、哪个 preference pane (首选项面板)。人类用多年积累的知识 tail (尾部)。

### Follow-ups (后续)

- **OSWorld-G** —— 564-sample grounding suite ( grounding 套件) + Jedi training set (训练集)。将 grounding ( grounding) 与 planning (规划) 分解，使你可以分别测量它们。
- **OSWorld-Human** —— 手动精选的 gold action trajectories (黄金动作轨迹)。显示顶尖 agent (智能体) 使用的步骤比必要多 1.4-2.7 倍（trajectory-efficiency gap (轨迹效率差距)）。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（Lesson 21）都在 WebArena 和 OSWorld 塑造的工作负载上训练。Benchmarks (基准测试) 是目标；生产模型是交付的答案。

### Benchmarking (基准测试) 何时出错

- **Screenshot-only evals (仅截图评估)。** OSWorld 是 screenshot-driven (截图驱动) 的；在 OSWorld 上评估使用 DOM 或 accessibility APIs (无障碍 API) 的 agent (智能体) 会错过 grounding challenge ( grounding 挑战)。
- **Ignoring trajectory length (忽略轨迹长度)。** 只评分 success rate (成功率) 会错过 OSWorld-Human 揭示的 1.4-2.7x step inefficiency (步骤低效)。
- **Stale self-hosted apps (过时的自托管应用)。** WebArena 的应用 pin (固定) 特定版本；未经重新精选就更新会破坏 comparability (可比性)。

## Build It

`code/main.py` 实现了一个玩具级 web-agent harness (网页智能体工具链)：

- 一个最小化的"shopping app (购物应用)" state machine (状态机)：list_items (列出商品)、add_to_cart (加入购物车)、checkout (结账)。
- 3 个任务的 gold trajectories (黄金轨迹)。
- 一个脚本化 agent (智能体) 尝试每个任务。
- Execution-based evaluator (基于执行的评估器)（state check (状态检查)）和 trajectory-efficiency metric (轨迹效率指标)（steps vs gold (步骤对比黄金)）。

运行方式：

```
python3 code/main.py
```

输出：per-task success rate (每个任务的成功率) 和 trajectory efficiency (轨迹效率)，镜像 OSWorld-Human 的方法论。

## Use It

- 在内部集群上自托管 **WebArena Verified** 进行持续评估。
- 在 VM fleet (虚拟机集群) 中运行 **OSWorld** 进行桌面 agent (智能体) 评估。
- **Computer-use agent (计算机使用智能体)**（Lesson 21）—— Claude、OpenAI CUA、Gemini —— 都在这类工作负载上训练。
- **你自己的产品流** —— 为你的前 20 个任务捕获 gold trajectories (黄金轨迹)；每周对它们运行 agent (智能体)。

## Ship It

`outputs/skill-web-desktop-harness.md` 构建了一个 WebArena/OSWorld-style harness (类 WebArena/OSWorld 工具链)，带有 execution-based eval (基于执行的评估) 和 trajectory efficiency metric (轨迹效率指标)。

## Exercises

1. 用第二个应用（论坛）扩展玩具 harness (工具链)。编写 3 个任务加 gold trajectories (黄金轨迹)。
2. 为每个任务添加 trajectory-efficiency reporting (轨迹效率报告)。在你的玩具上，agent (智能体) 是 1x、2x 还是 3x 超过 gold (黄金)？
3. 实现一个"distractor" tool (干扰工具)——gold trajectory (黄金轨迹) 从不使用它。脚本化 agent (智能体) 会被诱惑吗？
4. 阅读 OSWorld-G。如何在你自己的 evals (评估) 中分离 grounding failures ( grounding 失败) 和 planning failures (规划失败)？
5. 阅读 WebArena 的应用 README。升级一个 pin (固定) 的应用版本会破坏什么？

## Key Terms

| Term | 人们怎么说 | 实际含义 |
|------|----------|---------|
| WebArena | "Web agent benchmark (网页智能体基准测试)" | 跨越 4 个自托管应用的 812 个任务；gym-style evaluation (gym 风格评估) |
| VisualWebArena | "Visual WebArena (视觉 WebArena)" | 视觉 grounded (视觉 grounded) 的 WebArena；screenshots (截图) 是 observations (观察) |
| OSWorld | "Desktop agent benchmark (桌面智能体基准测试)" | 真实 Ubuntu/Windows/macOS 上的 369 个任务 |
| GUI grounding (GUI grounding/ grounding) | "Pixel-to-element mapping (像素到元素映射)" | 在 1920x1080 中定位 UI elements (元素) 的模型 |
| Operational knowledge (操作知识) | "OS know-how (操作系统知识)" | 哪个菜单、哪个 shortcut (快捷键)、哪个 preference pane (首选项面板) |
| OSWorld-G | "Grounding suite ( grounding 套件)" | 564 个 grounding-only (仅 grounding) 样本 + training set (训练集) |
| OSWorld-Human | "Gold trajectories (黄金轨迹)" | 手动专家动作序列以测量效率 |
| Trajectory efficiency (轨迹效率) | "Steps over gold (超过黄金的步骤)" | Agent (智能体) 步骤数除以人类最小值 |

## Further Reading

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用网页 benchmark (基准测试)
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨 OS 桌面 benchmark (基准测试)
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的 benchmark-shaped (基准测试形态) 能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数字
