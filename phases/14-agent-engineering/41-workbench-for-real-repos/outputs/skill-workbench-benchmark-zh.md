---
name: workbench-benchmark
description: 在同一个项目的自有示例应用上，让同一项任务分别走过仅提示词和工作台引导两条流水线，产出一份包含五项结果的前后对比报告。
version: 1.0.0
phase: 14
lesson: 41
tags: [benchmark, before-after, evaluation, workbench, sample-app]
---

给定一个代码库、一个智能体产品和一份小型示例应用 sample app（示例应用），生成一个可移植的评估 harness（框架），对比仅提示词 prompt-only（仅提示词）和工作台引导 workbench-guided（工作台引导）两条流水线。

产出：

1. `eval/sample_app/` — 从项目领域抽取的最小可用示例应用。
2. `eval/run_prompt_only.py` 和 `eval/run_workbench.py`，各接收一项任务描述并返回 `TaskOutcome`。
3. `eval/report.py`，运行两条流水线并写入 `before-after-report.md` 和 `comparison.json`。
4. 当工作台 outcome（结果）在固定任务套件上退化时导致 CI 失败的 workflow。
5. `docs/benchmark.md`，解释五项 outcome（结果）以及什么算退化。

硬性拒绝：

- 只有一条流水线的 benchmark（基准）。对比才是核心。
- 没有分母的百分比 outcome（结果）。始终报告 `n / m`。
- 智能体产品训练过的示例应用。使用领域定制的 fixture（固定装置）。
- 隐藏 false negatives（假阴性）的报告。必须枚举仅提示词更快的任务。

拒绝规则：

- 如果项目没有验收命令，拒绝交付 benchmark（基准）。没有可测量的东西。
- 如果工作台流水线在典型任务上的耗时超过仅提示词流水线的 3 倍，暴露这一发现；需要简化的是工作台，而不是模型。
- 如果 harness（框架）无法离线运行，拒绝接入 CI。网络抖动会腐蚀对比结果。

输出结构：

```
<repo>/
├── eval/
│   ├── sample_app/
│   ├── run_prompt_only.py
│   ├── run_workbench.py
│   └── report.py
├── outputs/eval/
│   ├── before-after-report.md
│   └── comparison.json
├── docs/benchmark.md
└── .github/workflows/benchmark.yml
```

结尾附上 "what to read next"，指向：

- 第 42 课，打包了工作台流水线所用全部 surface（工作面）的 capstone pack（结业包）。
- 第 19 课（SWE-bench、GAIA、AgentBench），本 benchmark（基准）所补充的宏观基准。
- 第 30 课（Eval-Driven Agent Development），benchmark（基准）接入后的持续评估循环。
