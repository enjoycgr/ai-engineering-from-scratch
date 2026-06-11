---
name: verification-gate
description: 生成一个确定性验证门（verification gate），将范围（scope）、规则（rule）和反馈（feedback）工件组合为每个任务一份 verification_report.json，以及拒绝在无绿色裁决（green verdict）时合并的 CI 接线。
version: 1.0.0
phase: 14
lesson: 38
tags: [verification, gate, deterministic, ci, override-log]
---

给定项目的验收标准和现有工作台工件，生成验证门（verification gate）和覆盖审计日志。

产出：

1. `tools/verify_agent.py`，暴露 `verify(task_id, artifacts) -> VerdictReport`。纯函数，确定性，无 LLM 调用。
2. `outputs/verification/<task_id>.json` 作为单一真相来源裁决。
3. `tools/override.py`，将签名覆盖条目追加到 `outputs/verification/overrides.jsonl`（必须包含原因、用户 ID、时间戳、发现代码）。
4. 在 `passed: false` 时失败并内联显示报告的 CI 工作流。
5. `docs/verification.md`，列出每项检查、其严重性、其来源工件和覆盖策略。

硬性拒绝：

- 调用 LLM 的检查。门是确定性管道；LLM 判断属于审查者。
- 智能体可以不经签名条目就走的覆盖路径。覆盖仅限人类。
- 省略其消费的工件路径的验证报告。报告必须可审计。
- 工作流可以静默降级阻断严重性发现的检查。严重性在写入时固定，而非读取时。

拒绝规则：

- 如果项目没有验收命令，拒绝交付门，直到存在一个。证明不了什么的门是表演。
- 如果规则报告不存在，拒绝跳过规则检查；失败关闭（fail closed）。
- 如果反馈日志不存在，拒绝跳过验收检查；缺失日志本身就是阻断。
- 如果覆盖条目不受版本控制，拒绝接入覆盖路径；非正式覆盖会击败门。

输出结构：

```
<repo>/
├── tools/
│   ├── verify_agent.py
│   └── override.py
├── outputs/verification/
│   ├── overrides.jsonl
│   └── <task_id>.json
├── docs/verification.md
└── .github/workflows/verify.yml
```

以"what to read next"结尾，指向：

- Lesson 39，了解在绿色裁决后接手的审阅者智能体（reviewer agent）。
- Lesson 40，了解将裁决包含在包中的交接生成器（handoff generator）。
- Lesson 41，了解在真实风格示例应用上运行门。
