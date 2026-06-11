---
name: feedback-runner
description: 包装 shell 命令，进行确定性的 stdout/stderr/exit/duration 捕获，为每条命令持久化一条 JSONL 记录，并在反馈缺失时拒绝推进智能体（agent）循环。
version: 1.0.0
phase: 14
lesson: 37
tags: [feedback, subprocess, runner, jsonl, loop-control]
---

给定一个在智能体循环内部运行 shell 命令的项目，生成一个反馈运行器（feedback runner）及其写入的 JSONL。

产出：

1. `tools/run_with_feedback.py`，暴露 `run_with_feedback(command: list[str], agent_note: str, timeout_s: float) -> FeedbackRecord`。
2. `feedback_record.jsonl` 位于工作台之下，每行一条记录。
3. `tools/feedback_loader.py`，返回活跃任务的最近 N 条记录。
4. 智能体循环在声称成功前调用的 `loop_can_advance(record) -> bool` 辅助函数。
5. 测试覆盖：成功路径、非零退出、超时、缺失二进制文件、确定性头/尾截断（deterministic head/tail truncation）。

硬性拒绝：

- 运行器中任何地方出现 `shell=True`。只允许 argv。
- 依赖挂钟或随机采样的截断。相同输入必须产生相同记录。
- 没有 `duration_ms` 的记录。缓慢探查是工作台卡住的首个迹象。
- 返回无界列表的加载器。限制为最近 N 条或分页。

拒绝规则：

- 如果项目通过 stdout 管道传输机密，拒绝在不经过脱敏（redaction）步骤的情况下交付运行器。显示本应被捕获的行。
- 如果项目有可以无限期挂起的命令，拒绝在没有默认超时和显式覆盖列表的情况下交付。
- 如果运行器在共享状态的工作器中运行，拒绝跳过 JSONL 追加周围的文件锁。多个写入者会撕裂文件。

输出结构：

```
<repo>/
├── feedback_record.jsonl
└── tools/
    ├── run_with_feedback.py
    ├── feedback_loader.py
    └── test_feedback_runner.py
```

以"what to read next"结尾，指向：

- Lesson 38，了解消费记录的验证门（verification gate）。
- Lesson 39，了解在为运行打分时读取反馈的审阅者智能体（reviewer agent）。
- Lesson 23，了解 OTel GenAI 约定，以便在反馈稳固后添加到遥测侧。
