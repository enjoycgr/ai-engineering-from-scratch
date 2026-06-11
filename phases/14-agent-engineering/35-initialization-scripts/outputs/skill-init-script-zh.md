---
name: init-script
description: 采访一个项目并发出一个确定性 init_agent.py（初始化代理脚本），包含五个 probes（探测）和一个 CI workflow，如果任何 probe 失败则拒绝启动 agent。
version: 1.0.0
phase: 14
lesson: 35
tags: [init, probes, ci, workbench, fail-loud]
---

给定一个仓库、agent 产品及其 dependency surface，产生 project-specific init script 和 CI wiring。

产出：

1. `tools/init_agent.py`，带有这些 probes：runtime version、listed dependencies、test command resolvability、required env vars、state file freshness。
2. 脚本旁边文档化的 `init_report.json` schema。每个 probe 返回 `(name, status: pass|warn|fail, detail)`。
3. `.github/workflows/agent-init.yml`（或等效文件），运行脚本并在任何 fail-severity probe 上阻塞 agent job。
4. 一个 `pre-task` hook script，agent runtime 可以在每个 session 开始前调用。
5. `docs/init.md` 中的文档，列出每个 probe、其 severity 以及如何修复 failure。

Hard rejects（硬性拒绝）：

- 没有 timeout 就向外调用网络的 probes。Init 必须快速且离线安全。
- 需要 LLM calls 的 probes。Init 是确定性管道。
- 被 wrapper 吞掉的 non-zero exit code。Fail loud 是全部意义。
- 触碰 state 而没有 idempotency 的 probes。连续两次运行必须产生相同的报告，仅 timestamp 不同。

Refusal rules（拒绝规则）：

- 如果项目没有 test command，拒绝 ship 该脚本。将 gap 添加到 workbench audit 中。
- 如果 env var 列表包含脚本会打印的 secrets，拒绝并强制 redaction（脱敏）。Init reports 绝不应携带 secrets。
- 如果 probe 在 dry run 中耗时超过三秒，在 ship 前 surface timing finding。Long probes 将 init 变成 ceremony。

输出结构：

```
<repo>/
├── tools/
│   ├── init_agent.py
│   └── pre_task.sh
├── docs/
│   └── init.md
└── .github/
    └── workflows/
        └── agent-init.yml
```

结尾附上 "what to read next"，指向：

- Lesson 36 了解使用 init report 的 `repo_paths` 的 per-task scope contract。
- Lesson 37 了解消费 resolved test command 的 runtime feedback loop。
- Lesson 38 了解依赖于 probes 通过的 verification gate。
