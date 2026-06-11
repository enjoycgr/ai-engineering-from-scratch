---
name: workbench-pack
description: 生成一个经过项目调优的即插即用 agent-workbench-pack（智能体工作台包）—— 规则根据团队历史打磨得更锋利，范围 glob 匹配仓库结构，评分维度增加一条领域专属条目。
version: 1.0.0
phase: 14
lesson: 42
tags: [capstone, workbench-pack, installer, schemas, drop-in]
---

给定一个代码库、团队的 incident（事故）历史以及运行在其中的智能体产品，输出一份调优后的 agent-workbench-pack（智能体工作台包）和一条安装器。

产出：

1. `agent-workbench-pack/` 目录，匹配正典布局：AGENTS.md、docs/、schemas/、scripts/、bin/、README.md、VERSION。
2. 一条 `bin/install.sh`，在没有 `--force` 时拒绝覆盖已有 pack（包），并把 `.workbench-version` 写入目标仓库。
3. 经项目调优的 `agent-rules.md`（至少包含一条根据团队最近六起事故推导出的规则）、`reviewer-rubric.md`（增加第六条领域维度）和 `scope_contract.schema.json`（包含项目专属 glob）。
4. 一条 `lint_pack.py` 脚本，在脚本和 schema（模式）之间或 `VERSION` 与 schema（模式）的 `schema_version` 之间出现漂移时失败。
5. 可选的 CI 集成，在演示分支上安装 pack（包）并对已知良好任务运行验证门。

硬性拒绝：

- 包含项目专属任务的 pack（包）。任务活在目标仓库的任务板上。
- 绑定到单一厂商 SDK 的 pack（包）。仅接受框架无关的 pack（包）；SDK 接入是目标仓库的职责。
- 会修改状态文件的安装器。安装器只幂等地操作 surface（工作面）；状态属于智能体和人。
- 没有对应检查函数的规则。空想的规则属于入门文档，不属于 pack（包）。

拒绝规则：

- 如果事故历史为空，拒绝交付调优的 `agent-rules.md`。使用正典默认值并暴露这一缺口。
- 如果目标仓库的 CI 与安装不兼容（没有 `.github/workflows/`，也没有等价物），拒绝可选 CI 步骤并记录手动路径。
- 如果团队使用 pack（包）的私有 fork，拒绝撰写公开安装器。私有安装器携带私有不变量。

输出结构：

```
agent-workbench-pack/
├── AGENTS.md
├── docs/
├── schemas/
├── scripts/
├── bin/install.sh
├── lint_pack.py
├── VERSION
└── README.md
```

结尾附上 "what to read next"，指向：

- 第 41 课，本 pack（包）所改进的前后对比基准。
- 第 30 课（Eval-Driven Agent Development），消费 pack（包）裁决结果的评估循环。
- [SkillKit](https://github.com/rohitg00/skillkit)，在 32 个 AI 智能体上分发 pack（包）。
