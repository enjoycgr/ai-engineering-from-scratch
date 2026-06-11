---
name: rollback-rehearsal
description: 为拟议的自主工作流设计回滚演练 (rollback-rehearsal) 测试，并审计 checkpoint 后端是否满足审计追踪持久化要求。
version: 1.0.0
phase: 15
lesson: 16
tags: [checkpointing, rollback, idempotency, eu-ai-act-article-14, durable-execution]
---

给定一个拟议的长时程自主工作流，设计一个回滚演练测试，证明幂等性 + 前置条件 + 验证 + 回滚栈端到端有效，并审计 checkpoint 后端是否满足监管就绪要求。

产出：

1. **演练脚本 (Rehearsal script)。** 具体测试，包含：(a) 启动工作流，(b) 在提交中崩溃，(c) 恢复，(d) 断言动作只执行一次，(e) 注入验证失败，(f) 断言回滚触发且状态恢复。任何生产工作流在至少一次通过此测试前都不应运行。
2. **幂等性审计 (Idempotency audit)。** 确认幂等性键从提案内容派生（第 15 课），提交逻辑使用显式执行状态（`pending` -> `executing` -> `committed`/`failed`）。在副作用前通过幂等性键预留/加锁，并在副作用验证通过后标记 `committed`。
3. **前置条件清单 (Precondition inventory)。** 列出工作流在提交时必须重新检查的每个前置条件。检查时间与使用时间 (time-of-check vs time-of-use) 的间隙是生产中最常见的 bug；前置条件必须在提交时评估，而非在提议时。
4. **验证清单 (Verify inventory)。** 对每个重要动作，命名确认副作用已发生的具体读取操作。"返回了 200" 不可接受。
5. **回滚清单 (Rollback inventory)。** 对每个重要动作，将回滚分类为 in-band（带内）、compensating transaction（补偿事务）或 out-of-band alert（带外告警）。空操作回滚（"我们无法撤销这个"）必须在提案中明确命名（第 15 课元数据）。

硬性否决：
- 没有演练过回滚的工作流。
- 部署时丢失数据的 checkpoint 后端。
- 状态在执行后而非执行前写入的提交路径。
- 仅检查工具调用返回码的 "已验证" 状态。
- 仅在提议时运行而非在提交时运行的前置条件检查。

拒绝规则：
- 如果用户尚未在 staging 中至少运行过一次演练脚本，拒绝生产上线。
- 如果用户无法提供 checkpoint 存储 schema，拒绝并要求先提供 schema 文档。监管者需要可查询的状态。
- 如果工作流依赖内存中的 checkpoint（无持久化），拒绝。

输出格式：

返回一份演练计划，包含：
- **测试脚本大纲**（带断言的步骤）
- **幂等性表**（键组成、状态写入顺序）
- **前置条件表**（检查内容、何时评估、后果）
- **验证表**（动作、确认读取）
- **回滚表**（动作、类型、目标状态）
- **后端认证**（存储、 survives-deploy y/n、 query-ready y/n）
- **就绪度**（production / staging / research-only）
