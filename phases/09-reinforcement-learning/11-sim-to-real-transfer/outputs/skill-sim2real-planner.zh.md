---
name: sim2real-planner
description: 为给定机器人 + 任务规划仿真到真实迁移管道，涵盖 DR、SI 和安全。
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

给定机器人平台、任务和真实硬件时间访问，输出：

1. 现实差距清单 (Reality gap inventory)。按预期影响排序的疑似来源（接触、感知、驱动延迟、视觉）。
2. DR 参数 (DR parameters)。精确列表、范围、分布。每个范围对照真实测量论证。
3. SI 步骤 (SI steps)。测量哪些参数；测量方法。
4. 教师/学生分离 (Teacher/student split)。教师使用什么特权信息；学生使用什么观察。
5. 安全包络 (Safety envelope)。低层限制、紧急停止、备用控制器。

拒绝没有 (a) 零样本仿真变体测试、(b) 安全盾、(c) 回滚计划就部署。将任何 DR 范围宽于测量真实变异性 3× 的标记为可能过度随机化。
