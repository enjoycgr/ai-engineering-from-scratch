# 仿真到现实迁移 (Sim-to-Real Transfer)

> 在模拟器中训练但在硬件上失败的策略，是记住了模拟器的策略。域随机化 (domain randomization)、域自适应 (domain adaptation) 和系统辨识 (system identification) 是使学习到的控制器跨越现实差距的三个工具。

**类型：** 学习
**语言：** Python
**先决条件：** Phase 9 · 08 (PPO)，Phase 2 · 10 (偏差/方差)
**时间：** ~45 分钟

## 问题

训练真实机器人缓慢、危险且昂贵。双足机器人需要数百万训练片段才能学会走路；真实双足即使跌倒一次也会损坏硬件。仿真给你无限的重置、确定性可重复性、并行环境和无物理损伤。

但模拟器是错的。轴承摩擦比 MuJoCo 模型大。相机有模拟器未包含的镜头畸变。电机有延迟、间隙和饱和，99% 的仿真模型跳过这些。风、灰尘和可变光线破坏在无菌渲染上训练的策略。**现实差距**——仿真分布与真实分布之间的系统性差异——是部署 RL 用于机器人学的中心问题。

你需要一个*对仿真到现实分布漂移鲁棒*的策略。三种历史方法：随机化模拟器（域随机化）、用少量真实数据自适应策略（域自适应 / 微调），或辨识真实系统参数并匹配它们（系统辨识）。2026 年主导配方结合所有三种与大规模并行仿真（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）。

## 概念

![三种仿真到现实机制：域随机化、自适应、系统辨识](../assets/sim-to-real.svg)

**域随机化 (Domain Randomization, DR)。** Tobin 等人 2017，Peng 等人 2018。训练期间，随机化每个可能在真实机器人上不同的仿真参数：质量、摩擦系数、电机 PD 增益、传感器噪声、相机位置、光照、纹理、接触模型。策略学习"今天它在哪个模拟器中"的条件分布，并在整个跨度上泛化。如果真实机器人落在训练包络内，策略就有效。

- **优点：** 不需要真实数据。一个配方，多种机器人。
- **缺点：** 过度随机化的训练产生"通用"但过于谨慎的策略。太多噪声 ≈ 太多正则化。

**系统辨识 (System Identification, SI)。** 在训练前将仿真器参数拟合到真实世界数据。如果你能测量真实机器人上的臂关节摩擦，将其插入仿真。然后训练一个期望这些值的策略。需要访问真实系统但直接减少现实差距。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残差模型误差对策略不可见；小的未辨识效应（例如，电机死区）仍然破坏部署。

**域自适应 (Domain Adaptation)。** 在仿真中训练，用少量真实数据微调。两种风味：

- **Real2Sim2Real：** 使用真实展开学习残差模拟器 `f(s, a, z) - f_sim(s, a)`，在修正后的仿真中训练。不用太多真实数据就闭合差距。
- **观察自适应：** 训练一个策略，通过学习的特征提取器（例如 GAN 像素到像素）将真实观察映射到仿真般的观察。控制器留在仿真中。

**特权学习 / 教师-学生。** Miki 等人 2022（ANYmal 四足）。在仿真中训练一个*教师*，它可以访问特权信息（真实摩擦、地形高度、IMU 漂移）。蒸馏一个*学生*，它只看到真实传感器观察。学生从历史中推断特权特征，跨物理参数鲁棒。

**大规模并行仿真。** 2024–2026。Isaac Lab、Mujoco MJX、Brax 都在单个 GPU 上运行数千个并行机器人。PPO 带 4,096 个并行人形机器人在数小时内收集数年经验。"现实差距"随着训练分布变宽而缩小；当这 4,096 个环境中每个都有不同随机化参数时，DR 几乎免费。

**真实世界 2026 配方（四足行走示例）：**

1. 带域随机化重力、摩擦、电机增益、负载的大规模并行仿真。
2. 用特权信息（地形图、身体速度真实值）训练的教师策略。
3. 仅从本体感觉（腿部关节编码器）使用教师蒸馏的学生策略。
4. 可选通过真实 IMU 上的自编码器进行观察自适应。
5. 部署。在 10+ 环境中零样本。如果失败，用安全约束 PPO 进行几分钟真实世界微调。

## 构建

本课的代码是在*嘈杂*转移的 GridWorld 上域随机化的微小演示。我们在"仿真"中训练经历随机化滑倒概率的策略，并在它训练期间从未见过的"真实"滑倒水平上评估。形状直接映射到 MuJoCo 到硬件迁移。

### 步骤 1：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模拟器暴露的参数。在真实机器人学中，它可能是摩擦、质量、电机增益——任何在仿真和真实之间变化的东西。

### 步骤 2：用 DR 训练

每片段开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任何东西。做很多片段。

### 步骤 3：在"真实"滑倒上零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支持内；`0.5` 和 `0.7` 在外。DR 训练的策略应在支持内保持接近最优，并在外优雅退化。固定滑倒训练的策略在其训练滑倒外会很脆弱。

### 步骤 4：与窄训练比较

只用 `slip = 0.0` 训练第二个策略。在相同滑倒扫描上评估。真实滑倒 > 0 时你应该看到灾难性下降。

## 陷阱

- **太多随机化。** 在 `slip ∈ [0, 0.9]` 上训练，你的策略太风险厌恶，从不尝试最优路径。匹配*预期的*真实世界分布，不是"任何事都可能发生。"。
- **太少随机化。** 在薄切片上训练，策略完全不能泛化。使用自适应课程（自动域随机化），随着策略改进加宽分布。
- **错误的参数空间。** 随机化错误的东西（相机色调当真实差距是电机延迟时）DR 不帮助。首先分析真实机器人。
- **特权信息泄漏。** 使用全局状态做动作而非观察的教师，可以产生学生无法赶上的结果。确保教师的策略对学生给定观察历史是可实现的。
- **仿真到仿真迁移失败。** 如果你的策略对更难的仿真变体不鲁棒，它也不会对真实世界鲁棒。部署前总是在留出仿真变体上测试。
- **没有真实世界安全包络。** 在仿真中"有效"且"在真实中有效"而没有低层安全盾的策略仍然可能损坏硬件。在非学习控制器中添加速率限制、扭矩限制、关节限制。

## 应用

2026 年仿真到真实栈：

| 领域 | 栈 |
|--------|-------|
| 腿部运动（ANYmal、Spot、人形） | Isaac Lab + DR + 特权教师 / 学生 |
| 操作（灵巧手、拾取放置） | Isaac Lab + DR + 视觉 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线自适应 |
| 手指/手内操作 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业臂 | MuJoCo-Warp + SI + 少量真实微调 |

对于所有尺度的控制，工作流程是一致的：尽可能拟合仿真，随机化你无法拟合的，训练巨大策略，蒸馏，用安全盾部署。

## 交付

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在固定滑倒 GridWorld（slip=0.0）上训练 Q-learning 智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制回报 vs 滑倒。
2. **中等。** 训练 DR Q-learning 智能体采样 `slip ~ Uniform[0, 0.3]`。评估相同扫描。DR 在 slip=0.5（分布外）买了多少？
3. **困难。** 实现课程：从 slip=0.0 开始，每次策略达到最优的 90% 时加宽 DR 范围。测量达到 slip=0.3 零样本的总环境步数 vs 固定 DR 基线。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 现实差距 (Reality gap) | "仿真到真实的差异" | 训练和部署物理/感知之间的分布漂移。 |
| 域随机化 (Domain randomization, DR) | "跨随机仿真训练" | 训练期间随机化仿真参数以使策略泛化。 |
| 系统辨识 (System identification, SI) | "测量真实并拟合仿真" | 估计真实物理参数；设置仿真以匹配。 |
| 域自适应 (Domain adaptation) | "在真实数据上微调" | 仿真训练后的小真实世界微调；可能自适应观察或动态。 |
| 特权信息 (Privileged info) | "教师的真实值" | 只有仿真有的信息；学生必须从观察历史推断它。 |
| 教师/学生 (Teacher/student) | "蒸馏特权 -> 可观察" | 用捷径训练的教师；学生不用它们学习模仿。 |
| ADR | "自动域随机化" | 随着策略改进加宽 DR 范围的课程。 |
| Real2Sim | "用真实数据闭合差距" | 学习残差以使仿真模仿真实展开。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 原始 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 动力学的 DR，四足运动。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大规模 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的教师-学生。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 驱动 2025–2026 部署的大规模并行仿真。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna 框架（使用模型进行规划 + 展开）支撑现代仿真到真实管道。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 带基准结果的仿真到真实方法分类。
