---
name: vla-action-format-picker
description: 为机器人任务选择离散 bin (RT-2)、FAST tokenization、flow-matching (pi0) 或双系统 (GR00T) 并计算 token 计数和延迟。
version: 1.0.0
phase: 12
lesson: 21
tags: [vla, rt-2, openvla, pi0, groot, action-tokenization]
---

给定机器人任务（操作、导航、类人全身）和延迟/准确率预算，选择动作格式并输出配置。

产出：

1. 格式选择。离散 256-bin（RT-2 / OpenVLA）、FAST DCT 压缩、连续 flow-matching（pi0）、或双系统（GR00T）。
2. Token 计数。每步 DOF × 每维 bin 对离散；每序列 DCT 系数对 FAST；单头输出对 flow。
3. 推理延迟。自回归每步（离散）vs 序列一次解码（flow）vs 双系统（系统 1 快速，系统 2 慢速）。
4. 共微调比率。网页 VQA : 机器人轨迹。OpenVLA ~0.5:1，pi0 类似。
5. 质量上限。OpenVLA 7B 在常见操作 ~75-80%；pi0 类似任务 ~82-85%；GR00T N1 类人 ~70%。
6. 迁移路径。从离散到 flow 当控制频率需要 >30 Hz。

硬性拒绝：
- 为 1 Hz 导航任务提议 flow-matching。离散 bin 足够快且更简单。
- 声称 FAST 无损压缩。DCT 丢弃高频运动细节。
- 推荐仅机器人数据训练。分布外指令失败。

拒绝规则：
- 如果控制频率 <10 Hz 且仅需要常见操作，拒绝 flow 并推荐 OpenVLA（开放、简单、可用）。
- 如果用户需要类人全身控制 (>30 DOF)，拒绝单系统并推荐 GR00T N1。
- 如果用户无法访问 Open X-Embodiment 或等效机器人数据，拒绝任何 VLA 并推荐行为克隆或模仿学习。

输出：一页计划，含格式选择、token 计数、延迟、共微调比率、质量上限、迁移路径。结尾附 arXiv 2307.15818（RT-2）、2406.09246（OpenVLA）、2410.24164（pi0）、2503.14734（GR00T）。
